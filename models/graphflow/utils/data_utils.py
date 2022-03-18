# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import io
import torch
import numpy as np
from scipy.sparse import *
from collections import Counter, defaultdict

from torch.utils.data import Dataset

from .bert_utils import *
from .eval_utils import normalize_text
from .process_utils import WhitespaceTokenizer
from . import constants as Constants
from .timer import Timer


################################################################################
# Dataset Prep #
################################################################################

def prepare_datasets(config):
    train_set = None if config['trainset'] is None else QADataset(config['trainset'], config)
    dev_set = None if config['devset'] is None else QADataset(config['devset'], config)
    test_set = None if config['testset'] is None else QADataset(config['testset'], config)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}

################################################################################
# Dataset Classes #
################################################################################

class QADataset(Dataset):
    """QA dataset."""

    def __init__(self, filename, config):
        self.filename = filename
        self.config = config
        paragraph_lens = []
        turn_num = []
        self.paragraphs = []
        self.vocab = Counter()
        dataset = read_json(filename)
        backgrounds = self._get_backgrounds()
        for paragraph in dataset['data']: # Paragraph level
            cid = paragraph['id']
            paragraph_lens.append(len(paragraph['annotated_context']['word']))
            turn_num.append(len(paragraph['qas']))
            # Prepare paragraphs
            para = {'context_id': cid, 'background': backgrounds[cid], 'examples': [], 'gold_answers':[]}
            for qas in paragraph['qas']: # Turn level
                # copy gold answers
                para['gold_answers'].append(qas['answer'])
                # Build vocab
                for w in qas['annotated_question']['word'] \
                        + paragraph['annotated_context']['word'] \
                        + qas['annotated_answer']['word']:
                    self.vocab[w.lower()] += 1 # [Huihan: not accurate if using predicted answer. Should check that in future code.]
                
                qas['annotated_question']['marker'] = []

                # Prepare a question-answer pair
                question = qas['annotated_question']


                answers = qas['additional_answers']

                normalized_answer = normalize_text(qas['answer'])
                sample = {'cid': paragraph['id'],
                          'evidence': paragraph['annotated_context'],
                          'raw_evidence': paragraph['context'],
                          'turn_id': qas['turn_id'],
                          'question': question,
                          'answers': answers,
                          'targets': qas['answer_span'],
                          'span_mask': 0}

                if Constants.QuAC_UNK_ANSWER == normalized_answer:
                    sample['unk_answer_targets'] = 1
                else:
                    sample['unk_answer_targets'] = 0
                    sample['span_mask'] = 1

                if qas['yesno'] == Constants.QuAC_YESNO_YES:
                    sample['yesno_targets'] = Constants.QuAC_YESNO_YES_LABEL
                elif qas['yesno'] == Constants.QuAC_YESNO_NO:
                    sample['yesno_targets'] = Constants.QuAC_YESNO_NO_LABEL
                else:
                    sample['yesno_targets'] = Constants.QuAC_YESNO_OTHER_LABEL

                if qas['followup'] == Constants.QuAC_FOLLOWUP_YES:
                    sample['followup_targets'] = Constants.QuAC_FOLLOWUP_YES_LABEL
                elif qas['followup'] == Constants.QuAC_FOLLOWUP_NO:
                    sample['followup_targets'] = Constants.QuAC_FOLLOWUP_NO_LABEL
                else:
                    sample['followup_targets'] = Constants.QuAC_FOLLOWUP_OTHER_LABEL

                
                para['examples'].append(sample)
            para['id'] = paragraph['id']
            para['evidence'] = paragraph['annotated_context']

            if self.config['predict_raw_text']:
                para['raw_evidence'] = paragraph['context']
            self.paragraphs.append(para)
    
    def _get_backgrounds(self, filename='/n/fs/nlp-huihanl/conversational-qa/local/Bert4QuAC/val_v0.2.json'):
        dataset = read_json(filename)
        backgrounds = {}
        for paragraph in dataset['data']:  # Paragraph level
            cid = paragraph['paragraphs'][0]['id']
            background = paragraph['background']
            backgrounds[cid] = background
        return backgrounds

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return self.paragraphs[idx]

################################################################################
# Read & Write Helper Functions #
################################################################################


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with io.open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def log_json(data, filename, mode='w', encoding='utf-8'):
    with io.open(filename, mode, encoding=encoding) as outfile:
        outfile.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_processed_file_contents(file_path, encoding="utf-8"):
    contents = get_file_contents(file_path, encoding=encoding)
    return contents.strip()

################################################################################
# DataLoader Helper Functions #
################################################################################

def sanitize_input(sample_batch, config, vocab, feature_dict, bert_tokenizer, training=True):
    """
    Reformats sample_batch for easy vectorization.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        vocab: word embedding dictionary.
        feature_dict: the features we want to concatenate to our embeddings.
        train: train or test?
    """
    sanitized_batch = defaultdict(list)
    batch_graphs = []
    for paragraph in sample_batch:
        # print("Paragraph:", paragraph)
        if 'id' in paragraph:
            sanitized_batch['id'].append(paragraph['id'])
        evidence = paragraph['evidence']['word']
        processed_e = [vocab[w.lower()] if w.lower() in vocab else vocab[Constants._UNK_TOKEN] for w in evidence]
        sanitized_batch['doc_token'].append(evidence)
        sanitized_batch['evidence'].append(processed_e)

        if config.get('static_graph', None):
            batch_graphs.append(paragraph['evidence']['graph'])

        if config['f_tf']:
            sanitized_batch['evidence_tf'].append(compute_tf(evidence))

        if config['predict_raw_text']:
            sanitized_batch['raw_evidence_text'].append(paragraph['raw_evidence'])
            sanitized_batch['offsets'].append(paragraph['evidence']['offsets'])
        else:
            sanitized_batch['evidence_text'].append(evidence)

        para_turn_ids = []
        para_ques = []
        para_ques_marker = []
        para_bert_ques_features = []
        para_features = []
        para_targets = []
        para_span_mask = []

        para_unk_answer_targets = []
        para_yesno_targets = []
        para_followup_targets = []

        para_answers = []
        for ex in paragraph['turns']:
            para_turn_ids.append(ex['turn_id'])
            question = ex['question']['word']
            processed_q = [vocab[w.lower()] if w.lower() in vocab else vocab[Constants._UNK_TOKEN] for w in question]
            para_ques.append(processed_q)
            para_ques_marker.append(ex['question']['marker'])

            if config['use_bert']:
                bert_ques_features = convert_text_to_bert_features(question, bert_tokenizer, config['bert_max_seq_len'], config['bert_doc_stride'])
                para_bert_ques_features.append(bert_ques_features)

            # featurize evidence document:
            para_features.append(featurize(ex['question'], paragraph['evidence'], feature_dict))
            para_targets.append(ex['targets'])
            para_span_mask.append(ex['span_mask'])
            para_answers.append(ex['answers'])

            para_unk_answer_targets.append(ex['unk_answer_targets'])
            para_yesno_targets.append(ex['yesno_targets'])
            para_followup_targets.append(ex['followup_targets'])

        sanitized_batch['question'].append(para_ques)
        sanitized_batch['question_marker'].append(para_ques_marker)
        if config['use_bert']:
            bert_evidence_features = convert_text_to_bert_features(evidence, bert_tokenizer, config['bert_max_seq_len'], config['bert_doc_stride'])
            sanitized_batch['bert_evidence_features'].append(bert_evidence_features)
            sanitized_batch['bert_question_features'].append(para_bert_ques_features)

        sanitized_batch['turn_ids'].append(para_turn_ids)
        sanitized_batch['features'].append(para_features)
        sanitized_batch['targets'].append(para_targets)
        sanitized_batch['span_mask'].append(para_span_mask)
        sanitized_batch['answers'].append(para_answers)
        sanitized_batch['unk_answer_targets'].append(para_unk_answer_targets)
        sanitized_batch['yesno_targets'].append(para_yesno_targets)
        sanitized_batch['followup_targets'].append(para_followup_targets)

    if config.get('static_graph', None):
        batch_graphs = cons_batch_graph(batch_graphs)
        sanitized_batch['evidence_graphs'] = vectorize_batch_graph(batch_graphs)

    return sanitized_batch

def vectorize_input(batch, config, bert_model, training=True, device=None):
    """
    - Vectorize question and question mask
    - Vectorize evidence documents, mask and features
    - Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['question'])

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for para_q in batch['question'] for q in para_q])
    max_turn_len = max([len(para_q) for para_q in batch['question']])
    xq = torch.LongTensor(batch_size, max_turn_len, max_q_len).fill_(0)
    xq_len = torch.LongTensor(batch_size, max_turn_len).fill_(1)
    num_turn = torch.LongTensor(batch_size).fill_(0)
    if config['use_ques_marker']:
        xq_f = torch.LongTensor(batch_size, max_turn_len, max_q_len).fill_(0)

    for i, para_q in enumerate(batch['question']):
        num_turn[i] = len(para_q)
        for j, q in enumerate(para_q):
            xq[i, j, :len(q)].copy_(torch.LongTensor(q))
            if config['use_ques_marker']:
                xq_f[i, j, :len(q)].copy_(torch.LongTensor(batch['question_marker'][i][j]))
            xq_len[i, j] = len(q)

    # Part 2: Document Words
    max_d_len = max([len(d) for d in batch['evidence']])
    xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
    xd_len = torch.LongTensor(batch_size).fill_(1)


    # 2(a): fill up DrQA section variables
    if config['f_tf']:
        xd_tf = torch.Tensor(batch_size, max_d_len).fill_(0)
        for i, d in enumerate(batch['evidence_tf']):
            xd_tf[i, :len(d)].copy_(torch.Tensor(d))

    xd_f = {}
    for i, d in enumerate(batch['evidence']):
        xd[i, :len(d)].copy_(torch.LongTensor(d))
        xd_len[i] = len(d)
        # Context features
        for j, para_features in enumerate(batch['features'][i]):
            for feat_key, feat_val in para_features.items():
                if not feat_key in xd_f:
                    xd_f[feat_key] = torch.zeros(batch_size, max_turn_len, max_d_len, dtype=torch.long)
                xd_f[feat_key][i, j, :len(d)].copy_(feat_val)

    # Part 3: Target representations
    targets = torch.LongTensor(batch_size, max_turn_len, 2).fill_(-100)
    for i, _target in enumerate(batch['targets']):
        for j in range(len(_target)):
            targets[i, j, 0] = _target[j][0]
            targets[i, j, 1] = _target[j][1]

    # Part 4: UNK/YES/NO answer masks
    span_mask = torch.Tensor(batch_size, max_turn_len).fill_(0)
    for i, _span_mask in enumerate(batch['span_mask']):
        for j in range(len(_span_mask)):
            span_mask[i, j] = _span_mask[j]


    unk_answer_targets = torch.Tensor(batch_size, max_turn_len).fill_(-100)
    yesno_targets = torch.LongTensor(batch_size, max_turn_len).fill_(-100)
    followup_targets = torch.LongTensor(batch_size, max_turn_len).fill_(-100)
    for i, _unk_answer_target in enumerate(batch['unk_answer_targets']):
        for j in range(len(_unk_answer_target)):
            unk_answer_targets[i, j] = _unk_answer_target[j]
            yesno_targets[i, j] = batch['yesno_targets'][i][j]
            followup_targets[i, j] = batch['followup_targets'][i][j]

    # Part 5: Previous answer markers
    if config['n_history'] > 0:
        if config['answer_marker_embed_dim'] != 0:
            xd_answer_marker = torch.LongTensor(batch_size, max_turn_len, max_d_len, config['n_history']).fill_(0)
            for i, _target in enumerate(batch['targets']):
                for j in range(len(_target)):
                    if _target[j][0] >= 0 and _target[j][1] >= 0:
                        for prev_answer_distance in range(config['n_history']):
                            turn_id = j + prev_answer_distance + 1
                            if turn_id < len(_target):
                                mark_prev_answer(_target[j][0], _target[j][1], xd_answer_marker[i, turn_id, :, prev_answer_distance], prev_answer_distance)

    # Part 6: Extract features from pretrained BERT models
    if config['use_bert']:
        with torch.set_grad_enabled(False):
            # Question words
            max_bert_q_num_chunks = max([len(para_bert_q) for ex_bert_q in batch['bert_question_features'] for para_bert_q in ex_bert_q])
            max_bert_q_len = max([len(bert_q.input_ids) for ex_bert_q in batch['bert_question_features'] for para_bert_q in ex_bert_q for bert_q in para_bert_q])
            bert_xq = torch.LongTensor(batch_size, max_turn_len, max_bert_q_num_chunks, max_bert_q_len).fill_(0)
            bert_xq_mask = torch.LongTensor(batch_size, max_turn_len, max_bert_q_num_chunks, max_bert_q_len).fill_(0)
            for i, ex_bert_q in enumerate(batch['bert_question_features']):
                for t, para_bert_q in enumerate(ex_bert_q):
                    for j, bert_q in enumerate(para_bert_q):
                        bert_xq[i, t, j, :len(bert_q.input_ids)].copy_(torch.LongTensor(bert_q.input_ids))
                        bert_xq_mask[i, t, j, :len(bert_q.input_mask)].copy_(torch.LongTensor(bert_q.input_mask))
            if device:
                bert_xq = bert_xq.to(device)
                bert_xq_mask = bert_xq_mask.to(device)

            layer_indexes = list(range(config['bert_layer_indexes'][0], config['bert_layer_indexes'][1]))
            all_encoder_layers, _ = bert_model(bert_xq.view(-1, bert_xq.size(-1)), token_type_ids=None, attention_mask=bert_xq_mask.view(-1, bert_xq_mask.size(-1)))
            torch.cuda.empty_cache()
            all_encoder_layers = torch.stack([x.view(bert_xq.shape + (-1,)) for x in all_encoder_layers], 0).detach()
            all_encoder_layers = all_encoder_layers[layer_indexes]
            bert_xq_f = extract_bert_ques_hidden_states(all_encoder_layers, max_q_len, batch['bert_question_features'], weighted_avg=config['use_bert_weight'])
            torch.cuda.empty_cache()

            # Document words
            max_bert_d_num_chunks = max([len(ex_bert_d) for ex_bert_d in batch['bert_evidence_features']])
            max_bert_d_len = max([len(bert_d.input_ids) for ex_bert_d in batch['bert_evidence_features'] for bert_d in ex_bert_d])
            bert_xd = torch.LongTensor(batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
            bert_xd_mask = torch.LongTensor(batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
            for i, ex_bert_d in enumerate(batch['bert_evidence_features']): # Example level
                for j, bert_d in enumerate(ex_bert_d): # Chunk level
                    bert_xd[i, j, :len(bert_d.input_ids)].copy_(torch.LongTensor(bert_d.input_ids))
                    bert_xd_mask[i, j, :len(bert_d.input_mask)].copy_(torch.LongTensor(bert_d.input_mask))
            if device:
                bert_xd = bert_xd.to(device)
                bert_xd_mask = bert_xd_mask.to(device)
            all_encoder_layers, _ = bert_model(bert_xd.view(-1, bert_xd.size(-1)), token_type_ids=None, attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)))
            torch.cuda.empty_cache()
            all_encoder_layers = torch.stack([x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers], 0).detach()
            all_encoder_layers = all_encoder_layers[layer_indexes]
            bert_xd_f = extract_bert_ctx_hidden_states(all_encoder_layers, max_d_len, batch['bert_evidence_features'], weighted_avg=config['use_bert_weight'])
            torch.cuda.empty_cache()

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'answers': batch['answers'],
                   'doc_token': batch['doc_token'],
                   'xq': xq.to(device) if device else xq,
                   'xq_len': xq_len.to(device) if device else xq_len,
                   'xd': xd.to(device) if device else xd,
                   'xd_len': xd_len.to(device) if device else xd_len,
                   'num_turn': num_turn.to(device) if device else num_turn,
                   'targets': targets.to(device) if device else targets,
                   'span_mask': span_mask.to(device) if device else span_mask}

        if config.get('static_graph', None):
            example['xd_graphs'] = batch['evidence_graphs']

        if config['f_tf']:
            example['xd_tf'] = xd_tf.to(device) if device else xd_tf

        example['unk_answer_targets'] = unk_answer_targets.to(device) if device else unk_answer_targets
        example['yesno_targets'] = yesno_targets.to(device) if device else yesno_targets
        example['followup_targets'] = followup_targets.to(device) if device else followup_targets

        if config['predict_raw_text']:
            example['raw_evidence_text'] = batch['raw_evidence_text']
            example['offsets'] = batch['offsets']
        else:
            example['evidence_text'] = batch['evidence_text']

        if config['use_bert']:
            example['bert_xq_f'] = bert_xq_f
            example['bert_xd_f'] = bert_xd_f

        if device:
            for feat_key in xd_f:
                xd_f[feat_key] = xd_f[feat_key].to(device)
        example['xd_f'] = xd_f

        if config['n_history'] > 0:
            if config['answer_marker_embed_dim'] != 0:
                example['xd_answer_marker'] = xd_answer_marker.to(device) if device else xd_answer_marker
            if config['use_ques_marker']:
                example['xq_f'] = xq_f.to(device) if device else xq_f
        return example


def vectorize_input_turn(batch, config, bert_model, QA_history, turn, device=None):
    """
    - Vectorize question and question mask
    - Vectorize evidence documents, mask and features
    - Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters: batch_size=1 for always
    batch_size = len(batch['question'])
    # batch_size = max_conv_len

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for para_q in batch['question'] for q in para_q])
    max_turn_len = max([len(para_q) for para_q in batch['question']])
    # max_turn_len = len(QA_history+1)
    # max_turn_len=max_conv_len
    xq = torch.LongTensor(batch_size, max_turn_len, max_q_len).fill_(0)
    xq_len = torch.LongTensor(batch_size, max_turn_len).fill_(1)
    num_turn = torch.LongTensor(batch_size).fill_(0)
    # num_turn = torch.LongTensor(batch_size).fill_(0)
    if config['use_ques_marker']:
        xq_f = torch.LongTensor(batch_size, max_turn_len, max_q_len).fill_(0)

    for i, para_q in enumerate(batch['question']):
        num_turn[i] = len(para_q)
        for j, q in enumerate(para_q):
            xq[i, j, :len(q)].copy_(torch.LongTensor(q))
            if config['use_ques_marker']:
                xq_f[i, j, :len(q)].copy_(
                    torch.LongTensor(batch['question_marker'][i][j]))
            xq_len[i, j] = len(q)

    # Part 2: Document Words
    max_d_len = max([len(d) for d in batch['evidence']])
    xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
    xd_len = torch.LongTensor(batch_size).fill_(1)

    # 2(a): fill up DrQA section variables
    if config['f_tf']:
        xd_tf = torch.Tensor(batch_size, max_d_len).fill_(0)
        for i, d in enumerate(batch['evidence_tf']):
            xd_tf[i, :len(d)].copy_(torch.Tensor(d))

    xd_f = {}
    for i, d in enumerate(batch['evidence']):
        xd[i, :len(d)].copy_(torch.LongTensor(d))
        xd_len[i] = len(d)
        # Context features
        for j, para_features in enumerate(batch['features'][i]):
            for feat_key, feat_val in para_features.items():
                if not feat_key in xd_f:
                    xd_f[feat_key] = torch.zeros(
                        batch_size, max_turn_len, max_d_len, dtype=torch.long)
                xd_f[feat_key][i, j, :len(d)].copy_(feat_val)

    # Part 3: Target representations
    targets = torch.LongTensor(batch_size, max_turn_len, 2).fill_(-100)
    for i, _target in enumerate(batch['targets']):
        for j in range(len(_target)):
            targets[i, j, 0] = _target[j][0]
            targets[i, j, 1] = _target[j][1]

    # Part 4: UNK/YES/NO answer masks
    span_mask = torch.Tensor(batch_size, max_turn_len).fill_(0)
    for i, _span_mask in enumerate(batch['span_mask']):
        for j in range(len(_span_mask)):
            span_mask[i, j] = _span_mask[j]

    unk_answer_targets = torch.Tensor(batch_size, max_turn_len).fill_(-100)
    yesno_targets = torch.LongTensor(batch_size, max_turn_len).fill_(-100)
    followup_targets = torch.LongTensor(
        batch_size, max_turn_len).fill_(-100)
    for i, _unk_answer_target in enumerate(batch['unk_answer_targets']):
        for j in range(len(_unk_answer_target)):
            unk_answer_targets[i, j] = _unk_answer_target[j]
            yesno_targets[i, j] = batch['yesno_targets'][i][j]
            followup_targets[i, j] = batch['followup_targets'][i][j]

    # Part 5: Previous answer markers
    if config['n_history'] > 0:
        if config['answer_marker_embed_dim'] != 0:
            xd_answer_marker = torch.LongTensor(
                batch_size, max_turn_len, max_d_len, config['n_history']).fill_(0)
            for prev_answer_distance in range(config['n_history']):
                hist_turn_id = turn - prev_answer_distance - 1
                if hist_turn_id >=0:
                    if QA_history[hist_turn_id][2] >= 0 and QA_history[hist_turn_id][3] >= 0: # only append valid answer?
                        print("History:", QA_history[hist_turn_id])
                        mark_prev_answer(QA_history[hist_turn_id][2], QA_history[hist_turn_id][3], xd_answer_marker[0, 0, :, prev_answer_distance], prev_answer_distance)

        #TODO check this!
    # Part 6: Extract features from pretrained BERT models
    if config['use_bert']:
        with torch.set_grad_enabled(False):
            # Question words
            max_bert_q_num_chunks = max(
                [len(para_bert_q) for ex_bert_q in batch['bert_question_features'] for para_bert_q in ex_bert_q])
            max_bert_q_len = max([len(bert_q.input_ids) for ex_bert_q in batch['bert_question_features']
                                  for para_bert_q in ex_bert_q for bert_q in para_bert_q])
            bert_xq = torch.LongTensor(
                batch_size, max_turn_len, max_bert_q_num_chunks, max_bert_q_len).fill_(0)
            bert_xq_mask = torch.LongTensor(
                batch_size, max_turn_len, max_bert_q_num_chunks, max_bert_q_len).fill_(0)
            for i, ex_bert_q in enumerate(batch['bert_question_features']):
                for t, para_bert_q in enumerate(ex_bert_q):
                    for j, bert_q in enumerate(para_bert_q):
                        bert_xq[i, t, j, :len(bert_q.input_ids)].copy_(
                            torch.LongTensor(bert_q.input_ids))
                        bert_xq_mask[i, t, j, :len(bert_q.input_mask)].copy_(
                            torch.LongTensor(bert_q.input_mask))
            if device:
                bert_xq = bert_xq.to(device)
                bert_xq_mask = bert_xq_mask.to(device)

            layer_indexes = list(
                range(config['bert_layer_indexes'][0], config['bert_layer_indexes'][1]))
            output = bert_model(
                bert_xq.view(-1, bert_xq.size(-1)), token_type_ids=None, attention_mask=bert_xq_mask.view(-1, bert_xq_mask.size(-1)), output_hidden_states=True)
            all_encoder_layers = output.hidden_states

            torch.cuda.empty_cache()
            all_encoder_layers = torch.stack(
                [x.view(bert_xq.shape + (-1,)) for x in all_encoder_layers], 0).detach()

            all_encoder_layers = all_encoder_layers[layer_indexes]

            bert_xq_f = extract_bert_ques_hidden_states(
                all_encoder_layers, max_q_len, batch['bert_question_features'], weighted_avg=config['use_bert_weight'])
            torch.cuda.empty_cache()

            # Document words
            max_bert_d_num_chunks = max(
                [len(ex_bert_d) for ex_bert_d in batch['bert_evidence_features']])
            max_bert_d_len = max(
                [len(bert_d.input_ids) for ex_bert_d in batch['bert_evidence_features'] for bert_d in ex_bert_d])
            bert_xd = torch.LongTensor(
                batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
            bert_xd_mask = torch.LongTensor(
                batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
            # Example level
            for i, ex_bert_d in enumerate(batch['bert_evidence_features']):
                for j, bert_d in enumerate(ex_bert_d):  # Chunk level
                    bert_xd[i, j, :len(bert_d.input_ids)].copy_(
                        torch.LongTensor(bert_d.input_ids))
                    bert_xd_mask[i, j, :len(bert_d.input_mask)].copy_(
                        torch.LongTensor(bert_d.input_mask))
            if device:
                bert_xd = bert_xd.to(device)
                bert_xd_mask = bert_xd_mask.to(device)
            output = bert_model(
                bert_xd.view(-1, bert_xd.size(-1)), token_type_ids=None, attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)),output_hidden_states=True)
            all_encoder_layers = output.hidden_states
            torch.cuda.empty_cache()
            all_encoder_layers = torch.stack(
                [x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers], 0).detach()
            all_encoder_layers = all_encoder_layers[layer_indexes]
            bert_xd_f = extract_bert_ctx_hidden_states(
                all_encoder_layers, max_d_len, batch['bert_evidence_features'], weighted_avg=config['use_bert_weight'])
            torch.cuda.empty_cache()

    with torch.set_grad_enabled(False):
        example = {'batch_size': batch_size,
                   'answers': batch['answers'],
                   'doc_token': batch['doc_token'],
                   'xq': xq.to(device) if device else xq,
                   'xq_len': xq_len.to(device) if device else xq_len,
                   'xd': xd.to(device) if device else xd,
                   'xd_len': xd_len.to(device) if device else xd_len,
                   'num_turn': num_turn.to(device) if device else num_turn,
                   'targets': targets.to(device) if device else targets,
                   'span_mask': span_mask.to(device) if device else span_mask}
        # print("doc_token after vectorization: ", batch['doc_token'])
        # print("Answers: ", batch['answers'])

        if config.get('static_graph', None):
            example['xd_graphs'] = batch['evidence_graphs']

        if config['f_tf']:
            example['xd_tf'] = xd_tf.to(device) if device else xd_tf

        
        example['unk_answer_targets'] = unk_answer_targets.to(
            device) if device else unk_answer_targets
        example['yesno_targets'] = yesno_targets.to(
            device) if device else yesno_targets
        example['followup_targets'] = followup_targets.to(
            device) if device else followup_targets

        if config['predict_raw_text']:
            example['raw_evidence_text'] = batch['raw_evidence_text']
            example['offsets'] = batch['offsets']
        else:
            example['evidence_text'] = batch['evidence_text']

        if config['use_bert']:
            example['bert_xq_f'] = bert_xq_f
            example['bert_xd_f'] = bert_xd_f

        if device:
            for feat_key in xd_f:
                xd_f[feat_key] = xd_f[feat_key].to(device)
        example['xd_f'] = xd_f

        if config['n_history'] > 0:
            if config['answer_marker_embed_dim'] != 0:
                example['xd_answer_marker'] = xd_answer_marker.to(
                    device) if device else xd_answer_marker
            if config['use_ques_marker']:
                example['xq_f'] = xq_f.to(device) if device else xq_f
        return example


def featurize(question, document, feature_dict):
    doc_len = len(document['word'])
    features = {}
    if 'f_qem' in feature_dict:
        features['f_qem'] = torch.zeros(doc_len, dtype=torch.long)
    if 'f_pos' in feature_dict:
        features['f_pos'] = torch.zeros(doc_len, dtype=torch.long)
    if 'f_ner' in feature_dict:
        features['f_ner'] = torch.zeros(doc_len, dtype=torch.long)

    q_uncased_words = set([w.lower() for w in question['word']])
    for i in range(doc_len):
        d_word = document['word'][i]
        if 'f_qem' in feature_dict:
            features['f_qem'][i] = feature_dict['f_qem']['yes'] if d_word.lower() in q_uncased_words else feature_dict['f_qem']['no']
        if 'f_pos' in feature_dict:
            assert 'pos' in document
            features['f_pos'][i] = feature_dict['f_pos'][document['pos'][i]] if document['pos'][i] in feature_dict['f_pos'] \
                                    else feature_dict['f_pos'][Constants._UNK_POS]
        if 'f_ner' in feature_dict:
            assert 'ner' in document
            features['f_ner'][i] = feature_dict['f_ner'][document['ner'][i]] if document['ner'][i] in feature_dict['f_ner'] \
                                    else feature_dict['f_ner'][Constants._UNK_NER]
    return features

def mark_prev_answer(span_start, span_end, evidence_answer_marker, prev_answer_distance):
    assert prev_answer_distance >= 0
    try:
        assert span_start >= 0
        assert span_end >= 0
    except:
        raise ValueError("Previous {0:d}th answer span should have been updated!".format(prev_answer_distance))
    # Modify "tags" to mark previous answer span.
    if span_start == span_end:
        evidence_answer_marker[span_start] = 4 * prev_answer_distance + 1
    else:
        evidence_answer_marker[span_start] = 4 * prev_answer_distance + 2
        evidence_answer_marker[span_end] = 4 * prev_answer_distance + 3
        for passage_index in range(span_start + 1, span_end):
            evidence_answer_marker[passage_index] = 4 * prev_answer_distance + 4

def compute_tf(doc):
    doc_len = float(len(doc))
    word_count = Counter(doc)
    tf = []
    for word in doc:
        tf.append(word_count[word] / doc_len)
    return tf

def cons_batch_graph(graphs):
    num_nodes = max([len(g['g_features']) for g in graphs])
    num_edges = max([g['num_edges'] for g in graphs])

    batch_edges = []
    batch_node2edge = []
    batch_edge2node = []
    for g in graphs:
        edges = {}
        node2edge = lil_matrix(np.zeros((num_edges, num_nodes)), dtype=np.float32)
        edge2node = lil_matrix(np.zeros((num_nodes, num_edges)), dtype=np.float32)
        edge_index = 0
        for node1, value in g['g_adj'].items():
            node1 = int(node1)
            for each in value:
                node2 = int(each['node'])
                if node1 == node2: # Ignore self-loops for now
                    continue
                edges[edge_index] = each['edge']
                node2edge[edge_index, node2] = 1
                edge2node[node1, edge_index] = 1
                edge_index += 1
        batch_edges.append(edges)
        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)
    batch_graphs = {'max_num_edges': num_edges,
                    'edge_features': batch_edges,
                    'node2edge': batch_node2edge,
                    'edge2node': batch_edge2node
                    }
    return batch_graphs

def vectorize_batch_graph(graph, edge_vocab=None, config=None):
    # # vectorize the graph
    # edge_features = []
    # for edges in graph['edge_features']:
    #     edges_v = []
    #     for idx in range(len(edges)):
    #         edges_v.append(edge_vocab.getIndex(edges[idx]))
    #     for _ in range(graph['max_num_edges'] - len(edges_v)):
    #         edges_v.append(edge_vocab.PAD)
    #     edge_features.append(edges_v)

    # edge_features = torch.LongTensor(np.array(edge_features))

    gv = {
          # 'edge_features': edge_features.to(config['device']) if config['device'] else edge_features,
          'node2edge': graph['node2edge'],
          'edge2node': graph['edge2node']
          }
    return gv
