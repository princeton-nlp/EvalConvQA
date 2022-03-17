import os
import numpy as np
from collections import Counter
import abc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


from .utils.coqa import compute_eval_metric
from .utils.quac import eval_fn as quac_eval_fn
from .utils.doqa import eval_fn as doqa_eval_fn
from .utils import constants as Constants
from .word_model import WordModel
from .models.graphflow import GraphFlow
from .utils.radam import RAdam


class Model(metaclass=abc.ABCMeta):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        # Book-keeping.
        self.config = config
        if self.config['pretrained']:
            state_dict_opt = self.init_saved_network(self.config['pretrained'])
        else:
            assert train_set is not None
            print('Train vocab: {}'.format(len(train_set.vocab)))
            vocab = Counter()
            for w in train_set.vocab:
                if train_set.vocab[w] >= config['min_freq']:
                    vocab[w] = train_set.vocab[w]
            print('Pruned train vocab: {}'.format(len(vocab)))
            # Building network.
            word_model = WordModel(saved_vocab_file=self.config['saved_vocab_file'],
                                   embed_size=self.config['vocab_embed_size'],
                                   filename=self.config['embed_file'],
                                   embed_type=self.config['embed_type'],
                                   top_n=self.config['top_vocab'],
                                   additional_vocab=vocab)
            self._init_new_network(train_set, word_model)

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        if self.config['use_bert'] and self.config.get('finetune_bert', None):
            for name, p in self.config['bert_model'].named_parameters():
                print('{}: {}'.format(name, str(p.size())))
                num_params += p.numel()
        print('#Parameters = {}\n'.format(num_params))

        self._init_optimizer()
        if self.config['pretrained'] and state_dict_opt:
            self.optimizer.load_state_dict(state_dict_opt)


    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['vocab_embed_size', 'hidden_size', 'f_qem', 'f_pos', 'f_ner',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold', 'max_answer_len']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.word_dict = saved_params['word_dict']
        self.feature_dict = saved_params['feature_dict']
        self.saved_epoch = saved_params['epoch']
        for k, v in self.feature_dict.items():
            self.config['num_features_{}'.format(k)] = len(v)
        state_dict = saved_params['state_dict']
        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]

        w_embedding = self._init_embedding(len(self.word_dict) + 1, self.config['vocab_embed_size'])
        self.network = GraphFlow(self.config, w_embedding)

        # Merge the arguments
        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

        if self.config['use_bert'] and self.config.get('finetune_bert', None):
            
            self.config['bert_model'].load_state_dict(state_dict['bert'])

        return state_dict['optimizer'] if state_dict else None

    def _init_new_network(self, train_set, word_model):
        self.feature_dict = self._build_feature_dict(train_set)
        for k, v in self.feature_dict.items():
            self.config['num_features_{}'.format(k)] = len(v)
        self.word_dict = word_model.get_vocab()
        w_embedding = self._init_embedding(word_model.vocab_size, self.config['vocab_embed_size'],
                                           pretrained_vecs=word_model.get_word_vecs())
        self.network = GraphFlow(self.config, w_embedding)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['use_bert'] and self.config.get('finetune_bert', None):
            parameters += [p for p in self.config['bert_model'].parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(parameters, lr=self.config['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, \
                    patience=1, verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def _build_feature_dict(self, train_set):
        feature_dict = {}
        if self.config['f_qem']:
            feature_dict['f_qem'] = {'yes': 0, 'no': 1}

        if self.config['f_pos'] or self.config['f_ner']:
            pos_tags = set([Constants._UNK_POS])
            ner_tags = set([Constants._UNK_NER])
            for ex in train_set:
                if self.config['f_pos']:
                    assert 'pos' in ex['evidence']
                    pos_tags |= set(ex['evidence']['pos'])
                if self.config['f_ner']:
                    assert 'ner' in ex['evidence']
                    ner_tags |= set(ex['evidence']['ner'])
            if self.config['f_pos']:
                print('{} pos tags: {}'.format(len(pos_tags), str(pos_tags)))
                feature_dict['f_pos'] = dict(zip(pos_tags, range(len(pos_tags))))
            if self.config['f_ner']:
                print('{} ner tags: {}'.format(len(ner_tags), str(ner_tags)))
                feature_dict['f_ner'] = dict(zip(ner_tags, range(len(ner_tags))))
        return feature_dict

    def compute_span_loss(self, score_s, score_e, targets, target_mask):
        assert targets.size(0) == score_s.size(0) == score_e.size(0)
        loss = F.nll_loss(score_s.view(-1, score_s.size(-1)), targets[:, :, 0].view(-1), reduction='none') + F.nll_loss(score_e.view(-1, score_e.size(-1)), targets[:, :, 1].view(-1), reduction='none')
        loss = torch.sum(loss.view_as(target_mask) * target_mask) / torch.clamp(torch.sum(target_mask), min=Constants.VERY_SMALL_NUMBER)
        return loss

    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'bert': self.config['bert_model'].state_dict() if self.config['use_bert'] and self.config.get('finetune_bert', None) else None,
                'optimizer': self.optimizer.state_dict()
            },
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'config': self.config,
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    @abc.abstractmethod
    def predict(self, ex, update=True, out_predictions=False):
        return

    @abc.abstractmethod
    def compute_answer_type_loss(self, score_c, answer_type_targets, turn_mask):
        return

    @abc.abstractmethod
    def extract_predictions(self, ex, score_s, score_e, score_c, turn_mask):
        return

    def _scores_to_text(self, text, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(0)
        scores = torch.ger(score_s, score_e)
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))

    def _scores_to_raw_text(self, raw_text, offsets, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(0)
        scores = torch.ger(score_s, score_e)
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        prediction = raw_text[offsets[s_idx][0]: offsets[e_idx][1]]
        span = (offsets[s_idx][0], offsets[e_idx][1])
        token_span = (s_idx, e_idx)
        return prediction, span, token_span

class CoQAModel(Model):
    """High level CoQA model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        super(CoQAModel, self).__init__(config, train_set)


    def predict(self, ex, step, update=True, out_predictions=False):
        # Train/Eval mode
        self.network.train(update)
        # Run forward
        with torch.set_grad_enabled(update):
            res = self.network(ex)
        score_s, score_e, score_c = res['start_logits'], res['end_logits'], res['score_c']

        output = {
            'metrics': {'f1': 0.0, 'em': 0.0},
            'loss': 0.0,
            'total_qs': 0,
            'total_dials': 0
        }


        # Compute loss
        loss = self.compute_span_loss(score_s, score_e, ex['targets'], ex['span_mask'])
        loss = loss + self.compute_answer_type_loss(score_c, ex['answer_type_targets'], res['turn_mask'])
        output['loss'] = loss.item()

        if update:
            # Accumulate gradients
            loss = loss / self.config['grad_accumulated_steps'] # Normalize our loss (if averaged)
            # Run backward
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0: # Wait for several backward steps
                if self.config['optimizer'] != 'bert_adam' and self.config['grad_clipping']:
                    # Clip gradients
                    parameters = [p for p in self.network.parameters() if p.requires_grad]
                    if self.config['use_bert'] and self.config.get('finetune_bert', None):
                        parameters += [p for p in self.config['bert_model'].parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

        if (not update) or self.config['predict_train']:
            predictions, spans = self.extract_predictions(ex, score_s, score_e, score_c, res['turn_mask'])
            expand_predictions = [y for x in predictions for y in x]
            expand_answers = [y for x in ex['answers'] for y in x]
            output['total_qs'] = len(expand_predictions)
            output['total_dials'] = len(predictions)
            f1, em = self.evaluate_predictions(expand_predictions, expand_answers)
            output['metrics']['f1'] = f1
            output['metrics']['em'] = em
            if out_predictions:
                output['predictions'] = predictions
                output['spans'] = spans
        torch.cuda.empty_cache()
        return output

    def compute_answer_type_loss(self, score_c, answer_type_targets, turn_mask):
        loss = F.cross_entropy(score_c.view(-1, score_c.size(-1)), answer_type_targets.view(-1), reduction='none')
        loss = torch.sum(loss.view_as(turn_mask) * turn_mask) / torch.clamp(torch.sum(turn_mask), min=Constants.VERY_SMALL_NUMBER)
        return loss

    def extract_predictions(self, ex, score_s, score_e, score_c, turn_mask):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        score_s = score_s.exp()
        score_e = score_e.exp()
        score_c = score_c.data.cpu()

        predictions = []
        spans = []
        for i, (_s, _e, _c) in enumerate(zip(score_s, score_e, score_c)): # Example-level
            para_pred = []
            para_span = []
            for j in range(_s.size(0)): # Turn-level
                if turn_mask[i, j] == 0: # This dialog has ended
                    break

                ans_type = np.argmax(_c[j]).item()
                if ans_type == Constants.CoQA_UNK_ANSWER_LABEL:
                    pred = Constants.CoQA_UNK_ANSWER
                    span = (-1, -1)
                elif ans_type == Constants.CoQA_ANSWER_YES_LABEL:
                    pred = Constants.CoQA_YES_ANSWER
                    span = (-1, -1)
                elif ans_type == Constants.CoQA_ANSWER_NO_LABEL:
                    pred = Constants.CoQA_NO_ANSWER
                    span = (-1, -1)
                else:
                    if self.config['predict_raw_text']:
                        pred, span = self._scores_to_raw_text(ex['raw_evidence_text'][i],
                                                                    ex['offsets'][i], _s[j], _e[j])
                    else:
                        pred, span = self._scores_to_text(ex['evidence_text'][i], _s[j], _e[j])

                para_pred.append(pred)
                para_span.append(span)
            predictions.append(para_pred)
            spans.append(para_span)
        return predictions, spans

    def evaluate_predictions(self, predictions, answers):
        assert len(predictions) == len(answers)
        f1_score = compute_eval_metric('f1', predictions, answers)
        em_score = compute_eval_metric('em', predictions, answers)
        return f1_score, em_score

class QuACModel(Model):
    """High level QuAC model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        super(QuACModel, self).__init__(config, train_set)

    def predict(self, ex, step, update=False, out_predictions=True):
        # Train/Eval mode
        self.network.train(update)
        # Run forward
        with torch.set_grad_enabled(update):
            res = self.network(ex)
        score_s, score_e, unk_probs, score_yesno, score_followup = res['start_logits'], res['end_logits'], res['unk_probs'], res['score_yesno'], res['score_followup']

        output = {}


        if (not update) or self.config['predict_train']:
            predictions, spans, yesnos, followups, token_spans, token_lists = self.extract_predictions(ex, score_s, score_e, unk_probs, score_yesno, score_followup, self.config['unk_answer_threshold'], res['turn_mask'])

            if out_predictions:
                output['predictions'] = predictions
                output['spans'] = spans
                output['yesnos'] = yesnos
                output['followups'] = followups
                output['token_spans'] = token_spans
                output['token_lists'] = token_lists
        torch.cuda.empty_cache()
        return output

    def compute_answer_type_loss(self, unk_probs, score_yesno, score_followup, unk_answer_targets, yesno_targets, followup_targets, turn_mask):
        loss = F.binary_cross_entropy(unk_probs.view(-1), unk_answer_targets.view(-1), reduction='none') \
                + F.cross_entropy(score_yesno.view(-1, score_yesno.size(-1)), yesno_targets.view(-1), reduction='none') \
                + F.cross_entropy(score_followup.view(-1, score_followup.size(-1)), followup_targets.view(-1), reduction='none')
        loss = torch.sum(loss.view_as(turn_mask) * turn_mask) / torch.clamp(torch.sum(turn_mask), min=Constants.VERY_SMALL_NUMBER)
        return loss

    def extract_predictions(self, ex, score_s, score_e, unk_probs, score_yesno, score_followup, unk_answer_threshold, turn_mask):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        score_s = score_s.exp()
        score_e = score_e.exp()
        score_yesno = score_yesno.data.cpu()
        score_followup = score_followup.data.cpu()


        predictions = []
        spans = []
        yesnos = []
        followups = []
        token_spans = []
        token_lists = []
        for i, (_s, _e, _unk, _yesno, _followup) in enumerate(zip(score_s, score_e, unk_probs, score_yesno, score_followup)): # Example-level
            para_pred = []
            para_span = []
            para_yesno = []
            para_followup = []
            para_token_span = []
            para_tokens = []
            for j in range(_s.size(0)): # Turn-level
                if turn_mask[i, j] == 0: # This dialog has ended
                    break

                if _unk[j].item() >= unk_answer_threshold:
                    pred = Constants.QuAC_UNK_ANSWER.upper()
                    tokens = [Constants.QuAC_UNK_ANSWER.upper()]
                    span = (-1, -1)
                    token_span = (-1, -1)
                else:
                    if self.config['predict_raw_text']:
                        pred, span, token_span = self._scores_to_raw_text(ex['raw_evidence_text'][i],
                                                                    ex['offsets'][i], _s[j], _e[j])
                        tokens = ex['doc_token'][i][token_span[0]:token_span[1]+1]
                    else:
                        pred, span = self._scores_to_text(ex['evidence_text'][i], _s[j], _e[j])
                        token_span = (0, 0)
                        tokens = []

                yesno_type = np.argmax(_yesno[j]).item()
                if yesno_type == Constants.QuAC_YESNO_YES_LABEL:
                    yesno = Constants.QuAC_YESNO_YES
                elif yesno_type == Constants.QuAC_YESNO_NO_LABEL:
                    yesno = Constants.QuAC_YESNO_NO
                else:
                    yesno = Constants.QuAC_YESNO_OTHER

                followup_type = np.argmax(_followup[j]).item()


                if followup_type == Constants.QuAC_FOLLOWUP_YES_LABEL:
                    followup = Constants.QuAC_FOLLOWUP_YES
                elif followup_type == Constants.QuAC_FOLLOWUP_NO_LABEL:
                    followup = Constants.QuAC_FOLLOWUP_NO
                else:
                    followup = Constants.QuAC_FOLLOWUP_OTHER

                
                para_pred.append(pred)
                para_span.append(span)
                para_yesno.append(yesno)
                para_followup.append(followup)
                para_token_span.append(token_span)
                para_tokens.append(tokens)
                # print("Span:", span)
                # print("Prediction:", pred)
                # print("Tokens:", tokens)
            predictions.append(para_pred)
            spans.append(para_span)
            yesnos.append(para_yesno)
            followups.append(para_followup)
            token_spans.append(para_token_span)
            token_lists.append(para_tokens)
        return predictions, spans, yesnos, followups, token_spans, token_lists
