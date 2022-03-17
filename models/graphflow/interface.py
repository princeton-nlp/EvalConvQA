from numpy.lib.function_base import _quantile_is_valid
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel
from models.graphflow.model import QuACModel
from models.graphflow.utils.data_utils import QADataset, sanitize_input, vectorize_input_turn

import spacy
from spacy.tokens import Doc

from pycorenlp import StanfordCoreNLP

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class QuacExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += "\nquestion_text: %s" % (self.question_text)

        return s

def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s

class WhitespaceTokenizer(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class ExampleProcessor():
    def __init__(self) -> None:
        self.corenlp = StanfordCoreNLP('http://localhost:9000')
        self.parser = spacy.load("en_core_web_sm")
        self.parser.tokenizer = WhitespaceTokenizer(self.parser.vocab)

    def process(self, text):
        paragraph = self.corenlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, ner',
            'outputFormat': 'json',
            'ssplit.newlineIsSentenceBreak': 'two'})
        print(paragraph)
        output = {'word': [],
                # 'lemma': [],
                'pos': [],
                'ner': [],
                'offsets': []}
        # print(paragraph["sentences"])
        for sent in paragraph['sentences']:
            for token in sent['tokens']:
                output['word'].append(_str(token['word']))
                output['pos'].append(token['pos'])
                output['ner'].append(token['ner'])
                output['offsets'].append(
                    (token['characterOffsetBegin'], token['characterOffsetEnd']))
        return output

class GraphFlow():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              and not self.args['no_cuda'] else "cpu")
        self.args['device'] = self.device
        self.model = QuACModel(self.args)
        self.QA_history = []
        self.history = []
        torch.manual_seed(self.args['seed'])
        
        self.bert_model = BertModel.from_pretrained(
            self.args['bert_model']).to(self.device)
        self.bert_model.eval()
        self.model.init_saved_network(self.args['model_name_or_path'])
        self.model.network = self.model.network.to(self.device)
        self.question_processor = ExampleProcessor()

    
    def tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args['bert_model'], do_lower_case=self.args['do_lower_case'])
        return self.tokenizer
    
    def load_partial_examples(self, predict_file):
        partial_test_set = QADataset(predict_file, self.args)
        return partial_test_set

    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer):
        example = {'id': partial_example['cid'],
                   'evidence': partial_example['evidence'],
                   'raw_evidence': partial_example['raw_evidence']}
        
        qa = {
            'turn_id': partial_example['turn_id'],
            'question': partial_example['question'],
            'answers': partial_example['answers'],
            'targets': partial_example['targets'],
            'span_mask': partial_example['span_mask'],
            'unk_answer_targets': partial_example['unk_answer_targets'],
            'yesno_targets': partial_example['yesno_targets'],
            'followup_targets': partial_example['followup_targets']
        }
        temp = []
        marker = []
        n_history = len(self.history) if self.args['n_history'] < 0 else min(
            self.args['n_history'], len(self.history))
        if n_history > 0:
            count = sum([not self.args['no_pre_question'],
                            not self.args['no_pre_answer']]) * len(self.history[-n_history:])
            for q, a, a_s, a_e in self.history[-n_history:]:
                if not self.args['no_pre_question']:
                    temp.extend(q)
                    marker.extend([count] * len(q))
                    count -= 1
                if not self.args['no_pre_answer']:
                    temp.extend(a)
                    marker.extend([count] * len(a))
                    count -= 1
        original_question = qa['question']['word']
        temp.extend(original_question)
        marker.extend([0] * len(original_question))
        qa['question']['word'] = temp
        qa['question']['marker'] = marker

        example['turns'] = [qa]
        test_loader = DataLoader([example], batch_size=1, shuffle=False,
                                    collate_fn=lambda x: x, pin_memory=True)
        for step, input_batch in enumerate(test_loader):
            input_batch = sanitize_input(input_batch, self.args, self.model.word_dict,
                                            self.model.feature_dict, self.tokenizer, training=False)
            x_batch = vectorize_input_turn(
                input_batch, self.args, self.bert_model, self.history, len(self.history), device=self.device)
            if not x_batch:
                continue  # When there are no target spans present in the batch

            res = self.model.predict(
                x_batch, step, update=False, out_predictions=True)
            prediction = res['predictions'][0][0]
            span = res['token_spans'][0][0]
            tokens = res['token_lists'][0][0]
            self.history.append((original_question, tokens, span[0], span[1]))
            self.QA_history.append((len(self.QA_history), " ".join(original_question), (" ".join(tokens), span[0], span[1]))) # turn, q_text, a_text
        
        return prediction, unique_id+1

    def convert_example(self, partial_example):
        example = QuacExample(
                qas_id=partial_example['turn_id'],
                question_text=" ".join(partial_example['question']['word']),
        )
        return example
    
