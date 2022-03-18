"""
    This file takes a QuAC data file as input and generates the input files for training a conversational reading comprehension model.
"""


import argparse
import json
import re
import time
import string
from collections import Counter, defaultdict
import spacy
from spacy.tokens import Doc

from pycorenlp import StanfordCoreNLP


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
        self.parser = spacy.load('en')
        self.parser.tokenizer = WhitespaceTokenizer(self.parser.vocab)

    def process(self, text):
        paragraph = self.corenlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, ner',
            'outputFormat': 'json',
            'ssplit.newlineIsSentenceBreak': 'two'})

        output = {'word': [],
                # 'lemma': [],
                'pos': [],
                'ner': [],
                'offsets': []}

        for sent in paragraph['sentences']:
            for token in sent['tokens']:
                output['word'].append(_str(token['word']))
                output['pos'].append(token['pos'])
                output['ner'].append(token['ner'])
                output['offsets'].append(
                    (token['characterOffsetBegin'], token['characterOffsetEnd']))
        return output
    
    def extract_sent_dep_tree(self, text):
        if len(text) == 0:
            return {'g_features': [], 'g_adj': {}, 'num_edges': 0}

        doc = self.parser(text)
        g_features = []
        dep_tree = defaultdict(list)
        boundary_nodes = []
        num_edges = 0
        for sent in doc.sents:
            boundary_nodes.append(sent[-1].i)
            for each in sent:
                g_features.append(each.text)
                if each.i != each.head.i:  # Not a root
                    dep_tree[each.head.i].append(
                        {'node': each.i, 'edge': each.dep_})
                    num_edges += 1

        for i in range(len(boundary_nodes) - 1):
            # Add connection between neighboring dependency trees
            dep_tree[boundary_nodes[i]].append(
                {'node': boundary_nodes[i] + 1, 'edge': 'neigh'})
            dep_tree[boundary_nodes[i] +
                    1].append({'node': boundary_nodes[i], 'edge': 'neigh'})
            num_edges += 2

        info = {'g_features': g_features,
                'g_adj': dep_tree,
                'num_edges': num_edges
                }
        return info


    def extract_sent_dep_order_tree(self, text):
        '''Keep both dependency and ordering info'''
        if len(text) == 0:
            return {'g_features': [], 'g_adj': {}, 'num_edges': 0}

        doc = self.parser(text)
        g_features = []
        dep_tree = defaultdict(list)

        num_edges = 0
        # Add word ordering info
        for i in range(len(doc) - 1):
            dep_tree[i].append({'node': i + 1, 'edge': 'neigh'})
            dep_tree[i + 1].append({'node': i, 'edge': 'neigh'})
            num_edges += 2

        # Add dependency info
        for sent in doc.sents:
            for each in sent:
                g_features.append(each.text)
                # Not a root
                if each.i != each.head.i and abs(each.head.i - each.i) != 1:
                    dep_tree[each.head.i].append(
                        {'node': each.i, 'edge': each.dep_})
                    num_edges += 1

        info = {'g_features': g_features,
                'g_adj': dep_tree,
                'num_edges': num_edges
                }
        return info
