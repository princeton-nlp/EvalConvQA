# -*- coding: utf-8 -*-
"""
Module to handle word vectors and initializing embeddings.
"""
import os
import string
from collections import Counter
import numpy as np

from .utils import constants as Constants
from .utils import dump_ndarray, load_ndarray, dump_json, load_json, Timer


################################################################################
# WordModel Class #
################################################################################

class GloveModel(object):

    def __init__(self, filename):
        self.word_vecs = {}
        self.vocab = []
        with open(filename, 'r') as input_file:
            for line in input_file.readlines():
                splitLine = line.split(' ')
                w = splitLine[0]
                self.word_vecs[w] = np.array([float(val) for val in splitLine[1:]])
                self.vocab.append(w)
        self.vector_size = len(self.word_vecs[w])

    def word_vec(self, word):
        word_list = [word, word.lower(), word.upper(), word.title(), string.capwords(word, '_')]

        for w in word_list:
            if w in self.word_vecs:
                return self.word_vecs[w]
        return None


class WordModel(object):
    """Class to get pretrained word vectors for a list of sentences. Can be used
    for any pretrained word vectors.
    """

    def __init__(self, saved_vocab_file=None, embed_size=None, filename=None, embed_type='glove', top_n=None, additional_vocab=Counter()):
        vocab_path = saved_vocab_file + '.vocab'
        word_vec_path = saved_vocab_file + '.npy'
        if os.path.exists(vocab_path) and \
                    os.path.exists(word_vec_path):
            print('Loading pre-built vocabs stored in {}'.format(saved_vocab_file))
            self.vocab = load_json(vocab_path)
            self.word_vecs = load_ndarray(word_vec_path)
            self.vocab_size = len(self.vocab) + 1
            self.embed_size = self.word_vecs.shape[1]
            assert self.embed_size == embed_size
        else:
            print('Building vocabs...')
            if filename is None:
                if embed_size is None:
                    raise Exception('Either embed_file or embed_size needs to be specified.')
                self.embed_size = embed_size
                self._model = None
            else:
                self.set_model(filename, embed_type)
                self.embed_size = self._model.vector_size

            # padding: 0
            self.vocab = {Constants._UNK_TOKEN: 1, Constants._QUESTION_SYMBOL: 2, Constants._ANSWER_SYMBOL: 3}
            n_added = 0
            for w, count in additional_vocab.most_common():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab) + 1
                    n_added += 1
            # print('Added {} words to the vocab in total.'.format(n_added))

            self.vocab_size = len(self.vocab) + 1
            print('Vocab size: {}'.format(self.vocab_size))
            # self.word_vecs = np.random.rand(self.vocab_size, self.embed_size) * 0.2 - 0.1
            self.word_vecs = np.random.uniform(-0.08, 0.08, (self.vocab_size, self.embed_size))
            i = 0.
            if self._model is not None:
                for word in self.vocab:
                    emb = self._model.word_vec(word)
                    if emb is not None:
                        i += 1
                        self.word_vecs[self.vocab[word]] = emb
            self.word_vecs[0] = 0
            print('Get_wordemb hit ratio: {}'.format(i / len(self.vocab)))
            dump_json(self.vocab, vocab_path)
            print('Saved vocab to {}'.format(vocab_path))
            dump_ndarray(self.word_vecs, word_vec_path)
            print('Saved word_vecs to {}'.format(word_vec_path))

    def set_model(self, filename, embed_type='glove'):
        timer = Timer('Load {}'.format(filename))
        if embed_type == 'glove':
            self._model = GloveModel(filename)
        else:
            from gensim.models.keyedvectors import KeyedVectors
            self._model = KeyedVectors.load_word2vec_format(filename, binary=True
                                                            if embed_type == 'word2vec' else False)
        print('Embeddings: vocab = {}, embed_size = {}'.format(len(self._model.vocab), self._model.vector_size))
        timer.finish()

    def get_vocab(self):
        return self.vocab

    def get_word_vecs(self):
        return self.word_vecs
