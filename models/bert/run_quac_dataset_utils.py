# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load QuAC dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
from tqdm import tqdm
import spacy
import re
from collections import Counter
import string
# import msgpack
import numpy as np
import unicodedata

from transformers import BertTokenizer
# from coref_resolver import find_coreference_f1s, resolve_coreference

logger = logging.getLogger(__name__)


class QuacExample(object):
    """
    A single training/test example for the QuAC dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            doc_tokens,
            background=None,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            rational_start_position=None,
            rational_end_position=None,
            additional_answers=None,
    ):
        self.qas_id = qas_id
        self.background = background
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.additional_answers = additional_answers
        self.rational_start_position = rational_start_position
        self.rational_end_position = rational_end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += "\nquestion_text: %s" % (self.question_text)
        s += "\ndoc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += "background: %s" % (self.background)
        s += "\norig_answer_text: [%s]" % (self.orig_answer_text)
        s += "\nstart_position: %d" % (self.start_position)
        s += "\nend_position: %d" % (self.end_position)
        s += "\nrational_start_position: %d" % (self.rational_start_position)
        s += "\nrational_end_position: %d" % (self.rational_end_position)
        if self.additional_answers is not None:
            s += "\nadditional_answers: [%s]" % ("\n".join(self.additional_answers))

        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 rational_mask=None,
                 cls_idx=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.cls_idx = cls_idx
        self.rational_mask = rational_mask
    
    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        s = ""
        s += "unique_id: %s" % (self.unique_id)
        s += "\nexample_index: %s" % (self.example_index)
        s += "\ndoc_span_index: %s" % (self.doc_span_index)
        s += "\ntokens: [%s]" % (" ".join(self.tokens))
        s += "\ntoken_to_orig_map: [%s]" % (" ".join(["%d:%d" % (x, y)
             for (x, y) in self.token_to_orig_map.items()]))
        s += "\ntoken_is_max_context: [%s]" % (" ".join([
            "%d:%s" % (x, y)
            for (x, y) in self.token_is_max_context.items()]))
        s += "\ninput_ids: [%s]" % (" ".join([str(x) for x in self.input_ids]))
        s += "\ninput_mask: [%s]" % (" ".join([str(x) for x in self.input_mask]))
        s += "\nsegment_ids: [%s]" % (" ".join([str(x) for x in self.segment_ids]))
        s += "\nstart_position: %d" % (self.start_position)
        s += "\nend_position: %d" % (self.end_position)
        s += "\ncls_idx: %d" % (self.cls_idx)
        s += "\nrational_mask: [%s]" % (" ".join([str(x) for x in self.rational_mask]))
        
        return s

def read_one_quac_example_extern(partial_example, QA_history, history_len=2, add_QA_tag=False):
    """Append the previous predicted answers to the context during evaluation"""
    long_questions=[]
    i = len(QA_history)
    i = 0
    total_length = len(QA_history)
    while i < history_len:
        index = total_length-1-i
        if index >= 0:
            long_question = ' '
            long_question = (' <A> ' if add_QA_tag else
                              ' ') + QA_history[index][2][0] + ' ' + long_question # answer
            long_question = (' <Q> ' if add_QA_tag else
                          ' ') + QA_history[index][1] + ' ' + long_question # question        
            long_question = long_question.strip()
            long_questions.append(long_question)    
        i+=1
    long_questions.append(((' <Q> ' if add_QA_tag else
                          ' ') + partial_example.question_text).strip())

    partial_example.question_text=long_questions
    return partial_example

def read_partial_quac_examples_extern(input_file):
    """Read the QuAC dataset into a list of paragraphs. Neither context nor QAs are tokenized yet.
    Format of each paragraph:
    {"context_id": str,
     "section_title": str,
     "background": str,
     "gold_answers": [{"text": str,
                       "answer_start": int},
                       ...]
     "examples": [
         {
            "qas_id": str,
            "original_question": str,
         }
     ]}
    
    """
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def find_span(offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)
    def normalize_text(text):
        return unicodedata.normalize('NFD', text)

    def space_extend(matchobj):
        return ' ' + matchobj.group(0) + ' '

    def pre_proc(text):
        text = re.sub(
            u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t',
            space_extend, text)
        text = text.strip(' \n')
        text = re.sub('\s+', ' ', text)
        return text
    
    def get_context_span(context, context_token):
        p_str = 0
        p_token = 0
        t_span = []
        while p_str < len(context):
            if re.match('\s', context[p_str]):
                p_str += 1
                continue

            token = context_token[p_token]
            token_len = len(token)
            if context[p_str:p_str + token_len] != token:
                log.info("Something wrong with get_context_span()")
                return []
            t_span.append((p_str, p_str + token_len))

            p_str += token_len
            p_token += 1
        return t_span

    """Main Stream"""
    nlp = spacy.load('en_core_web_sm')

    articles = []

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)["data"]
    
    for article in data:
        section_title = article['section_title']
        background = article['background']

        paragraph = article['paragraphs'][0]
        context_str = paragraph['context']
        context_nlp = nlp(pre_proc(context_str))
        context_tok = [normalize_text(w.text) for w in context_nlp]
        context_span = get_context_span(context_str, context_tok)
        cid = paragraph['id']
        qas = []
        gold_answers = []
        for qa in paragraph['qas']:
            qas_id = qa['id']
            question = qa['question']
            gold_answer = qa['orig_answer']['text']
            gold_answers.append(gold_answer)

            # findstart and end character positions
            start = int(qa['orig_answer']['answer_start'])
            # this matches the space after the token
            end = start+len(gold_answer)
            # generate rational
            chosen_text = context_str[start:end].lower()
            while len(chosen_text) > 0 and is_whitespace(chosen_text[0]):
                chosen_text = chosen_text[1:]
                start += 1
            while len(chosen_text) > 0 and is_whitespace(chosen_text[-1]):
                chosen_text = chosen_text[:-1]
                end -= 1
            r_start, r_end = find_span(context_span, start, end)
            # answer span is the same as rational span
            start_position = r_start
            end_position = r_end

            example = QuacExample(
            qas_id=qas_id,
            background=background,
            question_text=question,
            doc_tokens=context_tok,
            orig_answer_text="",
            start_position=start_position,
            end_position=end_position,
            rational_start_position=r_start,
            rational_end_position=r_end,
            additional_answers=None,
            )
            qas.append(example)

        a = dict(
            context_id=cid,
            section_title=section_title,
            background=background,
            orig_context=context_str,
            context=context_tok,
            examples=qas,
            gold_answers=gold_answers
        )
        articles.append(a)
    return articles

def convert_one_example_to_features(example, unique_id, example_index, tokenizer, max_seq_length, doc_stride, max_query_length):
    # unique id for all features
    logger.info("*** Generating feature for example index %d ***" % example_index)
    unique_id = unique_id
    query_tokens = []
    for qa in example.question_text:
        query_tokens.extend(tokenizer.tokenize(qa))
    
    cls_idx = 3  # not one of below
    if example.orig_answer_text == 'yes':
        cls_idx = 0  # yes
    elif example.orig_answer_text == 'no':
        cls_idx = 1  # no
    elif example.orig_answer_text == 'CANNOTANSWER':
        cls_idx = 2  # CANNOTANSWER

    if len(query_tokens) > max_query_length:  # keep tail, not head
        query_tokens.reverse()
        query_tokens = query_tokens[0:max_query_length]
        query_tokens.reverse()

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    tok_r_start_position, tok_r_end_position = None, None

    # rational part
    tok_r_start_position = orig_to_tok_index[
        example.rational_start_position]
    if example.rational_end_position < len(example.doc_tokens) - 1:
        tok_r_end_position = orig_to_tok_index[
            example.rational_end_position + 1] - 1
    else:
        tok_r_end_position = len(all_doc_tokens) - 1
    # rational part end

    # if tok_r_end_position is None:
    #     print('DEBUG')

    if cls_idx < 3:
        tok_start_position, tok_end_position = 0, 0
    else:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position +
                                                    1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position,
            tokenizer, example.orig_answer_text)
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    
    features = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        slice_cls_idx = cls_idx
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        # cur_id = 2 - query_tokens.count('[SEP]')

        # assert cur_id >= 0

        tokens.append(tokenizer.cls_token)
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            # if token == '[SEP]':
            #     cur_id += 1
        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)
        # cur_id += 1

        # assert cur_id <= 3

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(
                tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans,
                                                    doc_span_index,
                                                    split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)

        tokens.append(tokenizer.sep_token)
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        rational_start_position = None
        rational_end_position = None

        # rational_part
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if example.rational_start_position == -1 or not (
                tok_r_start_position >= doc_start
                and tok_r_end_position <= doc_end):
            out_of_span = True
        if out_of_span:
            rational_start_position = 0
            rational_end_position = 0
        else:
            doc_offset = len(query_tokens) + 2
            rational_start_position = tok_r_start_position - doc_start + doc_offset
            rational_end_position = tok_r_end_position - doc_start + doc_offset
        # rational_part_end

        rational_mask = [0] * len(input_ids)
        if not out_of_span:
            rational_mask[rational_start_position:rational_end_position +
                            1] = [1] * (rational_end_position -
                                        rational_start_position + 1)

        if cls_idx >= 3:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start
                    and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                slice_cls_idx = 2
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        else:
            start_position = 0
            end_position = 0

        logger.info("*** Example ***")
        logger.info("unique_id: %s" % (unique_id))
        logger.info("example_index: %s" % (example_index))
        logger.info("doc_span_index: %s" % (doc_span_index))
        logger.info("tokens: %s" % " ".join(tokens))
        logger.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y)
                for (x, y) in token_to_orig_map.items()]))
        logger.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y)
            for (x, y) in token_is_max_context.items()
        ]))
        logger.info("input_ids: %s" %
                    " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" %
                    " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" %
                    " ".join([str(x) for x in segment_ids]))

        if slice_cls_idx >= 3:
            answer_text = " ".join(
                tokens[start_position:(end_position + 1)])
        else:
            tmp = ['yes', 'no', 'CANNOTANSWER']
            answer_text = tmp[slice_cls_idx]

        rational_text = " ".join(
            tokens[rational_start_position:(rational_end_position +
                                            1)])
        # logger.info("start_position: %d" % (start_position))
        # logger.info("end_position: %d" % (end_position))
        # logger.info("rational_start_position: %d" %
        #             (rational_start_position))
        # logger.info("rational_end_position: %d" %
        #             (rational_end_position))
        # logger.info("answer: %s" % (answer_text))
        # logger.info("rational: %s" % (rational_text))

        feature = InputFeatures(unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    rational_mask=rational_mask,
                    cls_idx=slice_cls_idx)
        features.append(feature)             
        unique_id+=1

    return features, unique_id

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult", [
    "unique_id", "start_logits", "end_logits", "yes_logits", "no_logits",
    "unk_logits"
])


def recover_predicted_answer(example, features, results, tokenizer, n_best_size, max_answer_length, do_lower_case, 
                             verbose_logging):
    """Retrieves the predicted answer text from current example, features and prediction results"""

    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result.unique_id] = result
    
    logger.info("Recovering predicted answer from result of unique_id: [%s]" % (",".join([str(x) for x in unique_id_to_result.keys()])))
    
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index",
            "start_index",
            "end_index",
            "score",
            "cls_idx",
        ])
    

    prelim_predictions = []

    score_yes, score_no, score_span, score_unk = -float('INF'), -float(
        'INF'), -float('INF'), float('INF')
    min_unk_feature_index, max_yes_feature_index, max_no_feature_index, max_span_feature_index = - \
        1, -1, -1, -1  # the paragraph slice with min null score
    max_span_start_indexes, max_span_end_indexes = [], []
    max_start_index, max_end_index = -1, -1
    # null_start_logit = 0  # the start logit at the slice with min null score
    # null_end_logit = 0  # the end logit at the slice with min null score

    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        # if we could have irrelevant answers, get the min score of irrelevant
        # feature_null_score = result.start_logits[0] + result.end_logits[0]

        # feature_yes_score, feature_no_score, feature_unk_score, feature_span_score = result.cls_logits

        feature_yes_score, feature_no_score, feature_unk_score = result.yes_logits[
            0] * 2, result.no_logits[0] * 2, result.unk_logits[0] * 2
        start_indexes, end_indexes = _get_best_indexes(
            result.start_logits,
            n_best_size), _get_best_indexes(result.end_logits, n_best_size)

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(
                        start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                feature_span_score = result.start_logits[
                    start_index] + result.end_logits[end_index]
                prelim_predictions.append(
                    _PrelimPrediction(feature_index=feature_index,
                                        start_index=start_index,
                                        end_index=end_index,
                                        score=feature_span_score,
                                        cls_idx=3))

        if feature_unk_score < score_unk:  # find min score_noanswer
            score_unk = feature_unk_score
            min_unk_feature_index = feature_index
        if feature_yes_score > score_yes:  # find max score_yes
            score_yes = feature_yes_score
            max_yes_feature_index = feature_index
        if feature_no_score > score_no:  # find max score_no
            score_no = feature_no_score
            max_no_feature_index = feature_index

    prelim_predictions.append(
        _PrelimPrediction(feature_index=min_unk_feature_index,
                            start_index=0,
                            end_index=0,
                            score=score_unk,
                            cls_idx=2))
    prelim_predictions.append(
        _PrelimPrediction(feature_index=max_yes_feature_index,
                            start_index=0,
                            end_index=0,
                            score=score_yes,
                            cls_idx=0))
    prelim_predictions.append(
        _PrelimPrediction(feature_index=max_no_feature_index,
                            start_index=0,
                            end_index=0,
                            score=score_no,
                            cls_idx=1))

    prelim_predictions = sorted(prelim_predictions,
                                key=lambda p: p.score,
                                reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "score", "cls_idx"])

    seen_predictions = {}
    nbest = []
    cls_rank = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:  # including yes/no/noanswer pred
            break
        feature = features[pred.feature_index]
        if pred.cls_idx == 3:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index +
                                                            1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer, do_lower_case,
                                        verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(text=final_text,
                                    score=pred.score,
                                    cls_idx=pred.cls_idx))
        else:
            text = ['yes', 'no', 'CANNOTANSWER']
            nbest.append(
                _NbestPrediction(text=text[pred.cls_idx],
                                    score=pred.score,
                                    cls_idx=pred.cls_idx))

    if len(nbest) < 1:
        nbest.append(
            _NbestPrediction(text='CANNOTANSWER',
                                score=-float('inf'),
                                cls_idx=2))

    assert len(nbest) >= 1

    probs = _compute_softmax([p.score for p in nbest])

    nbest_json = []

    for i, entry in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        # output["start_logit"] = entry.start_logit
        # output["end_logit"] = entry.end_logit
        output["socre"] = entry.score
        nbest_json.append(output)

    assert len(nbest_json) >= 1

    predicted_answer = confirm_preds(nbest_json)

    return predicted_answer


def write_predictions_turn(predictions,output_prediction_file):
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(predictions, indent=4) + "\n")

def write_skipped_questions(skipped_questions, question_file):
    with open(question_file, "w") as writer:
        writer.write(json.dumps(skipped_questions, indent=4) + "\n")
        
def confirm_preds(nbest_json):
    # Do something for some obvious wrong-predictions
    subs = [
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'true', 'false'
    ]  # very hard-coding, can be extended.
    ori = nbest_json[0]['text']
    if len(ori) < 2:  # mean span like '.', '!'
        for e in nbest_json[1:]:
            if _normalize_answer(e['text']) in subs:
                return e['text']
        return 'CANNOTANSWER'
    return ori

def get_final_text(pred_text, orig_text, tokenizer, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" %
                        (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits),
                             key=lambda x: x[1],
                             reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def score(pred, truth):
    def _f1_score(pred, answers):
        def _score(g_tokens, a_tokens):
            common = Counter(g_tokens) & Counter(a_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1. * num_same / len(g_tokens)
            recall = 1. * num_same / len(a_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        if pred is None or answers is None:
            return 0

        if len(answers) == 0:
            return 1. if len(pred) == 0 else 0.

        g_tokens = _normalize_answer(pred).split()
        ans_tokens = [_normalize_answer(answer).split() for answer in answers]
        scores = [_score(g_tokens, a) for a in ans_tokens]
        if len(ans_tokens) == 1:
            score = scores[0]
        else:
            score = 0
            for i in range(len(ans_tokens)):
                scores_one_out = scores[:i] + scores[(i + 1):]
                score += max(scores_one_out)
            score /= len(ans_tokens)
        return score

    # Main Stream
    assert len(pred) == len(truth)
    pred, truth = pred.items(), truth.items()
    no_ans_total = no_total = yes_total = normal_total = total = 0
    no_ans_f1 = no_f1 = yes_f1 = normal_f1 = f1 = 0
    all_f1s = []
    for (p_id, p), (t_id, t), in zip(pred, truth):
        assert p_id == t_id
        total += 1
        this_f1 = _f1_score(p, t)
        f1 += this_f1
        all_f1s.append(this_f1)
        if t[0].lower() == 'no':
            no_total += 1
            no_f1 += this_f1
        elif t[0].lower() == 'yes':
            yes_total += 1
            yes_f1 += this_f1
        elif t[0].lower() == 'cannotanswer':
            no_ans_total += 1
            no_ans_f1 += this_f1
        else:
            normal_total += 1
            normal_f1 += this_f1

    f1 = 100. * f1 / total
    if no_total == 0:
        no_f1 = 0.
    else:
        no_f1 = 100. * no_f1 / no_total
    if yes_total == 0:
        yes_f1 = 0
    else:
        yes_f1 = 100. * yes_f1 / yes_total
    if no_ans_total == 0:
        no_ans_f1 = 0.
    else:
        no_ans_f1 = 100. * no_ans_f1 / no_ans_total
    normal_f1 = 100. * normal_f1 / normal_total
    result = {
        'total': total,
        'f1': f1,
        'no_total': no_total,
        'no_f1': no_f1,
        'yes_total': yes_total,
        'yes_f1': yes_f1,
        'no_ans_total': no_ans_total,
        'no_ans_f1': no_ans_f1,
        'normal_total': normal_total,
        'normal_f1': normal_f1,
    }
    return result, all_f1s
