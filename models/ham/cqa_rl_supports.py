from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from enum import unique
import tensorflow as tf
import numpy as np
from copy import deepcopy
from models.ham.cqa_supports import *
from models.ham.cqa_model import *
from models.ham.cqa_gen_batches import *
# from cqa_selection_supports import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
def convert_features_to_feed_dict(features):
    batch_unique_ids, batch_input_ids, batch_input_mask = [], [], []
    batch_segment_ids, batch_start_positions, batch_end_positions, batch_history_answer_marker = [], [], [], []
    batch_yesno, batch_followup = [], []
    batch_metadata = []
    
    yesno_dict = {'y': 0, 'n': 1, 'x': 2}
    followup_dict = {'y': 0, 'n': 1, 'm': 2}
    
    for feature in features:
        batch_unique_ids.append(feature.unique_id)
        batch_input_ids.append(feature.input_ids)
        batch_input_mask.append(feature.input_mask)
        batch_segment_ids.append(feature.segment_ids)
        batch_start_positions.append(feature.start_position)
        batch_end_positions.append(feature.end_position)
        batch_history_answer_marker.append(feature.history_answer_marker)
        batch_yesno.append(yesno_dict[feature.metadata['yesno']])
        batch_followup.append(followup_dict[feature.metadata['followup']])
        batch_metadata.append(feature.metadata)
    
    feed_dict = {'unique_ids': batch_unique_ids, 'input_ids': batch_input_ids, 
              'input_mask': batch_input_mask, 'segment_ids': batch_segment_ids, 
              'start_positions': batch_start_positions, 'end_positions': batch_end_positions, 
              'history_answer_marker': batch_history_answer_marker, 'yesno': batch_yesno, 'followup': batch_followup, 
              'metadata': batch_metadata}
    return feed_dict

def convert_one_example_to_example_variations(example, max_considered_history_turns, append_self):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []

    # if the example is the first question in the dialog, it does not contain history answers, 
    # so we simply append it.
    if len(example.metadata['tok_history_answer_markers']) == 0:
        example.metadata['history_turns'] = []
        new_examples.append(example)
    else:
        for history_turn, marker, history_turn_text in zip(
                example.metadata['history_turns'][- max_considered_history_turns:], 
                example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                example.metadata['history_turns_text'][- max_considered_history_turns:]):
            each_new_example = deepcopy(example)
            each_new_example.history_answer_marker = marker
            each_new_example.metadata['history_turns'] = [history_turn]
            each_new_example.metadata['tok_history_answer_markers'] = [marker]
            each_new_example.metadata['history_turns_text'] = [history_turn_text]
            new_examples.append(each_new_example)
            
        if append_self:
            # after the variations that contain histories, we append an example that is without any 
            # history. If the the current question is topic shift, all the attention weights should be
            # on this no-history variation.
            each_new_example = deepcopy(example)
            each_new_example.history_answer_marker = [0] * len(example.metadata['tok_history_answer_markers'][0])
            each_new_example.metadata['history_turns'] = []
            each_new_example.metadata['tok_history_answer_markers'] = []
            each_new_example.metadata['history_turns_text'] = []
            new_examples.append(each_new_example)
             
    return new_examples

def convert_examples_to_example_variations(examples, max_considered_history_turns, FLAGS):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []
    for example in examples:
        # if the example is the first question in the dialog, it does not contain history answers, 
        # so we simply append it.
        if len(example.metadata['tok_history_answer_markers']) == 0:
            example.metadata['history_turns'] = []
            new_examples.append(example)
        else:
            for history_turn, marker, history_turn_text in zip(
                    example.metadata['history_turns'][- max_considered_history_turns:], 
                    example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                    example.metadata['history_turns_text'][- max_considered_history_turns:]):
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = marker
                each_new_example.metadata['history_turns'] = [history_turn]
                each_new_example.metadata['tok_history_answer_markers'] = [marker]
                each_new_example.metadata['history_turns_text'] = [history_turn_text]
                new_examples.append(each_new_example)
                
            if FLAGS.append_self:
                # after the variations that contain histories, we append an example that is without any 
                # history. If the the current question is topic shift, all the attention weights should be
                # on this no-history variation.
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = [0] * len(example.metadata['tok_history_answer_markers'][0])
                each_new_example.metadata['history_turns'] = []
                each_new_example.metadata['tok_history_answer_markers'] = []
                each_new_example.metadata['history_turns_text'] = []
                new_examples.append(each_new_example)
             
    return new_examples

def convert_one_example_to_example_variations_with_question_reformulated(example, max_considered_history_turns):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []
    if len(example.metadata['tok_history_answer_markers']) == 0:
        example.metadata['history_turns'] = []
        new_examples.append(example)
    else:
        immediate_previous_history_question = example.metadata['history_turns_text'][-1][0]
        for history_turn, marker, history_turn_text in zip(
                example.metadata['history_turns'][- max_considered_history_turns:], 
                example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                example.metadata['history_turns_text'][- max_considered_history_turns:]):
            each_new_example = deepcopy(example)
            each_new_example.history_answer_marker = marker
            each_new_example.metadata['history_turns'] = [history_turn]
            each_new_example.metadata['tok_history_answer_markers'] = [marker]
            each_new_example.metadata['history_turns_text'] = [history_turn_text]
            each_new_example.question_text = immediate_previous_history_question + each_new_example.question_text
            new_examples.append(each_new_example)
    return new_examples

def convert_examples_to_example_variations_with_question_reformulated(examples, max_considered_history_turns):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []
    for example in examples:
        # if the example is the first question in the dialog, it does not contain history answers, 
        # so we simply append it.
        if len(example.metadata['tok_history_answer_markers']) == 0:
            example.metadata['history_turns'] = []
            new_examples.append(example)
        else:
            immediate_previous_history_question = example.metadata['history_turns_text'][-1][0]
            for history_turn, marker, history_turn_text in zip(
                    example.metadata['history_turns'][- max_considered_history_turns:], 
                    example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                    example.metadata['history_turns_text'][- max_considered_history_turns:]):
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = marker
                each_new_example.metadata['history_turns'] = [history_turn]
                each_new_example.metadata['tok_history_answer_markers'] = [marker]
                each_new_example.metadata['history_turns_text'] = [history_turn_text]
                each_new_example.question_text = immediate_previous_history_question + each_new_example.question_text
                new_examples.append(each_new_example)
    return new_examples

def convert_one_example_to_variations_and_then_features(example, example_index, tokenizer, max_seq_length, 
                                doc_stride, max_query_length, max_considered_history_turns, reformulate_question, front_padding, append_self, unique_id):
    # different from the "convert_examples_to_features" in cqa_supports.py, we return two masks with the feature (example/variaton trackers).
    # the first mask is the example index, and the second mask is the variation index. Wo do this to keep track of the features generated
    # by different examples and variations.
    
    all_features = []
    example_features_nums = [] # keep track of how many features are generated from the same example (regardless of example variations)
    example_tracker = []
    variation_tracker = []
    # matching_signals_dict = {}
    next_unique_id = unique_id
    
    
    example_features_num = []
    if reformulate_question:
        variations = convert_one_example_to_example_variations_with_question_reformulated(example, max_considered_history_turns)
    else:
        variations = convert_one_example_to_example_variations(example, max_considered_history_turns, append_self)
    for variation_index, variation in enumerate(variations):
        features = convert_one_example_to_features([variation], tokenizer, max_seq_length, doc_stride, max_query_length, front_padding)
        
        # the example_index and unique_id in features are wrong due to the generation of example variations.
        # we fix them here.
        for i in range(len(features)):
            features[i].example_index = example_index
            features[i].unique_id = next_unique_id
            next_unique_id += 1
        all_features.extend(features)
        variation_tracker.extend([variation_index] * len(features))
        example_tracker.extend([example_index] * len(features))
        example_features_num.append(len(features))
    # every variation of the same example should generate the same amount of features
    assert len(set(example_features_num)) == 1
    example_features_nums.append(example_features_num[0]) 
    assert len(all_features) == len(example_tracker)
    assert len(all_features) == len(variation_tracker)
    # return all_features, example_tracker, variation_tracker, example_features_nums, matching_signals_dict
    return all_features, example_tracker, variation_tracker, example_features_nums, next_unique_id

def convert_examples_to_variations_and_then_features(examples, tokenizer, max_seq_length, 
                                doc_stride, max_query_length, max_considered_history_turns, is_training, FLAGS):
    # different from the "convert_examples_to_features" in cqa_supports.py, we return two masks with the feature (example/variaton trackers).
    # the first mask is the example index, and the second mask is the variation index. Wo do this to keep track of the features generated
    # by different examples and variations.
    
    all_features = []
    example_features_nums = [] # keep track of how many features are generated from the same example (regardless of example variations)
    example_tracker = []
    variation_tracker = []
    # matching_signals_dict = {}
    unique_id = 1000000000
    
    
    # when training, we shuffle the data for more stable training.
    # we shuffle here so that we do not need to shuffle when generating batches
    num_examples = len(examples)    
    if is_training:
        np.random.seed(0)
        idx = np.random.permutation(num_examples)
        examples_shuffled = np.asarray(examples)[idx]
    else:
        examples_shuffled = np.asarray(examples)
    
    for example_index, example in enumerate(examples_shuffled):
        example_features_num = []
        if FLAGS.reformulate_question:
            variations = convert_examples_to_example_variations_with_question_reformulated([example], max_considered_history_turns)
        else:
            variations = convert_examples_to_example_variations([example], max_considered_history_turns)
        for variation_index, variation in enumerate(variations):
            features = convert_examples_to_features([variation], tokenizer, max_seq_length, doc_stride, max_query_length, is_training)
            # matching_signals = extract_matching_signals(variation, glove, tfidf_vectorizer)
            # matching_signals_dict[(example_index, variation_index)] = matching_signals
            
            # the example_index and unique_id in features are wrong due to the generation of example variations.
            # we fix them here.
            for i in range(len(features)):
                features[i].example_index = example_index
                features[i].unique_id = unique_id
                unique_id += 1
            all_features.extend(features)
            variation_tracker.extend([variation_index] * len(features))
            example_tracker.extend([example_index] * len(features))
            example_features_num.append(len(features))
        # every variation of the same example should generate the same amount of features
        assert len(set(example_features_num)) == 1
        example_features_nums.append(example_features_num[0]) 
    assert len(all_features) == len(example_tracker)
    assert len(all_features) == len(variation_tracker)
    # return all_features, example_tracker, variation_tracker, example_features_nums, matching_signals_dict
    return all_features, example_tracker, variation_tracker, example_features_nums


def get_turn_features(metadata):
    # extract current turn id, history turn id from metadata as a part of states
    res = []
    for m in metadata:
        if len(m['history_turns']) > 0:
            history_turn_id = m['history_turns'][0]
        else:
            history_turn_id = 0
        res.append([m['turn'], history_turn_id, m['turn'] - history_turn_id])
    return res

def fix_history_answer_marker_for_bhae(sub_batch_history_answer_marker, turn_features):
    res = []
    for marker, turn_feature in zip(sub_batch_history_answer_marker, turn_features):
        turn_diff = turn_feature[2]
        marker = np.asarray(marker)
        marker[marker == 1] = turn_diff
        res.append(marker.tolist())
        
    return res