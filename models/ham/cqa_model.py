from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import six
import tensorflow as tf
import models.ham.modeling as modeling

def bert_rep(bert_config, is_training, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        history_answer_marker=history_answer_marker,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()
    sent_rep = model.get_pooled_output()
    return final_hidden, sent_rep


def bert_segment_rep(final_hidden):
    first_token_tensor = tf.squeeze(final_hidden[:, 0:1, :], axis=1)    
    return first_token_tensor

def cqa_model(final_hidden, FLAGS):
#     final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
#     batch_size = final_hidden_shape[0]
#     seq_length = final_hidden_shape[1]
#     hidden_size = final_hidden_shape[2]

    final_hidden_shape = tf.shape(final_hidden)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/cqa/output_weights", [2, FLAGS.bert_hidden],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/cqa/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)

def aux_cqa_model(final_hidden, FLAGS):

    final_hidden_shape = tf.shape(final_hidden)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/cqa/aux_output_weights", [2, FLAGS.bert_hidden],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/cqa/aux_output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)

def yesno_model(sent_rep):
    logits = tf.layers.dense(sent_rep, 3, activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='yesno_model')
    return logits

def followup_model(sent_rep):
    logits = tf.layers.dense(sent_rep, 3, activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='followup_model')
    return logits

def history_attention_net(bert_representation, history_attention_input, mtl_input, slice_mask, slice_num, FLAGS):
    
    # 12 * 768 --> e.g. 20 * 768
    history_attention_input = tf.pad(history_attention_input, [[0, FLAGS.train_batch_size - slice_num], [0, 0]])  
    
    # the splits contains 12 feature groups. e.g. the first might be 4 * 768 (the number 4 is is just an example)
    splits = tf.split(history_attention_input, slice_mask, 0)
    
    # --> 11 * 768
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0]])   
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    
    # --> 12 * 11 * 768
    input_tensor = tf.stack(padded, axis=0)
    input_tensor.set_shape([FLAGS.train_batch_size, FLAGS.max_history_turns, FLAGS.bert_hidden])
    
    if FLAGS.history_attention_hidden:
        hidden = tf.layers.dense(input_tensor, 100, activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_hidden')
        logits = tf.layers.dense(hidden, 1, activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
    else:
        # --> 12 * 11 * 1
        logits = tf.layers.dense(input_tensor, 1, activation=None,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
    # --> 12 * 11
    logits = tf.squeeze(logits, axis=2)
    
    # mask: 12 * 11
    logits_mask = tf.sequence_mask(slice_mask, FLAGS.max_history_turns, dtype=tf.float32)
    logits_mask = tf.reverse(logits_mask, axis=[1])
    exp_logits_masked = tf.exp(logits) * logits_mask 
    
    # --> e.g. 4 * 11
    exp_logits_masked = tf.slice(exp_logits_masked, [0, 0], [slice_num, -1])
    probs = exp_logits_masked / tf.reduce_sum(exp_logits_masked, axis=1, keepdims=True)
    
    # e.g. 4 * 11 * 768
    input_tensor = tf.slice(input_tensor, [0, 0, 0], [slice_num, -1, -1])
    
    # 4 * 11 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    mtl_input = tf.pad(mtl_input, [[0, FLAGS.train_batch_size - slice_num], [0, 0]]) 
    splits = tf.split(mtl_input, slice_mask, 0)
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0]])   
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    mtl_input = tf.stack(padded, axis=0)
    mtl_input = tf.slice(mtl_input, [0, 0, 0], [slice_num, -1, -1])
    
    
    # 4 * 768
    new_mtl_input = tf.reduce_sum(mtl_input * probs, axis=1)
    
    # after slicing, the shape information is lost, we rest it
    new_mtl_input.set_shape([None, FLAGS.bert_hidden])
    
    
    # 12 * 384 * 768 --> 20 * 384 * 768
    bert_representation = tf.pad(bert_representation, [[0, FLAGS.train_batch_size - slice_num], [0, 0], [0, 0]])
    splits = tf.split(bert_representation, slice_mask, 0)
    
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0], [0, 0]])
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
        
    # --> 12 * 11 * 384 * 768
    token_tensor = tf.stack(padded, axis=0)
    # --> 4 * 11 * 384 * 768
    token_tensor = tf.slice(token_tensor, [0, 0, 0, 0], [slice_num, -1, -1, -1])
    
    # 4 * 11 * 1 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    # 4 * 384 * 768
    new_bert_representation = tf.reduce_sum(token_tensor * probs, axis=1)
    new_bert_representation.set_shape([None, FLAGS.max_seq_length, FLAGS.bert_hidden])
    
    return new_bert_representation, new_mtl_input, tf.squeeze(probs)

def disable_history_attention_net(bert_representation, history_attention_input, mtl_input, slice_mask, slice_num, FLAGS):
    
    # 12 * 768 --> e.g. 20 * 768
    history_attention_input = tf.pad(history_attention_input, [[0, FLAGS.train_batch_size - slice_num], [0, 0]])  
    
    # the splits contains 12 feature groups. e.g. the first might be 4 * 768 (the number 4 is is just an example)
    splits = tf.split(history_attention_input, slice_mask, 0)
    
    # --> 11 * 768
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0]])   
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    
    # --> 12 * 11 * 768
    input_tensor = tf.stack(padded, axis=0)
    input_tensor.set_shape([FLAGS.train_batch_size, FLAGS.max_history_turns, FLAGS.bert_hidden])
    
    # we assign equal logits, which means equal attention weights. this helps us to see whether the attention net works as expected
    logits = tf.ones((FLAGS.train_batch_size, FLAGS.max_history_turns))

    
    # mask: 12 * 11
    logits_mask = tf.sequence_mask(slice_mask, FLAGS.max_history_turns, dtype=tf.float32)
    logits_mask = tf.reverse(logits_mask, axis=[1])
    exp_logits_masked = tf.exp(logits) * logits_mask 
    
    # --> e.g. 4 * 11
    exp_logits_masked = tf.slice(exp_logits_masked, [0, 0], [slice_num, -1])
    probs = exp_logits_masked / tf.reduce_sum(exp_logits_masked, axis=1, keepdims=True)
    
    # e.g. 4 * 11 * 768
    input_tensor = tf.slice(input_tensor, [0, 0, 0], [slice_num, -1, -1])
    
    # 4 * 11 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    mtl_input = tf.pad(mtl_input, [[0, FLAGS.train_batch_size - slice_num], [0, 0]]) 
    splits = tf.split(mtl_input, slice_mask, 0)
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0]])   
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    mtl_input = tf.stack(padded, axis=0)
    mtl_input = tf.slice(mtl_input, [0, 0, 0], [slice_num, -1, -1])
    
    
    # 4 * 768
    new_mtl_input = tf.reduce_sum(mtl_input * probs, axis=1)
    
    # after slicing, the shape information is lost, we rest it
    new_mtl_input.set_shape([None, FLAGS.bert_hidden])
    
    
    # 12 * 384 * 768 --> 20 * 384 * 768
    bert_representation = tf.pad(bert_representation, [[0, FLAGS.train_batch_size - slice_num], [0, 0], [0, 0]])
    splits = tf.split(bert_representation, slice_mask, 0)
    
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0], [0, 0]])
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
        
    # --> 12 * 11 * 384 * 768
    token_tensor = tf.stack(padded, axis=0)
    # --> 4 * 11 * 384 * 768
    token_tensor = tf.slice(token_tensor, [0, 0, 0, 0], [slice_num, -1, -1, -1])
    
    # 4 * 11 * 1 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    # 4 * 384 * 768
    new_bert_representation = tf.reduce_sum(token_tensor * probs, axis=1)
    new_bert_representation.set_shape([None, FLAGS.max_seq_length, FLAGS.bert_hidden])
    
    return new_bert_representation, new_mtl_input, tf.squeeze(probs)

def fine_grained_history_attention_net(bert_representation, mtl_input, slice_mask, slice_num, FLAGS):
    
    # first concat the bert_representation and mtl_input togenther
    # so that we can process them together
    # shape for bert_representation: 12 * 384 * 768, shape for mtl_input: 12 * 768
    # after concat: 12 * 385 * 768
    
    # 12 * 385 * 768 --> 20 * 385 * 768
    bert_representation = tf.concat([bert_representation, tf.expand_dims(mtl_input, axis=1)], axis=1)
    bert_representation = tf.pad(bert_representation, [[0, FLAGS.train_batch_size - slice_num], [0, 0], [0, 0]])
    splits = tf.split(bert_representation, slice_mask, 0)
    # print(tf.shape(splits[0]))
    
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0], [0, 0]])
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
        
    # --> 12 * 11 * 385 * 768
    token_tensor = tf.stack(padded, axis=0)
    # print(tf.shape(token_tensor))
    token_tensor.set_shape([FLAGS.train_batch_size, FLAGS.max_history_turns, FLAGS.max_seq_length + 1, FLAGS.bert_hidden])
    
    # --> 12 * 385 * 11 * 768
    token_tensor_t = tf.transpose(token_tensor, [0, 2, 1, 3])
    
    if FLAGS.history_attention_hidden:
        hidden = tf.layers.dense(token_tensor_t, 100, activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_hidden')
        logits = tf.layers.dense(hidden, 1, activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
    else:
        # --> 12 * 385 * 11 * 1
        logits = tf.layers.dense(token_tensor_t, 1, activation=None,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
    
    # --> 12 * 385 * 11
    logits = tf.squeeze(logits, axis=-1)
    
    # mask: 12 * 11 --> after expand_dims: 12 * 1 * 11
    logits_mask = tf.sequence_mask(slice_mask, FLAGS.max_history_turns, dtype=tf.float32)
    logits_mask = tf.reverse(logits_mask, axis=[1])
    logits_mask = tf.expand_dims(logits_mask, axis=1)
    exp_logits_masked = tf.exp(logits) * logits_mask 
    
    # --> e.g. 4 * 385 * 11
    exp_logits_masked = tf.slice(exp_logits_masked, [0, 0, 0], [slice_num, -1, -1])
    probs = exp_logits_masked / tf.reduce_sum(exp_logits_masked, axis=2, keepdims=True)

    # --> 4 * 385 * 11 * 768
    token_tensor_t = tf.slice(token_tensor_t, [0, 0, 0, 0], [slice_num, -1, -1, -1])
    
    # 4 * 385 * 11 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    # 4 * 385 * 768
    new_bert_representation = tf.reduce_sum(token_tensor_t * probs, axis=2)
    
    new_bert_representation.set_shape([None, FLAGS.max_seq_length + 1, FLAGS.bert_hidden])
    
    new_bert_representation, new_mtl_input = tf.split(new_bert_representation, [FLAGS.max_seq_length, 1], axis=1)
    new_mtl_input = tf.squeeze(new_mtl_input, axis=1)
    
    return new_bert_representation, new_mtl_input, tf.squeeze(probs)
