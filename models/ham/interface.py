from models.ham.cqa_supports import *
from models.ham.cqa_model import *
from models.ham.cqa_gen_batches import *
from models.ham.cqa_rl_supports import *
import os
import tensorflow as tf
import models.ham.modeling as modeling
import models.ham.tokenization as tokenization


class BertHAM():
    def __init__(self, args):
        self.QA_history = []
        self.args=args

        bert_config = modeling.BertConfig.from_json_file(self.args.bert_config_file)

        # tf Graph input
        self.unique_ids = tf.placeholder(tf.int32, shape=[None], name='unique_ids')
        self.input_ids = tf.placeholder(tf.int32, shape=[None, self.args.max_seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, self.args.max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, self.args.max_seq_length], name='segment_ids')
        self.start_positions = tf.placeholder(tf.int32, shape=[None], name='start_positions')
        self.end_positions = tf.placeholder(tf.int32, shape=[None], name='end_positions')
        self.history_answer_marker = tf.placeholder(tf.int32, shape=[None, self.args.max_seq_length], name='history_answer_marker')
        self.training = tf.placeholder(tf.bool, name='training')
        self.yesno_labels = tf.placeholder(tf.int32, shape=[None], name='yesno_labels')
        self.followup_labels = tf.placeholder(tf.int32, shape=[None], name='followup_labels')

        # a unique combo of (e_tracker, f_tracker) is called a slice
        self.slice_mask = tf.placeholder(tf.int32, shape=[self.args.predict_batch_size], name='slice_mask') 
        self.slice_num = tf.placeholder(tf.int32, shape=None, name='slice_num') 
        # for auxiliary loss
        self.aux_start_positions = tf.placeholder(tf.int32, shape=[None], name='aux_start_positions')
        self.aux_end_positions = tf.placeholder(tf.int32, shape=[None], name='aux_end_positions')

        bert_representation, cls_representation = bert_rep(
                bert_config=bert_config,
                is_training=self.training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                segment_ids=self.segment_ids,
                history_answer_marker=self.history_answer_marker,
                use_one_hot_embeddings=False
                )

        reduce_mean_representation = tf.reduce_mean(bert_representation, axis=1)
        reduce_max_representation = tf.reduce_max(bert_representation, axis=1) 

        if self.args.history_attention_input == 'CLS':
            history_attention_input = cls_representation    
        elif self.args.history_attention_input == 'reduce_mean':
            history_attention_input = reduce_mean_representation
        elif self.args.history_attention_input == 'reduce_max':
            history_attention_input = reduce_max_representation
        else:
            print('FLAGS.history_attention_input not specified')
    
        if self.args.mtl_input == 'CLS':
            mtl_input = cls_representation    
        elif self.args.mtl_input == 'reduce_mean':
            mtl_input = reduce_mean_representation
        elif self.args.mtl_input == 'reduce_max':
            mtl_input = reduce_max_representation
        else:
            print('FLAGS.mtl_input not specified')

        if self.args.disable_attention:
            new_bert_representation, new_mtl_input, self.attention_weights = disable_history_attention_net(bert_representation, 
                                                                                            history_attention_input, mtl_input, 
                                                                                            self.slice_mask,
                                                                                            self.slice_num, self.args)
        else:
            if self.args.fine_grained_attention:
                new_bert_representation, new_mtl_input, self.attention_weights = fine_grained_history_attention_net(bert_representation, 
                                                                                                mtl_input,
                                                                                                self.slice_mask,
                                                                                                self.slice_num, self.args)

            else:
                new_bert_representation, new_mtl_input, self.attention_weights = history_attention_net(bert_representation, 
                                                                                                history_attention_input, mtl_input,
                                                                                                self.slice_mask,
                                                                                                self.slice_num, self.args)

        (self.start_logits, self.end_logits) = cqa_model(new_bert_representation, self.args)
        self.yesno_logits = yesno_model(new_mtl_input)
        self.followup_logits = followup_model(new_mtl_input)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        if self.args.init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, self.args.init_checkpoint)
            tf.train.init_from_checkpoint(self.args.init_checkpoint, assignment_map)

        # compute loss
        seq_length = modeling.get_shape_list(self.input_ids)[1]
        def compute_loss(logits, positions):
            one_hot_positions = tf.one_hot(
                positions, depth=seq_length, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
            return loss

        start_loss = compute_loss(self.start_logits, self.start_positions)
        end_loss = compute_loss(self.end_logits, self.end_positions)

        yesno_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.yesno_logits, labels=self.yesno_labels))
        followup_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.followup_logits, labels=self.followup_labels))

        if self.args.MTL:
            cqa_loss = (start_loss + end_loss) / 2.0
            if self.args.MTL_lambda < 1:
                self.total_loss = self.args.MTL_mu * cqa_loss * cqa_loss + self.args.MTL_lambda * yesno_loss + self.args.MTL_lambda * followup_loss
            else:
                self.total_loss = cqa_loss + yesno_loss + followup_loss

        else:
            self.total_loss = (start_loss + end_loss) / 2.0

        # Initializing the variables
        init = tf.global_variables_initializer()
        tf.get_default_graph().finalize()
        self.session = tf.Session()
        self.session.run(init)
    
    def tokenizer(self):
        tokenizer = tokenization.FullTokenizer(vocab_file=self.args.vocab_file, do_lower_case=self.args.do_lower_case)
        return tokenizer
    
    def load_partial_examples(self, partial_eval_examples_file):
        paragraphs = read_partial_quac_examples_extern(input_file=partial_eval_examples_file)
        return paragraphs
    
    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer):
        question = partial_example.question_text
        turn = int(partial_example.qas_id.split("#")[1])
        char_to_word_offset = partial_example.char_to_word_offset
        example = read_one_quac_example_extern(partial_example, self.QA_history, char_to_word_offset, self.args.history_len, self.args.use_history_answer_marker, self.args.only_history_answer)
        val_features, val_example_tracker, val_variation_tracker, val_example_features_nums, unique_id = convert_one_example_to_variations_and_then_features(
                                                example=example, example_index=example_idx, tokenizer=tokenizer, 
                                                max_seq_length=self.args.max_seq_length, doc_stride=self.args.doc_stride, 
                                                max_query_length=self.args.max_query_length, max_considered_history_turns=self.args.max_considered_history_turns,
                                                reformulate_question=self.args.reformulate_question, front_padding=self.args.front_padding, append_self=self.args.append_self,unique_id=unique_id)

        val_batch = cqa_gen_example_aware_batch_single(val_features, val_example_tracker, val_variation_tracker, self.args.predict_batch_size)

        batch_results = []
        batch_features, batch_slice_mask, batch_slice_num, output_features = val_batch

        fd = convert_features_to_feed_dict(batch_features) # feed_dict
        fd_output = convert_features_to_feed_dict(output_features)

        if self.args.better_hae:
            turn_features = get_turn_features(fd['metadata'])
            fd['history_answer_marker'] = fix_history_answer_marker_for_bhae(fd['history_answer_marker'], turn_features)

        if self.args.history_ngram != 1:                     
            batch_slice_mask, group_batch_features = group_histories(batch_features, fd['history_answer_marker'], 
                                                                batch_slice_mask, batch_slice_num)
            fd = convert_features_to_feed_dict(group_batch_features)

        feed_dict={self.unique_ids: fd['unique_ids'], self.input_ids: fd['input_ids'], 
                    self.input_mask: fd['input_mask'], self.segment_ids: fd['segment_ids'], 
                    self.start_positions: fd_output['start_positions'], self.end_positions: fd_output['end_positions'], 
                    self.history_answer_marker: fd['history_answer_marker'], self.slice_mask: batch_slice_mask, 
                    self.slice_num: batch_slice_num,
                    self.aux_start_positions: fd['start_positions'], self.aux_end_positions: fd['end_positions'],
                    self.yesno_labels: fd_output['yesno'], self.followup_labels: fd_output['followup'], self.training: False}

        start_logits_res, end_logits_res, yesno_logits_res, followup_logits_res, batch_total_loss, attention_weights_res = self.session.run([self.start_logits, self.end_logits, self.yesno_logits, self.followup_logits, 
                                        self.total_loss, self.attention_weights], feed_dict=feed_dict)

        for each_unique_id, each_start_logits, each_end_logits, each_yesno_logits, each_followup_logits in zip(fd_output['unique_ids'], start_logits_res, end_logits_res, yesno_logits_res, followup_logits_res):
            
            each_unique_id = int(each_unique_id)
            each_start_logits = [float(x) for x in each_start_logits.flat]
            each_end_logits = [float(x) for x in each_end_logits.flat]
            each_yesno_logits = [float(x) for x in each_yesno_logits.flat]
            each_followup_logits = [float(x) for x in each_followup_logits.flat]
            batch_results.append(RawResult(unique_id=each_unique_id, start_logits=each_start_logits, 
                                        end_logits=each_end_logits, yesno_logits=each_yesno_logits,
                                        followup_logits=each_followup_logits))

        pred_text, pred_answer_s, pred_answer_e, pred_yesno, pred_followup = get_last_prediction(example, example_idx, output_features, batch_results, self.args.n_best_size, self.args.max_answer_length, self.args.do_lower_case)

        self.QA_history.append((turn, question, (pred_text, pred_answer_s, pred_answer_e)))

        return pred_text, unique_id