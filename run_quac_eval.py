from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
import sys
from io import open

# import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm, trange
import collections
import json
import math
import os

import six

import numpy as np
from copy import deepcopy
import itertools
from time import time
import traceback

from run_quac_eval_util import rewrite_with_coreference, filter_with_coreference, write_automatic_eval_result, write_invalid_category, load_context_indep_questions


logger = logging.getLogger(__name__)

def main():
    print("Program start")     
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--type",
                        default='bert',
                        type=str,
                        choices=['graphflow', 'bert', 'ham', 'excord'],
                        required=True,
                        help="Aliases for model to evaluate. Eg. 'bert', 'ham', 'excord', 'graphflow.")
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        type=str,
        help="The output directory where the model checkpoints to be evaluated is written."
    )
    parser.add_argument(
        "--write_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where evaluation results will be stored."
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        required=True,
        help="Path to QuAC dev file. Download from: https://s3.amazonaws.com/my89public/quac/val_v0.2.json")
    
    # Eval specific parameters
    parser.add_argument('--history_len',
                        type=int,
                        default=2,
                        help='length of history')
    parser.add_argument('--start_i', type=int, default=0, help="start passage index of evaluation")
    parser.add_argument('--end_i', type=int,
                        default=1000, help="end passage index of evaluation")
    parser.add_argument('--match_metric', type=str, default='f1', choices=['f1', 'em'], help="which metric to use for detecting invalid questions")
    parser.add_argument('--add_background', action='store_true', help="Whether or not to add background section during validity evaluation")
    parser.add_argument('--skip_entity', action='store_true', help="Whether to ignore special entities (e.g. named entity) in validity evaluation")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Turn on if using Auto-Replace by replacing the original question with context-independent questions"
    )
    parser.add_argument(
        "--canard_path",
        default="/n/fs/nlp-data/conversational-qa/canard/test.json",
        type=str,
        help="The path to CANARD test set, which is QuAC dev set with context-independent questions."
    )
    parser.add_argument(
        '--rewrite',
        action="store_true",
        help="Turn on if using Auto-Rewrite by rewriting the question with coreference model"
    )
    parser.add_argument(
        '--pred',
        action="store_true",
        help="Turn on if using Auto-Pred by only using predicted answer as history without replacing or rewriting invalid questions."
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help=
        "Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument('--logfile',
                        type=str,
                        default=None,
                        help='Which file to keep log.')
    parser.add_argument('--logmode',
                        type=str,
                        default=None,
                        help='logging mode, `w` or `a`')
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal QuAC evaluation.")


    # Excord and Bert specific
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, roberta-base")
    
    # Bert specific
    parser.add_argument('--rationale_beta', type=float, default=0,
                        help="Multiplier for rationale loss.")

    # HAM specific args
    parser.add_argument('--bert_config_file', type=str, default="/n/fs/nlp-huihanl/conversational-qa/local/Bert4QuAC/other/wwm_uncased_L-24_H-1024_A-16/bert_config.json", help="bert_config.json for bert-large-uncased")
    parser.add_argument('--vocab_file', type=str, default="/n/fs/nlp-huihanl/conversational-qa/local/Bert4QuAC/other/attentive_history_selection/model/vocab.txt", help="downloadable from https://worksheets.codalab.org/worksheets/0xb92c7222574046eea830c26cd414faec")
    parser.add_argument('--init_checkpoint', type=str, default="/n/fs/nlp-huihanl/conversational-qa/local/Bert4QuAC/other/attentive_history_selection/model/model_52000.ckpt", help="downloadable from https://worksheets.codalab.org/worksheets/0xb92c7222574046eea830c26cd414faec")
    parser.add_argument('--train_batch_size', type=int, default=16, help="Set to 16 to match training batch size for the original model")
    parser.add_argument('--predict_batch_size', type=int, default=16, help="Set to 16 to match training batch size for the original model")
    parser.add_argument('--max_history_turns', type=int, default=11)
    parser.add_argument('--max_considered_history_turns', type=int, default=4)
    parser.add_argument('--MTL_lambda', type=float, default=0.1)
    parser.add_argument('--MTL_mu', type=float, default=0.8)
    parser.add_argument('--history_selection', type=str, default="previous_j")
    parser.add_argument('--history_attention_input', type=str, default="reduce_mean")
    parser.add_argument('--mtl_input', type=str, default="reduce_mean")
    parser.add_argument('--history_ngram', type=int, default=1)
    parser.add_argument('--bert_hidden', type=int, default=1024)
    parser.add_argument('--only_history_answer', action='store_true')
    parser.add_argument('--use_history_answer_marker', action='store_true')
    parser.add_argument('--better_hae', action='store_true')
    parser.add_argument('--MTL', action='store_true')
    parser.add_argument('--disable_attention', action='store_true')
    parser.add_argument('--history_attention_hidden', action='store_true')
    parser.add_argument('--reformulate_question', action='store_true')
    parser.add_argument('--front_padding', action='store_true')
    parser.add_argument('--fine_grained_attention', action='store_true')
    parser.add_argument('--append_self', action='store_true')

    # GraphFlow specific args
    parser.add_argument(
        "--embed_file",
        default='/n/fs/nlp-huihanl/conversational-qa/local/Bert4QuAC/glovecove/glove.840B.300d.txt',
        type=str,
        help="GloVE embedding file. Downloadable from glovecove.")
    parser.add_argument(
        "--saved_vocab_file",
        default='/n/fs/nlp-huihanl/conversational-qa/GraphFlow/data/quac/word_model_min_5',
        type=str,
        help="Saved vocab file after training.")
    parser.add_argument(
        "--pretrained",
        default='/n/fs/nlp-huihanl/conversational-qa/GraphFlow/out/quac/graphflow_dynamic_graph',
        type=str,
        help="Saved model after training.")
    
    # Processing data
    parser.add_argument(
        "--min_freq",
        default=5,
        type=int,
        help="")
    parser.add_argument(
        "--top_vocab",
        default=200000,
        type=int,
        help="")
    parser.add_argument(
        "--n_history",
        default=2,
        type=int,
        help="")
    parser.add_argument(
        "--max_turn_num",
        default=20,
        type=int,
        help="")
    parser.add_argument(
        "--no_pre_question",
        action="store_true",
        help="")
    parser.add_argument(
        "--no_pre_answer",
        action="store_true",
        help="")
    parser.add_argument(
        "--embed_type",
        default='glove',
        type=str,
        help="")
    parser.add_argument(
        "--vocab_embed_size",
        default=300,
        type=int,
        help="")
    parser.add_argument(
        "--fix_vocab_embed",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_qem",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_pos",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_ner",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_tf",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_ques_marker",
        action="store_true",
        help="")
    parser.add_argument(
        "--ctx_exact_match_embed_dim",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--ctx_pos_embed_dim",
        default=12,
        type=int,
        help="")
    parser.add_argument(
        "--ctx_ner_embed_dim",
        default=8,
        type=int,
        help="")
    parser.add_argument(
        "--answer_marker_embed_dim",
        default=10,
        type=int,
        help="")
    parser.add_argument(
        "--ques_marker_embed_dim",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--ques_turn_marker_embed_dim",
        default=5,
        type=int,
        help="")
    parser.add_argument(
        "--hidden_size",
        default=300,
        type=int,
        help="")
    parser.add_argument(
        "--word_dropout",
        default=0.3,
        type=float,
        help="")
    parser.add_argument(
        "--bert_dropout",
        default=0.4,
        type=float,
        help="")
    parser.add_argument(
        "--rnn_dropout",
        default=0.3,
        type=float,
        help="")
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="")
    parser.add_argument(
        "--bignn",
        action="store_true",
        help="")
    parser.add_argument(
        "--static_graph",
        action="store_true",
        help="")
    parser.add_argument(
        "--temporal_gnn",
        action="store_true",
        help="")
    parser.add_argument(
        "--ctx_graph_hops",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--ctx_graph_topk",
        default=10,
        type=int,
        help="")
    parser.add_argument(
        "--graph_learner_num_pers",
        default=1,
        type=int,
        help="")
    parser.add_argument(
        "--use_spatial_kernels",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_position_enc",
        action="store_true",
        help="")
    parser.add_argument(
        "--n_spatial_kernels",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--max_position_distance",
        default=160,
        type=int,
        help="")
    parser.add_argument(
        "--position_emb_size",
        default=50,
        type=int,
        help="")
    parser.add_argument(
        "--use_bert",
        action="store_true",
        help="")
    parser.add_argument(
        "--finetune_bert",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_bert_weight",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_bert_gamma",
        action="store_true",
        help="")
    parser.add_argument(
        "--bert_max_seq_len",
        default=500,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--bert_doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--bert_dim",
        default=1024,
        type=int,
        help="")
    parser.add_argument(
        "--bert_model",
        default='bert-large-uncased',
        type=str,
        help="")
    parser.add_argument(
        "--bert_layer_indexes",
        default=[0,24],
        nargs="+",
        help="")
    parser.add_argument(
        "--optimizer",
        default='adamax',
        type=str,
        help="")
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="")
    parser.add_argument(
        "--grad_clipping",
        default=10,
        type=int,
        help="")
    parser.add_argument(
        "--max_answer_len",
        default=35,
        type=int,
        help=
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help=
        "")
    parser.add_argument(
        "--grad_accumulated_steps",
        default=1,
        type=int,
        help=
        "")
    parser.add_argument(
        "--test_batch_size",
        default=1,
        type=int,
        help=
        "")
    parser.add_argument(
        "--max_epochs",
        default=1000,
        type=int,
        help=
        "")
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help=
        "")
    parser.add_argument(
        "--verbose",
        default=1000,
        type=int,
        help=
        "")
    parser.add_argument(
        "--unk_answer_threshold",
        type=float,
        default=0.3,
        help="",
    )
    parser.add_argument(
        "--out_predictions", action="store_true", help=""
    )
    parser.add_argument(
        "--predict_raw_text", action="store_true", help=""
    )
    parser.add_argument(
        "--out_pred_in_folder", action="store_true", help=""
    )
    parser.add_argument(
        "--shuffle", action="store_true", help=""
    )
    parser.add_argument(
        "--cuda_id",
        default=0,
        type=int,
        help=
        "")

    args = parser.parse_args()

    # Set model_class. You can add your own model here.
    if args.type == "bert":
        from models.bert.interface import BertOrg
        model_class = BertOrg
        if not args.do_lower_case:
            logger.warn("You probably want to use --do_lower_case when using BERT.")
    elif args.type == "ham":
        from models.ham.interface import BertHAM
        model_class = BertHAM
        if not args.do_lower_case:
            logger.warn("You probably want to use --do_lower_case when using BERT.")
        import tensorflow as tf
        tf.set_random_seed(args.seed)
        tf.logging.set_verbosity(tf.logging.INFO)
        device = None
    elif args.type == "excord":
        from models.excord.interface import Excord
        model_class = Excord
        if args.do_lower_case:
            logger.warn("Do not use --do_lower_case when using Excord.")
        import torch
        torch.manual_seed(args.seed)
    elif args.type == "graphflow":
        from models.graphflow.interface import GraphFlow
        model_class = GraphFlow
        if args.do_lower_case:
            logger.warn(
                "Do not use --do_lower_case when using GraphFlow")
    else:
        raise NotImplementedError

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filename=args.logfile,
        filemode=args.logmode)
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified."
        )

    if not os.path.exists(args.output_dir):
        raise ValueError(
            "Output directory {} does not contain model checkpoint.".format(args.output_dir))
    
    logger.info("***** Loading pre-trained model *****")
    logger.info("Model directory: %s", args.output_dir)
    
    # initialize model and tokenizer
    model = model_class(args=args)
    tokenizer = model.tokenizer()

    # load predict file
    partial_eval_examples_file = args.predict_file
    # load partially filled examples according to your model
    partial_examples = model.load_partial_examples(partial_eval_examples_file)
    
    # load context independent questions if needed
    if args.replace:
        context_indep_questions = load_context_indep_questions(args.canard_path)

    # Set list of QuAC passages to evaluate on
    passage_index_list = range(int(args.start_i), int(args.end_i))
    evaluation_result = {}

    if args.replace:
        # Opt to replace invalid questions with context independent questions
        option = "replace"
    elif args.rewrite:
        # Opt to rewrite invalid questions with coreference resolution model
        option = "rewrite"
    else:
        # Opt to filter out invalid questions
        option = "filter"

    evaluation_file = os.path.join(args.write_dir, "predictions_{}_{}_{}_{}_{}.json".format(args.type, option, "skipent" if args.skip_entity else "keptent", args.start_i, args.end_i))
    invalid_turns_file = os.path.join(args.write_dir, "modified_questions_{}_{}_{}_{}_{}.json".format(args.type, option, "skipent" if args.skip_entity else "keptent", args.start_i, args.end_i))
    
    next_unique_id=1000000000
    invalid_turns_dict = {}

    for passage_index in passage_index_list:
        modified_turns = {"Rewritten": [], "Replaced": [], "Not Found": [], "Filtered": []}
        
        passage_index = int(passage_index)

        # read into a list of paragraph examples
        paragraph = partial_examples[passage_index] 
        context_id = paragraph["context_id"]
        background = paragraph["background"]
        gold_answers = paragraph["gold_answers"] # a list of gold answers
        examples = paragraph["examples"]

        # clear model QA history for new passage
        model.QA_history = []

        # A quick hack for graphflow
        if args.type == "graphflow":
            model.history = []

        predictions=[]
        for data_idx in range(len(examples)):
            partial_example = examples[data_idx]

            # A quick hack for graphflow
            if args.type == "graphflow":
                attr_partial_example = model.convert_example(partial_example)
            else:
                attr_partial_example = partial_example
            
            if args.pred:
                invalid = False
            elif args.rewrite:
                invalid, modified_question = rewrite_with_coreference(attr_partial_example, background, gold_answers, model.QA_history, history_len=2, match_metric=args.match_metric, add_background=args.add_background, skip_entity=args.skip_entity)
            else:
                invalid = filter_with_coreference(attr_partial_example, background, gold_answers, model.QA_history, history_len=2, match_metric=args.match_metric, add_background=args.add_background, skip_entity=args.skip_entity)
            
            if not invalid:
                logger.info("Valid {}".format(attr_partial_example.qas_id))
                prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx, tokenizer=tokenizer)
                predictions.append((attr_partial_example.qas_id, prediction))
            else:
                # Replace the original question with context independent question, also doing that in conversation history
                if args.replace:
                    if attr_partial_example.qas_id in context_indep_questions:
                        logger.info("Replaced {}".format(attr_partial_example.qas_id))
                        # graphflow needs to annotate replaced questions at runtime
                        if args.type == "graphflow":
                            annotated_question = model.question_processor.process(context_indep_questions[attr_partial_example.qas_id])
                            partial_example["question"]["word"] = annotated_question["word"]
                        else:
                            partial_example.question_text = context_indep_questions[attr_partial_example.qas_id]
                        modified_turns["Replaced"].append((attr_partial_example.qas_id, context_indep_questions[attr_partial_example.qas_id]))
                    else:
                        logger.info("Not Found {}".format(attr_partial_example.qas_id))
                        modified_turns["Not Found"].append(attr_partial_example.qas_id)
                    
                    prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx, tokenizer=tokenizer)
                    predictions.append((attr_partial_example.qas_id, prediction))
                elif args.rewrite:
                    logger.info("Rewritten {} to: \'{}\'".format(attr_partial_example.qas_id, modified_question))
                    if args.type == "graphflow":
                        # graphflow needs to annotate rewritten questions at runtime
                        annotated_question = model.question_processor.process(modified_question)
                        partial_example["question"]["word"] = annotated_question["word"]
                    else:
                        partial_example.question_text = modified_question
                    modified_turns["Rewritten"].append((attr_partial_example.qas_id, modified_question))
                    prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx, tokenizer=tokenizer)
                    predictions.append((attr_partial_example.qas_id, prediction))
                else:
                    logger.info("Removed {}".format(attr_partial_example.qas_id))
                    modified_turns["Filtered"].append(attr_partial_example.qas_id)
                
        invalid_turns_dict[passage_index] = modified_turns
        evaluation_result[passage_index] = {"CID": context_id, "Predictions": predictions}
    
    write_automatic_eval_result(json_file=evaluation_file, evaluation_result=evaluation_result)
    write_invalid_category(json_file=invalid_turns_file, skip_dictionary=invalid_turns_dict)
                

if __name__ == "__main__":
    main()
 
