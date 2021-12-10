from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
import sys
from io import open

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import collections
import json
import math
import os

import numpy as np

# import model interface
from interface import QAModel

from run_quac_eval_util import rewrite_with_coreference, filter_with_coreference, write_automatic_eval_result, write_invalid_category, load_context_indep_questions


logger = logging.getLogger(__name__)

def main():
    print("Program start")     
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--type",
                        default=None,
                        type=str,
                        required=True,
                        help="Aliases for model to evaluate. Eg. 'bert', 'ham', 'excord'.")
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        type=str,
        help=
        "The directory where the model checkpoints to be evaluated is written."
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
        help="Path to QuAC dev file")
    
    # Eval specific parameters

    parser.add_argument('--history_len',
                        type=int,
                        default=2,
                        help='length of history')
    parser.add_argument('--start_i', type=int, default=0, help="start passage index of evaluation")
    parser.add_argument('--end_i', type=int,
                        default=1000, help="end passage index of evaluation")
    parser.add_argument('--match_metric', type=str, default='f1', choices=['f1', 'em'])
    parser.add_argument('--add_background', action='store_true', help="Whether or not to add background section during validity evaluation")
    parser.add_argument('--skip_entity', action='store_true', help="Whether to ignore special entities in validity evaluation")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Turn on if replacing the original question with context-independent questions"
    )
    parser.add_argument(
        "--canard_path",
        default="/n/fs/nlp-data/conversational-qa/canard/test.json",
        type=str,
        help="The path to CANARD test set, which is QuAC dev set."
    )
    parser.add_argument(
        '--rewrite',
        action="store_true",
        help="Whether to rewrite the question with coreference model"
    )
    # Other parameters
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

    args = parser.parse_args()

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
    model_class = QAModel()
    model = model_class(args=args)
    tokenizer = model.tokenizer()

    # load predict file
    partial_eval_examples_file = args.predict_file
    partial_examples = model.load_partial_examples(partial_eval_examples_file)
    
    # load context independent questions if needed
    if args.replace:
        context_indep_questions = load_context_indep_questions(args.canard_path)

    # list of QuAC passages to evaluate on
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
        
        # read into a list of passage examples
        paragraph = partial_examples[passage_index]
        context_id = paragraph["context_id"]
        background = paragraph["background"]
        gold_answers = paragraph["gold_answers"] # a list of gold answers
        examples = paragraph["examples"]

        # clear model QA history for new passage
        model.QA_history = []
        
        predictions=[]
        for data_idx in range(len(examples)):
            partial_example = examples[data_idx]
            
            if args.rewrite:
                invalid, modified_question = rewrite_with_coreference(partial_example, background, gold_answers, model.QA_history, history_len=2, match_metric=args.match_metric, add_background=args.add_background, skip_entity=args.skip_entity)
            else:
                invalid = filter_with_coreference(partial_example, background, gold_answers, model.QA_history, history_len=2, match_metric=args.match_metric, add_background=args.add_background, rm_cannotanswer=args.rm_cannotanswer, skip_entity=args.skip_entity)
            
            if not invalid:
                logger.info("Valid {}".format(partial_example.qas_id))
                prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx)
                predictions.append((partial_example.qas_id, prediction))
            else:
                # Replace the original question with context independent question, also doing that in conversation history
                if args.replace:
                    if partial_example.qas_id in context_indep_questions:
                        logger.info("Replaced {}".format(partial_example.qas_id))
                        partial_example.question_text = context_indep_questions[partial_example.qas_id]
                        modified_turns["Replaced"].append((partial_example.qas_id, context_indep_questions[partial_example.qas_id]))
                    else:
                        logger.info("Not Found {}".format(partial_example.qas_id))
                        modified_turns["Not Found"].append(partial_example.qas_id)
                    
                    prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx)
                    predictions.append((partial_example.qas_id, prediction))
                elif args.rewrite:
                    logger.info("Rewritten {} to: \'{}\'".format(partial_example.qas_id, modified_question))
                    partial_example.question_text = modified_question
                    modified_turns["Rewritten"].append((partial_example.qas_id, modified_question))
                    prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx)
                    predictions.append((partial_example.qas_id, prediction))
                else:
                    modified_turns["Filtered"].append(partial_example.qas_id)
                
        invalid_turns_dict[passage_index] = modified_turns
        evaluation_result[passage_index] = {"CID": context_id, "Predictions": predictions}

    write_automatic_eval_result(json_file=evaluation_file, evaluation_result=evaluation_result)
    write_invalid_category(json_file=invalid_turns_file, skip_dictionary=invalid_turns_dict)
                

if __name__ == "__main__":
    main()
 
