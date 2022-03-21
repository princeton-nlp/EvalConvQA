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
"""Run BERT on QuAC."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from io import open
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# import wandb

from modeling import BertForQuAC, RobertaForQuAC

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, WEIGHTS_NAME, CONFIG_NAME

from run_quac_dataset_utils import read_quac_train_examples, convert_train_examples_to_features

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--type",
                        default='bert',
                        type=str,
                        choices=['roberta', 'bert'],
                        required=True,
                        help=".")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model checkpoints and predictions will be written."
    )

    # Other parameters
    parser.add_argument(
        "--train_file",
        default="",
        type=str,
        help="QuAC training file")
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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=48,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.06,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
        "of training.")
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
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal QuAC evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
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
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help="")
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--logfile',
                        type=str,
                        default=None,
                        help='Which file to keep log.')
    parser.add_argument('--logmode',
                        type=str,
                        default=None,
                        help='logging mode, `w` or `a`')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='no tensor board')
    parser.add_argument('--qa_tag',
                        action='store_true',
                        help='add qa tag or not')
    parser.add_argument('--history_len',
                        type=int,
                        default=2,
                        help='length of history')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--rationale_beta', type=float, default=100,
                        help="Multiplier for rationale loss.")

    args = parser.parse_args()
    print(args)

    if args.type == "bert":
        model_class = BertForQuAC
        if not args.do_lower_case:
            logger.warn("You probably want to use --do_lower_case when using BERT.")
    elif args.type == "roberta":
        model_class = RobertaForQuAC
        if args.do_lower_case:
            logger.warn("You probably do not want to use --do_lower_case when using RoBERTa.")
    else:
        raise NotImplementedError

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filename=args.logfile,
        filemode=args.logmode)

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    # wandb.init(entity='huihanlhh',project='quac')
    # wandb.config.update(args)
    if args.match_metric not in ['em','f1']:
        raise ValueError("Match metric has to be em or f1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError(
            "`do_train` must be True."
        )

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir
    ) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    if args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path)
        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(device)
        model.beta = args.rationale_beta

    if args.do_train:
        if args.local_rank in [-1, 0] and args.tensorboard:
            from tensorboardX import SummaryWriter
            tb_writer = SummaryWriter()
        # Prepare data loader
        cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(
            args.type, str(args.max_seq_length), str(args.doc_stride),
            str(args.max_query_length), str(args.max_answer_length),
            str(args.history_len), str(args.qa_tag))
        cached_train_examples_file = args.train_file + '_examples_{0}_{1}_{2}.pk'.format(
            args.type, str(args.history_len), str(args.qa_tag))

        # try train_examples
        try:
            with open(cached_train_examples_file, "rb") as reader:
                train_examples = pickle.load(reader)
        except:
            logger.info("  No cached file %s", cached_train_examples_file)
            train_examples = read_quac_train_examples(input_file=args.train_file,
                                                tokenizer=tokenizer,
                                                history_len=args.history_len,
                                                add_QA_tag=args.qa_tag,
                                                )
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train examples into cached file %s",
                            cached_train_examples_file)
                with open(cached_train_examples_file, "wb") as writer:
                    pickle.dump(train_examples, writer)


        # try train_features
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            logger.info("  No cached file %s", cached_train_features_file)
            train_features = convert_train_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length
            )
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s",
                            cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)


        all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                       dtype=torch.long)
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long)
        all_rational_mask = torch.tensor(
            [f.rational_mask for f in train_features], dtype=torch.long)
        all_cls_idx = torch.tensor([f.cls_idx for f in train_features],
                                   dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, all_start_positions,
                                   all_end_positions, all_rational_mask,
                                   all_cls_idx)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        num_train_optimization_steps = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs
        # if args.local_rank != -1:
        #     num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            args.weight_decay
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(
                    tqdm(train_dataloader,
                         desc="Iteration",
                         disable=args.local_rank not in [-1, 0])):
                batch = tuple(
                    t.to(device)
                    for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions, rational_mask, yesno_options, cls_idx = batch
                loss = model(input_ids, segment_ids, input_mask,
                             start_positions, end_positions, rational_mask, yesno_options,
                             cls_idx)
                # loss = gather(loss, 0)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if args.tensorboard:
                            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                            tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        else:
                            logger.info('Step: {}\tLearning rate: {}\tLoss: {}\t'.format(global_step, scheduler.get_last_lr()[0], (tr_loss - logging_loss) / args.logging_steps))
                        logging_loss = tr_loss

    if args.do_train and (args.local_rank == -1
                          or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)

if __name__ == "__main__":
    main()
