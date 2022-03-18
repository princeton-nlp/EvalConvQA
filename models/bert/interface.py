from models.bert.modeling import BertForQuAC, RobertaForQuAC
from transformers import AutoTokenizer
from models.bert.run_quac_dataset_utils import read_partial_quac_examples_extern, read_one_quac_example_extern, convert_one_example_to_features, recover_predicted_answer, RawResult

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class BertOrg():
    def __init__(self, args):
        self.args = args
        self.model = BertForQuAC.from_pretrained(self.args.model_name_or_path)
        self.QA_history = []
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        self.model = self.model.to(self.device)
    
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, do_lower_case=self.args.do_lower_case)
        return tokenizer
    
    def load_partial_examples(self, partial_eval_examples_file):
        paragraphs = read_partial_quac_examples_extern(partial_eval_examples_file)
        return paragraphs
    
    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer):
        question = partial_example.question_text
        turn = int(partial_example.qas_id.split("#")[1])
        example = read_one_quac_example_extern(partial_example, self.QA_history, history_len=2, add_QA_tag=False)
        
        curr_eval_features, next_unique_id= convert_one_example_to_features(example=example, unique_id=unique_id, example_index=example_idx, tokenizer=tokenizer, max_seq_length=self.args.max_seq_length,
                                    doc_stride=self.args.doc_stride, max_query_length=self.args.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in curr_eval_features],
                                            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in curr_eval_features],
                                    dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in curr_eval_features],
                                    dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0),
                                        dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_feature_index)
        # Run prediction for full data

        eval_dataloader = DataLoader(eval_data,
                                    sampler=None,
                                    batch_size=1)
        curr_results = []
        # Run prediction for current example
        for input_ids, input_mask, segment_ids, feature_indices in eval_dataloader:

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            print(type(input_ids[0]), type(input_mask[0]), type(segment_ids[0]))
            # Assume the logits are a list of one item
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_yes_logits, batch_no_logits, batch_unk_logits = self.model(
                    input_ids, segment_ids, input_mask)
            for i, feature_index in enumerate(feature_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                yes_logits = batch_yes_logits[i].detach().cpu().tolist()
                no_logits = batch_no_logits[i].detach().cpu().tolist()
                unk_logits = batch_unk_logits[i].detach().cpu().tolist()
                eval_feature = curr_eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                curr_results.append(
                    RawResult(unique_id=unique_id,
                                start_logits=start_logits,
                                end_logits=end_logits,
                                yes_logits=yes_logits,
                                no_logits=no_logits,
                                unk_logits=unk_logits))
        predicted_answer = recover_predicted_answer(
            example=example, features=curr_eval_features, results=curr_results, tokenizer=tokenizer, n_best_size=self.args.n_best_size, max_answer_length=self.args.max_answer_length,
            do_lower_case=self.args.do_lower_case, verbose_logging=self.args.verbose_logging)
        self.QA_history.append((turn, question, (predicted_answer, None, None)))
        return predicted_answer, next_unique_id

class RobertaOrg():
    def __init__(self, args, device):
        self.args = args
        self.model = RobertaForQuAC.from_pretrained(self.args.model_name_or_path)
        self.QA_history = []
        self.device = device
    
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, do_lower_case=self.args.do_lower_case)
        return tokenizer
    
    def load_partial_examples(self, cached_partial_eval_examples_file):
        paragraphs = read_partial_quac_examples_extern(cached_partial_eval_examples_file)
        return paragraphs
    
    def predict_one_human_turn(self, paragraph, question):
        return
    
    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer):
        question = partial_example.question_text
        turn = int(partial_example.qas_id.split("#")[1])
        example = read_one_quac_example_extern(partial_example, self.QA_history, history_len=2, add_QA_tag=False)
        
        curr_eval_features, next_unique_id= convert_one_example_to_features(example=example, unique_id=unique_id, example_index=example_idx, tokenizer=tokenizer, max_seq_length=self.args.max_seq_length,
                                    doc_stride=self.args.doc_stride, max_query_length=self.args.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in curr_eval_features],
                                            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in curr_eval_features],
                                    dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in curr_eval_features],
                                    dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0),
                                        dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_feature_index)
        # Run prediction for full data

        eval_dataloader = DataLoader(eval_data,
                                    sampler=None,
                                    batch_size=1)
        curr_results = []
        # Run prediction for current example
        for input_ids, input_mask, segment_ids, feature_indices in eval_dataloader:

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            # Assume the logits are a list of one item
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_yes_logits, batch_no_logits, batch_unk_logits = self.model(
                    input_ids, segment_ids, input_mask)
            for i, feature_index in enumerate(feature_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                yes_logits = batch_yes_logits[i].detach().cpu().tolist()
                no_logits = batch_no_logits[i].detach().cpu().tolist()
                unk_logits = batch_unk_logits[i].detach().cpu().tolist()
                eval_feature = curr_eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                curr_results.append(
                    RawResult(unique_id=unique_id,
                                start_logits=start_logits,
                                end_logits=end_logits,
                                yes_logits=yes_logits,
                                no_logits=no_logits,
                                unk_logits=unk_logits))
        predicted_answer = recover_predicted_answer(
            example=example, features=curr_eval_features, results=curr_results, n_best_size=self.args.n_best_size, max_answer_length=self.args.max_answer_length,
            do_lower_case=self.args.do_lower_case, verbose_logging=self.args.verbose_logging)
        self.QA_history.append((turn, question, (predicted_answer, None, None)))
        return predicted_answer, next_unique_id