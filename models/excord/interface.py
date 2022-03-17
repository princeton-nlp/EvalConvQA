from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    RobertaTokenizer
)
from models.excord.quac import (
    QuacProcessor,
    quac_convert_example_to_features_pt,
    QuacResult,
)
from models.excord.quac_metrics import (
    compute_one_prediction_logits,
)
from models.excord.modeling_auto import AutoModelForQuestionAnswering


import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class Excord():
    def __init__(self, args):
        self.args = args
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.args.model_name_or_path, config=config)
        self.QA_history = []
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        self.model = self.model.to(self.device)
    
    def tokenizer(self):
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.args.model_name_or_path, do_lower_case=self.args.do_lower_case)
        return self.tokenizer
    
    def load_partial_examples(self, predict_file):
        processor = QuacProcessor(
            tokenizer=self.tokenizer, orig_history=False)
        partial_examples = processor.get_partial_dev_examples(data_dir=None, filename=predict_file)
        return partial_examples

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer):
        processor = QuacProcessor(tokenizer=tokenizer, orig_history=False)
        question = partial_example.question_text

        example = processor.process_one_dev_example(self.QA_history, example_idx, partial_example)
        dataset, features = quac_convert_example_to_features_pt(
            example, tokenizer, self.args.max_seq_length, self.args.doc_stride, self.args.max_query_length)
        new_features = []
        next_unique_id = unique_id
        for example_feature in features:
            example_feature.unique_id = next_unique_id
            new_features.append(example_feature)
            next_unique_id += 1
        features = new_features
        del new_features
        # Run prediction for full data

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=1)
        curr_results = []
        # Run prediction for current example
        for batch in eval_dataloader:

            batch = tuple(t.to(self.device) for t in batch)
            # Assume the logits are a list of one item
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                feature_indices = batch[3]

                temp_outputs = self.model(**inputs)
                batch_start_logits = temp_outputs.start_logits
                batch_end_logts = temp_outputs.end_logits
                outputs = [batch_start_logits, batch_end_logts]
            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                example_unique_id = int(eval_feature.unique_id)

                output = [Excord.to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = QuacResult(example_unique_id, start_logits,
                                    end_logits,0)

                curr_results.append(result)
        prediction, nbest_prediction = compute_one_prediction_logits(
            example,
            features,
            curr_results,
            self.args.n_best_size,
            self.args.max_answer_length,
            self.args.do_lower_case,
            self.args.verbose_logging,
            self.args.null_score_diff_threshold,
            tokenizer
        )
        self.QA_history.append((example_idx, question, (prediction, None, None)))
        return prediction, next_unique_id
