# Ditch the Gold Standard: Re-evaluating Conversational Question Answering
This is the repository for our ACL'2022 paper [Ditch the Gold Standard: Re-evaluating Conversational Question Answering](https://arxiv.org/pdf/2112.08812.pdf). 

## Overview

In this work, we conduct the first large-scale human evaluation of state-of-the-art conversational QA systems. In our evaluation, human annotators chat with conversational QA models about passages from the [QuAC](https://quac.ai) development set, and after that the annotators judge the correctness of model answers. We release the human annotated dataset in the following section. 

We also identify a critical issue with the current automatic evaluation, which pre-collectes human-human conversations and uses ground-truth answers as conversational history (differences between different evaluations are shown in the following figure). By comparison, we find that the automatic evaluation does not always agree with the human evaluation. We propose a new evaluation protocol that is based on predicted history and question rewriting. Our experiments show that the new protocol better reflects real-world performance compared to the original automatic evaluation. We also provide the new evaluation protocol code in the following.

![Different evaluation protocols](figs/example.png)

## Human Evaluation Dataset
You can download the human annotation dataset from `data/human_annotation_data.json`. The json file contains one data field `data`, which is a list of conversations. Each conversation contains the following fields: 

* `model_name`: The model evaluated. One of `bert4quac`, `graphflow`, `ham`, `excord`.
* `context`: The passage used in this conversation.
* `dialog_id`: The ID from the original QuAC dataset.
* `qas`: The conversation, which contains a list of QA pairs. Each QA pair has the following fields:
  * `turn_id`: The number of turn. 
  * `question`: The question from the human annotator.
  * `answer`: The answer from the model.
  * `valid`: Whether the question is valid (annotated by our human annotator).
  * `answerable`: Whether the question is answerable (annotated by our human annotator).
  * `correct`: Whether the model's answer is correct (annotated by our human annotator).

## Automatic model evaluation interface

We provide a convenient interface to test model performance on a few evaluation protocols compared in our paper, including `Auto-Pred`, `Auto-Replace` and our proposed evaluation protocol, `Auto-Rewrite`, which better demonstrates models' performance in human-model conversations. Please refer to our paper for more details. Following is a figure describing how Auto-Rewrite works.

![Auto-rewrite](figs/autorewrite.png)

## Setup

### Install dependencies

Please install all dependency packages using the following command:
```
pip install -r requirements.txt
```

### Download the datasets

Our experiments use [QuAC dataset](https://quac.ai) for passages and conversations, and the test set of [CANARD dataset](https://sites.google.com/view/qanta/projects/canard) for context-independent questions in `Auto-Replace`.

## Evaluating existing models

We provide our implementations for the four models that we used in our paper: BERT, [GraphFlow](https://www.ijcai.org/Proceedings/2020/171), [HAM](https://dl.acm.org/doi/abs/10.1145/3357384.3357905), [ExCorD](https://aclanthology.org/2021.acl-long.478/). We modified exisiting implementation online to use model predictions as conversation history. Below are the instructions to run evaluation script on each of these models.

### BERT
We implemented and trained our own BERT model.
```
# Run Training
python run_quac_train.py \
  --type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --output_dir ${directory_to_save_model} \
  --overwrite_output_dir \
  --train_file ${path_to_quac_train_file} \
  --train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --history_len 2 \
  --warmup_proportion 0.1 \
  --max_grad_norm -1 \
  --weight_decay 0.01 \
  --rationale_beta 0 \ # important for BERT

# Run Evaluation (Auto-Rewrite as example)
python run_quac_eval.py \
  --type bert \
  --output_dir ${directory-to-model-checkpoint} \
  --write_dir ${directory-to-write-evaluation-result} \
  --predict_file val_v0.2.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 64 \
  --match_metric f1 \
  --add_background \
  --skip_entity \
  --rewrite \
  --start_i ${index_of_first_passage_to_eval} \
  --end_i ${index_of_last_passage_to_eval_exclusive} \
```


### GraphFlow
We did not find an uploaded model checkpoint so we trained our own using [their training script](https://github.com/hugochan/GraphFlow).
```
# Run Evaluation (Auto-Rewrite as example)

# Download Stanford CoreNLP package
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
rm -f stanford-corenlp-latest.zip

# Start StanfordCoreNLP server
java -mx4g -cp "${directory_to_standford_corenlp_package}" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 &

python run_quac_eval.py \
    --type graphflow \
    --predict_file ${path-to-annotated-dev-json-file} \
    --output_dir ${directory-to-model-checkpoint} \
    --saved_vocab_file ${directory-to-saved-model-vocab} \
    --pretrained ${directory-to-model-checkpoint} \
    --write_dir /n/fs/scratch/huihanl/unified/graphflow/write \
    --match_metric f1 \
    --add_background \
    --skip_entity \
    --rewrite \
    --fix_vocab_embed \
    --f_qem \
    --f_pos \
    --f_ner \
    --use_ques_marker \
    --use_gnn \
    --temporal_gnn \
    --use_bert \
    --use_bert_weight \
    --shuffle \
    --out_predictions \
    --predict_raw_text \
    --out_pred_in_folder \
    --optimizer adamax \
    --start_i ${index_of_first_passage_to_eval} \
    --end_i ${index_of_last_passage_to_eval_exclusive} \
```


### HAM
The orgininal model checkpoint can be downloaded from [CodaLab](https://worksheets.codalab.org/rest/bundles/0x5c08cb0fb90c4afd8a2811bb63023cce/contents/blob/)

```
# Run Evaluation (Auto-Rewrite as example)
python run_quac_eval.py \
  --type ham \
  --output_dir ${directory-to-model-checkpoint} \
  --write_dir ${directory-to-write-evaluation-result} \
  --predict_file val_v0.2.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 64 \
  --do_lower_case \
  --history_len 6 \
  --match_metric f1 \
  --add_background \
  --skip_entity \
  --replace \
  --init_checkpoint ${directory-to-model-checkpoint}/model_52000.ckpt \
  --bert_config_file ${directory-to-pretrained-bert-large-uncased}/bert_config.json \
  --vocab_file ${directory-to-model-checkpoint}/vocab.txt \
  --MTL_mu 0.8 \
  --MTL_lambda 0.1 \
  --mtl_input reduce_mean \
  --max_answer_length 40 \
  --max_considered_history_turns 4 \
  --bert_hidden 1024 \
  --fine_grained_attention \
  --better_hae \
  --MTL \
  --use_history_answer_marker \
  --start_i ${index_of_first_passage_to_eval} \
  --end_i ${index_of_last_passage_to_eval_exclusive} \
```


### ExCorD
The original model checkpoint can be downloaded from [their repo](https://drive.google.com/file/d/1Xf0-XUvGi7jgiAAdA5BQLk7p5ikc_wOl/view?usp=sharing)

```
# Run Evaluation (Auto-Rewrite as example)
python run_quac_eval.py \
  --type excord \
  --output_dir ${directory-to-model-checkpoint} \
  --write_dir ${directory-to-write-evaluation-result} \
  --predict_file val_v0.2.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 64 \
  --match_metric f1 \
  --add_background \
  --skip_entity \
  --rewrite \
  --start_i ${index_of_first_passage_to_eval} \
  --end_i ${index_of_last_passage_to_eval_exclusive} \
```

## Citation

```
@inproceedings{li2022ditch,
   title={Ditch the Gold Standard: Re-evaluating Conversational Question Answering},
   author={Li, Huihan and Gao, Tianyu and Goenka, Manan and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2022}
}
```
