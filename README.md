# Ditch the Gold Standard: Re-evaluating Conversational Question Answering
This is the repository for our paper [Ditch the Gold Standard: Re-evaluating Conversational Question Answering](https://arxiv.org/pdf/2112.08812.pdf). 

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

## Auto-Rewrite

Our proposed evaluation protocol, Auto-Rewrite, better demonstrates models' performance in human-model conversations. Please refer to our paper for more details. Following is a figure describing how Auto-Rewrite works.

![Auto-rewrite](figs/autorewrite.png)

To use our Auto-Rewite evaluation on your own model, follow the steps:

* Step 1: Write a model interface following the template `interface.py`.

* Step 2: Add model to the evaluation script `run_quac_eval.py`.

* Step 3: Run evaluation script. See `run.sh` for reference.

## Citation

```
@article{li2021ditch,
   title={Ditch the Gold Standard: Re-evaluating Conversational Question Answering},
   author={Li, Huihan and Gao, Tianyu and Goenka, Manan and Chen, Danqi},
   journal={arXiv preprint arXiv:2112.08812},
   year={2021}
}
```
