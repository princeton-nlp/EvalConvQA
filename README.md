# Ditch the Gold Standard: Re-evaluating Conversational Question Answering
This is the repository for our paper [Ditch the Gold Standard: Re-evaluating Conversational Question Answering](https://arxiv.org/pdf/2112.08812.pdf). 

## Overview

In this work, we conduct the first large-scale human evaluation of state-of-the-art conversational QA systems. In our evaluation, human annotators chat with conversational QA models about passages from the [QuAC](https://quac.ai) development set, and after that the annotators judge the correctness of model answers. We release the human annotated dataset in the following section. 

We also identify a critical issue with the current automatic evaluation, which pre-collectes human-human conversations and uses ground-truth answers as conversational history (differences between different evaluations are shown in the following figure). By comparison, we find that the automatic evaluation does not always agree with the human evaluation. We propose a new evaluation protocol that is based on using predicted history and question rewriting. Our experiments show that the new protocol better reflects real-world performance compared to the original automatic evaluation. We also provide the new evaluation protocol code in the following.

![This is an image](figs/example.png)

## Human Annotation Dataset
You can download the human annotation dataset from `data/human_annotation_data.json`.

## Evaluation

Step 1: Write a model interface following the template `interface.py`.

Step 2: Add model to the evaluation script `run_quac_eval.py`.

Step 3: Run evaluation script. See `run.sh` for reference.
