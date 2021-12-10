from __future__ import absolute_import, division, print_function
from io import open
import json

from coreference_resolution import find_coreference_f1s

def filter_with_coreference(partial_example, background, gold_answers, QA_history, history_len=2, match_metric='em', add_background=False, skip_entity=True):
    """Append the previous predicted answers to the context during evaluation"""
    qa_gold = ""
    qa_pred = ""
    i = 0
    total_length = len(QA_history)
    while i < history_len:
        index = total_length-1-i
        if index >= 0:
            qa_gold = gold_answers[QA_history[index][0]] + ' ' + qa_gold #using turn_id to access gold answers
            qa_gold = QA_history[index][1] + ' ' + qa_gold

            qa_pred = QA_history[index][2][0] + ' ' + qa_pred # answer text
            qa_pred = QA_history[index][1] + ' ' + qa_pred # question
        i+=1
    qa_gold += "<Q> "+ partial_example.question_text
    qa_pred += "<Q> "+ partial_example.question_text

    if add_background:
        qa_gold = background + " " + qa_gold
        qa_pred = background + " " + qa_pred

    f1s, resolved_gold, resolved_pred = find_coreference_f1s(qa_gold, qa_pred, skip_entity)
    
    if match_metric == 'em':
        modified_gold = resolved_gold.split("< Q >")[-1].strip()
        modified_pred = resolved_pred.split("< Q >")[-1].strip()
        skip = (modified_gold != modified_pred)
    elif match_metric == 'f1':
        skip = False if all([f1 > 0 for f1 in f1s]) else True

    return skip

def rewrite_with_coreference(partial_example, background, gold_answers, QA_history, history_len=2, match_metric='f1', add_background=True, skip_entity=True):
    qa_gold = ""
    qa_pred = ""
    i = 0
    total_length = len(QA_history)
    while i < history_len:
        index = total_length-1-i
        if index >= 0:
            qa_gold = gold_answers[QA_history[index][0]] + ' ' + qa_gold #using turn_id to access gold answers
            qa_gold = QA_history[index][1] + ' ' + qa_gold

            qa_pred = QA_history[index][2][0] + ' ' + qa_pred # answer text
            qa_pred = QA_history[index][1] + ' ' + qa_pred # question
        i+=1
    qa_gold += "<Q> "+ partial_example.question_text
    qa_pred += "<Q> "+ partial_example.question_text

    if add_background:
        qa_gold = background + " " + qa_gold
        qa_pred = background + " " + qa_pred

    f1s, resolved_gold, resolved_pred = find_coreference_f1s(qa_gold, qa_pred, skip_entity)
    
    modified_gold = resolved_gold.split("< Q >")[-1].strip()
    modified_pred = resolved_pred.split("< Q >")[-1].strip()

    if match_metric == 'em':
        skip = (modified_gold != modified_pred)
    elif match_metric == 'f1':
        skip = False if all([f1 > 0 for f1 in f1s]) else True

    return skip, modified_gold

def write_automatic_eval_result(json_file, evaluation_result):
    """evaluation_results = [{"CID": ..., 
                              "Predictions": [
                                  (qa_id, span),
                                  ...
                              ]}, ...]"""

    with open(json_file, 'w') as fout:
        for passage_index, predictions in evaluation_result.items():
            output_dict = {'best_span_str': [], 'qid': [], 'yesno':[], 'followup': []}
            for qa_id, span in predictions["Predictions"]:
                output_dict['best_span_str'].append(span)
                output_dict['qid'].append(qa_id)
                output_dict['yesno'].append('y')
                output_dict['followup'].append('y')
            fout.write(json.dumps(output_dict) + '\n')

def write_invalid_category(json_file, skip_dictionary):
    with open(json_file, 'w') as fout:
        fout.write(json.dumps(skip_dictionary, indent=2))

def load_context_indep_questions(canard_path):
    data = {}
    with open(canard_path, encoding="utf-8") as c:
        canard_data = json.load(c)
        for entry in canard_data:
            cid = entry["QuAC_dialog_id"]
            turn = entry["Question_no"]-1
            rewrite = entry["Rewrite"]
            qid = cid + "_q#" + str(turn)
            data[qid] = rewrite
    return data


            