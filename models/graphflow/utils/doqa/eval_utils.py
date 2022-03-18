import json, string, re
from collections import Counter, defaultdict
from argparse import ArgumentParser


def is_overlapping(x1, x2, y1, y2):
  return max(x1, y1) <= min(x2, y2)

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def exact_match_score(prediction, ground_truth):
  return (normalize_answer(prediction) == normalize_answer(ground_truth))

def leave_one_out_max(prediction, ground_truths, article):
  if len(ground_truths) == 1:
    return metric_max_over_ground_truths(prediction, ground_truths, article)[1]
  else:
    t_f1 = []
    # leave out one ref every time
    for i in range(len(ground_truths)):
      idxes = list(range(len(ground_truths)))
      idxes.pop(i)
      refs = [ground_truths[z] for z in idxes]
      t_f1.append(metric_max_over_ground_truths(prediction, refs, article)[1])
  return 1.0 * sum(t_f1) / len(t_f1)


def metric_max_over_ground_truths(prediction, ground_truths, article):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = compute_span_overlap(prediction, ground_truth, article)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths, key=lambda x: x[1])


def handle_cannot(refs):
  num_cannot = 0
  num_spans = 0
  for ref in refs:
    if ref == 'CANNOTANSWER':
      num_cannot += 1
    else:
      num_spans += 1
  if num_cannot >= num_spans:
    refs = ['CANNOTANSWER']
  else:
    refs = [x for x in refs if x != 'CANNOTANSWER']
  return refs


def leave_one_out(refs):
  if len(refs) == 1:
    return 1.
  splits = []
  for r in refs:
    splits.append(r.split())
  t_f1 = 0.0
  for i in range(len(refs)):
    m_f1 = 0
    for j in range(len(refs)):
      if i == j:
        continue
      f1_ij = f1_score(refs[i], refs[j])
      if f1_ij > m_f1:
        m_f1 = f1_ij
    t_f1 += m_f1
  return t_f1 / len(refs)


def compute_span_overlap(pred_span, gt_span, text):
  if gt_span == 'CANNOTANSWER':
    if pred_span == 'CANNOTANSWER':
      return 'Exact match', 1.0
    return 'No overlap', 0.
  fscore = f1_score(pred_span, gt_span)
  pred_start = text.find(pred_span)
  gt_start = text.find(gt_span)

  if pred_start == -1 or gt_start == -1:
    return 'Span indexing error', fscore

  pred_end = pred_start + len(pred_span)
  gt_end = gt_start + len(gt_span)

  fscore = f1_score(pred_span, gt_span)
  overlap = is_overlapping(pred_start, pred_end, gt_start, gt_end)

  if exact_match_score(pred_span, gt_span):
    return 'Exact match', fscore
  if overlap:
    return 'Partial overlap', fscore
  else:
    return 'No overlap', fscore


def eval_fn(gold_results, pred_results, raw_context, min_f1=0.4):
  total_qs = 0.
  f1_stats = defaultdict(list)
  human_f1 = []
  HEQ = 0.
  DHEQ = 0.
  total_dials = 0.
  for dial_idx, ex_results in enumerate(gold_results):
    good_dial = 1.
    for turn_idx, turn_results in enumerate(ex_results):
      gold_spans = handle_cannot(turn_results)
      hf1 = leave_one_out(gold_spans)

      pred_span = pred_results[dial_idx][turn_idx]

      max_overlap, _ = metric_max_over_ground_truths( \
        pred_span, gold_spans, raw_context[dial_idx])
      max_f1 = leave_one_out_max( \
        pred_span, gold_spans, raw_context[dial_idx])

      # dont eval on low agreement instances
      if hf1 < min_f1:
        continue

      human_f1.append(hf1)
      if max_f1 >= hf1:
        HEQ += 1.
      else:
        good_dial = 0.
      f1_stats[max_overlap].append(max_f1)
      total_qs += 1.
    DHEQ += good_dial
    total_dials += 1

  DHEQ_score = DHEQ / total_dials
  if total_qs == 0:
    HEQ_score = 0
  else:
    HEQ_score = HEQ / total_qs
  all_f1s = sum(f1_stats.values(), [])
  if len(all_f1s) == 0:
    overall_f1 = 0
  else:
    overall_f1 = sum(all_f1s) / len(all_f1s)
  metric_json = {"f1": overall_f1, "heq": HEQ_score, "dheq": DHEQ_score}
  return metric_json, total_qs, total_dials
