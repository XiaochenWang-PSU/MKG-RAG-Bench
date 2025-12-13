import re
import string
import math
from collections import Counter
from typing import List, Union, Sequence
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def normalize_answer(s):
    s = s.lower()
    s = s.translate(_PUNC_TABLE)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)

def tokenize(s):
    return s.split() if s else []

def compute_metrics_close(y_true, y_pred):
    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_true]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

def bleu(y_true, y_pred):
    return sentence_bleu([y_true], y_pred)

def exact_match(y_true, y_pred):
    return float(y_true == y_pred)

def macro_average_f1(y_true, y_pred):
    pred_toks = tokenize(y_pred)
    gold_toks = tokenize(y_true)

    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def compute_metrics_open(y_true, y_pred):
    from nltk.translate.bleu_score import sentence_bleu
    bleu_score = 0
    exact_match_score = 0
    macro_average_f1_score = 0

    for i in range(len(y_true)):
        prediction = normalize_answer(y_pred[i])
        ground_truth = normalize_answer(y_true[i])

        bleu_score += bleu(ground_truth, prediction)
        exact_match_score += exact_match(ground_truth, prediction)
        macro_average_f1_score += macro_average_f1(ground_truth, prediction)

    return {
        "exact_match": exact_match_score/len(y_pred),
        "macro_average_f1": macro_average_f1_score/len(y_pred),
        "bleu": bleu_score/len(y_pred),
    }