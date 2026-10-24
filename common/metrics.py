"""Evaluation metrics and validation-based threshold selection.

All anomaly scores follow one convention: higher score = more anomalous.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def ranking_metrics(y_true, scores):
    """Threshold-free metrics: ROC-AUC and PR-AUC (average precision)."""
    scores = np.nan_to_num(np.asarray(scores, float), posinf=1e12, neginf=-1e12)
    return {"roc_auc": roc_auc_score(y_true, scores),
            "pr_auc": average_precision_score(y_true, scores)}


def decision_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "flagged": float(np.mean(y_pred)),
    }


def evaluate(y_true, scores, y_pred):
    return {**ranking_metrics(y_true, scores), **decision_metrics(y_true, y_pred)}


def best_f1_threshold(y_val, scores_val):
    """Score threshold maximising F1 on validation data (predict s >= t).

    The threshold is placed half-way between the chosen validation score and
    the next lower one, so it transfers more robustly to unseen data.
    """
    scores_val = np.nan_to_num(np.asarray(scores_val, float), posinf=1e12)
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))
    t = thresholds[best]
    lower = scores_val[scores_val < t]
    return float((t + lower.max()) / 2) if lower.size else float(t)


def contamination_threshold(train_scores, contamination):
    """Threshold flagging the given fraction of training points."""
    return float(np.quantile(train_scores, 1 - contamination))
