import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)

def compute_metrics(targets, preds, thr=0.5):
    metrics = {}

    try:
        metrics["auc"] = float(roc_auc_score(targets, preds))
    except:
        metrics["auc"] = float("nan")

    pred_labels = (preds >= thr).astype(int)

    metrics["accuracy"] = float(accuracy_score(targets, pred_labels))
    metrics["precision"] = float(precision_score(targets, pred_labels, zero_division=0))
    metrics["recall"] = float(recall_score(targets, pred_labels, zero_division=0))
    metrics["f1"] = float(f1_score(targets, pred_labels, zero_division=0))

    return metrics
