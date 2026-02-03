from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    auc,
)


def _safe_roc_auc(y_true, y_proba) -> Optional[float]:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None


def _safe_pr_auc(y_true, y_proba) -> Optional[float]:
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return float(auc(recall, precision))
    except Exception:
        return None


def compute_metrics(y_true, y_pred, y_proba_dropout, num_classes: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute classification metrics (binary or 3-class).
    """
    if num_classes is None:
        def _to_list(values):
            return values.tolist() if hasattr(values, "tolist") else list(values)

        unique = set(_to_list(y_true))
        unique.update(_to_list(y_pred))
        num_classes = 3 if len(unique) > 2 else 2

    labels = [0, 1, 2] if num_classes == 3 else [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = fp = fn = tp = None
    if num_classes == 2 and cm.size == 4:
        tn, fp, fn, tp = cm.ravel()

    y_true_arr = np.asarray(y_true)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": _safe_roc_auc((y_true_arr == 1).astype(int), y_proba_dropout),
        "pr_auc": _safe_pr_auc((y_true_arr == 1).astype(int), y_proba_dropout),
        "confusion_matrix": cm.tolist(),
    }

    if num_classes == 3:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

    metrics.update(
        {
            "TN": None if tn is None else int(tn),
            "FP": None if fp is None else int(fp),
            "FN": None if fn is None else int(fn),
            "TP": None if tp is None else int(tp),
        }
    )
    return metrics
