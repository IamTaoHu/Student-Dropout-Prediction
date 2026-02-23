from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    y_pred = y_proba.argmax(axis=1)

    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro"))

    # OVR AUCs require probability matrix with shape [n, n_classes]
    try:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except Exception:
        out["roc_auc_ovr_macro"] = float("nan")

    try:
        out["pr_auc_ovr_macro"] = float(
            average_precision_score(np.eye(y_proba.shape[1])[y_true], y_proba, average="macro")
        )
    except Exception:
        out["pr_auc_ovr_macro"] = float("nan")

    return out


def format_summary_row(model_name: str, m: Dict[str, float]) -> str:
    return (
        f"{model_name:<18} | "
        f"{m['accuracy']:<10.4f} | {m['f1_macro']:<10.4f} | {m['recall_macro']:<12.4f} | "
        f"{m['roc_auc_ovr_macro']:<18.4f} | {m['pr_auc_ovr_macro']:<17.4f}"
    )


def format_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels=("Dropout", "Enrolled", "Graduate"),
) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    header = " " * 14 + " ".join([f"{i}:{lab:<8}" for i, lab in enumerate(labels)])
    lines = [header]
    for i, lab in enumerate(labels):
        row = " ".join([f"{cm[i, j]:<10d}" for j in range(3)])
        lines.append(f"{i}:{lab:<10} {row}")
    return "\n".join(lines)
