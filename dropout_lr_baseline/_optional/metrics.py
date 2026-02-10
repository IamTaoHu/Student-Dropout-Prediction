from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _safe_metric(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba,
    labels: list[int],
    task: str,
) -> dict:
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, digits=4
        ),
    }

    if task == "binary":
        metrics["f1"] = _safe_metric(f1_score, y_true, y_pred, pos_label=labels[-1])
        metrics["recall"] = _safe_metric(
            recall_score, y_true, y_pred, pos_label=labels[-1]
        )
        if y_proba is not None:
            metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, y_proba)
            metrics["pr_auc"] = _safe_metric(average_precision_score, y_true, y_proba)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
        return metrics

    metrics["f1_macro"] = _safe_metric(
        f1_score, y_true, y_pred, average="macro"
    )
    metrics["f1_weighted"] = _safe_metric(
        f1_score, y_true, y_pred, average="weighted"
    )
    metrics["recall_macro"] = _safe_metric(
        recall_score, y_true, y_pred, average="macro"
    )

    if y_proba is not None:
        metrics["roc_auc_ovr_macro"] = _safe_metric(
            roc_auc_score, y_true, y_proba, multi_class="ovr", average="macro"
        )
        y_true_bin = label_binarize(y_true, classes=labels)
        metrics["pr_auc_macro"] = _safe_metric(
            average_precision_score, y_true_bin, y_proba, average="macro"
        )
    else:
        metrics["roc_auc_ovr_macro"] = None
        metrics["pr_auc_macro"] = None

    return metrics


def print_metrics_table(metrics: dict, task: str) -> None:
    if task == "binary":
        headers = ["accuracy", "f1", "recall", "roc_auc", "pr_auc"]
    else:
        headers = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "recall_macro",
            "roc_auc_ovr_macro",
            "pr_auc_macro",
        ]

    header_line = "  ".join(f"{h:>16s}" for h in headers)
    value_line = "  ".join(f"{_format_value(metrics.get(h)):>16s}" for h in headers)

    print("Metrics Summary")
    print(header_line)
    print(value_line)
    print("Confusion Matrix")
    for row in metrics.get("confusion_matrix", []):
        print(" ".join(f"{int(x):6d}" for x in row))
