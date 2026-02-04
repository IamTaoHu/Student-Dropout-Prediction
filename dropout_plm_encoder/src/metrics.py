from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
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

LABEL2ID = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
ID2LABEL = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}


def map_labels(series):
    """
    Map target labels to integers using LABEL2ID.
    Supports numeric labels {0,1,2} and string labels case-insensitively.
    """
    if hasattr(series, "dtype") and hasattr(series, "astype"):
        series_values = series
    else:
        series_values = np.asarray(series)

    if pd.api.types.is_numeric_dtype(series_values):
        unique_vals = set(pd.Series(series_values).dropna().unique().tolist())
        if unique_vals.issubset({0, 1, 2}):
            return series_values.astype(int)
        raise ValueError(
            "Numeric target column must contain only {0, 1, 2} for multi-class "
            f"dropout prediction. Found: {sorted(unique_vals)}"
        )

    mapping = {k.lower(): v for k, v in LABEL2ID.items()}
    mapped = (
        np.asarray(series_values, dtype=object)
        .astype(str)
    )
    normalized = np.vectorize(lambda v: str(v).strip().lower())(mapped)
    result = [mapping.get(v) for v in normalized]
    if any(v is None for v in result):
        bad = sorted({str(v) for v in np.asarray(series_values, dtype=object)})
        raise ValueError(
            "Could not map target labels. Expected one of: Dropout, Enrolled, Graduate "
            f"(case-insensitive). Found: {bad}"
        )
    return np.asarray(result, dtype=int)


def print_class_distribution(labels) -> None:
    values, counts = np.unique(labels, return_counts=True)
    dist = {int(v): int(c) for v, c in zip(values, counts)}
    print("[INFO] Class distribution:", dist)


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


def _safe_roc_auc_ovr_macro(y_true, y_proba) -> Optional[float]:
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return None


def _safe_pr_auc_ovr_macro(y_true, y_proba, labels) -> Optional[float]:
    try:
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)
        pr_aucs = []
        for idx, label in enumerate(labels):
            y_true_bin = (y_true_arr == label).astype(int)
            pr_auc = _safe_pr_auc(y_true_bin, y_proba_arr[:, idx])
            if pr_auc is not None:
                pr_aucs.append(pr_auc)
        if not pr_aucs:
            return None
        return float(np.mean(pr_aucs))
    except Exception:
        return None


def compute_metrics_multiclass(
    y_true,
    y_pred,
    y_proba,
    labels=(0, 1, 2),
) -> Dict[str, Any]:
    """
    Compute multiclass (3-class) metrics.
    Inputs:
      - y_true: shape (n,)
      - y_pred: shape (n,)
      - y_proba: shape (n, 3)
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        str(label): {
            "precision": report.get(str(label), {}).get("precision", 0.0),
            "recall": report.get(str(label), {}).get("recall", 0.0),
            "f1": report.get(str(label), {}).get("f1-score", 0.0),
            "support": report.get(str(label), {}).get("support", 0.0),
        }
        for label in labels
    }

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "TN": None,
        "FP": None,
        "FN": None,
        "TP": None,
    }

    roc_auc = _safe_roc_auc_ovr_macro(y_true, y_proba)
    if roc_auc is not None:
        metrics["roc_auc_ovr_macro"] = roc_auc
    pr_auc = _safe_pr_auc_ovr_macro(y_true, y_proba, labels=list(labels))
    if pr_auc is not None:
        metrics["pr_auc_ovr_macro"] = pr_auc

    return metrics


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


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def print_summary_table(model_name: str, metrics: dict) -> None:
    headers = [
        ("Model", 18),
        ("accuracy", 10),
        ("f1_macro", 10),
        ("recall_macro", 12),
        ("roc_auc_ovr_macro", 18),
        ("pr_auc_ovr_macro", 18),
    ]
    row = [
        _fmt(model_name),
        _fmt(metrics.get("accuracy")),
        _fmt(metrics.get("f1_macro")),
        _fmt(metrics.get("recall_macro")),
        _fmt(metrics.get("roc_auc_ovr_macro")),
        _fmt(metrics.get("pr_auc_ovr_macro")),
    ]
    header_line = " | ".join(h.ljust(w) for h, w in headers)
    sep_line = "-+-".join("-" * w for _, w in headers)
    row_line = " | ".join(val.ljust(w) for (val, (_, w)) in zip(row, headers))
    print(header_line)
    print(sep_line)
    print(row_line)


def print_per_class_table(metrics: dict, id2label: dict) -> None:
    per_class = metrics.get("per_class") or {}
    headers = [
        ("Class", 5),
        ("label", 12),
        ("precision", 10),
        ("recall", 10),
        ("f1", 10),
        ("support", 8),
    ]
    header_line = " | ".join(h.ljust(w) for h, w in headers)
    sep_line = "-+-".join("-" * w for _, w in headers)
    print(header_line)
    print(sep_line)
    for class_id in [0, 1, 2]:
        key = str(class_id)
        vals = per_class.get(key, {})
        row = [
            _fmt(class_id),
            _fmt(id2label.get(class_id, "")),
            _fmt(vals.get("precision")),
            _fmt(vals.get("recall")),
            _fmt(vals.get("f1")),
            _fmt(vals.get("support"), nd=0),
        ]
        row_line = " | ".join(val.ljust(w) for (val, (_, w)) in zip(row, headers))
        print(row_line)


def print_confusion_matrix(metrics: dict, id2label: dict) -> None:
    cm = metrics.get("confusion_matrix")
    if not cm:
        print("[INFO] No confusion_matrix available.")
        return
    labels = [0, 1, 2]
    cell_w = 6
    header = " " * (cell_w + 6)
    for label in labels:
        name = id2label.get(label, "")
        header += f"{label}:{name}".ljust(cell_w + 2)
    print(header.rstrip())
    for i, row in enumerate(cm):
        row_label = f"{i}:{id2label.get(i, '')}".ljust(cell_w + 4)
        row_cells = "".join(str(v).rjust(cell_w) + "  " for v in row)
        print(row_label + row_cells.rstrip())
