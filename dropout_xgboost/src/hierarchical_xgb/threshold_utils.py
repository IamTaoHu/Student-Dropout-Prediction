from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from hierarchical_xgb.model_utils import make_binary_for_label


def _sanitize_proba_for_logloss(y_proba, n_classes: int) -> np.ndarray:
    p = np.asarray(y_proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != int(n_classes):
        raise ValueError(f"Expected proba shape (n_samples, {n_classes}), got {p.shape}.")
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.clip(np.sum(p, axis=1, keepdims=True), 1e-12, None)
    return p


def eval_4class_full(y_true, y_pred, y_proba=None, labels=None) -> dict:
    labels_idx = list(range(len(labels)))
    rep = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=labels,
        digits=4, zero_division=0, output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels_idx).tolist()
    ll = None
    if y_proba is not None:
        try:
            ll = float(log_loss(y_true, _sanitize_proba_for_logloss(y_proba, len(labels)), labels=labels_idx))
        except Exception:
            ll = None
    per_class_recall = {}
    per_class_precision = {}
    for name in labels:
        cls = rep.get(name, {})
        per_class_recall[name] = float(cls.get("recall", 0.0))
        per_class_precision[name] = float(cls.get("precision", 0.0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(rep.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(rep.get("weighted avg", {}).get("f1-score", 0.0)),
        "logloss": float(ll) if ll is not None else None,
        "confusion_matrix": matrix,
        "classification_report": rep,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
    }


def evaluate_4class(y_true, y_pred, y_proba=None, labels=None):
    payload = eval_4class_full(y_true, y_pred, y_proba=y_proba, labels=labels)
    return (
        float(payload["macro_f1"]),
        float(payload["accuracy"]),
        float(payload["balanced_accuracy"]),
        payload["logloss"],
        payload["classification_report"],
    )


def recall_for_class(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> float:
    mask = y_true == class_id
    denom = int(np.sum(mask))
    if denom == 0:
        return 0.0
    return int(np.sum((y_pred == class_id) & mask)) / denom


def predict_3stage_from_proba(p1, p2, p3_pos, labels, t1, t2, stage1_target: str, stage2_target: str, stage3_pos: str, stage3_neg: str, t3: float = 0.5):
    stage1_id = labels.index(stage1_target)
    stage2_id = labels.index(stage2_target)
    stage3_pos_id = labels.index(stage3_pos)
    stage3_neg_id = labels.index(stage3_neg)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3_pos = np.asarray(p3_pos, dtype=float)
    pred = np.empty(p1.shape[0], dtype=int)
    is_dist = p1 >= t1
    pred[is_dist] = stage1_id
    rest = ~is_dist
    is_wd = (p2 >= t2) & rest
    pred[is_wd] = stage2_id
    remaining = rest & (~is_wd)
    pred[remaining] = np.where(p3_pos[remaining] >= float(t3), stage3_pos_id, stage3_neg_id)
    return pred


def proba_3stage_hard(p1, p2, p3_pos, labels, t1, t2, stage1_target: str, stage2_target: str, stage3_pos: str, stage3_neg: str, t3: float = 0.5):
    stage1_id = labels.index(stage1_target)
    stage2_id = labels.index(stage2_target)
    stage3_pos_id = labels.index(stage3_pos)
    stage3_neg_id = labels.index(stage3_neg)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3_pos, dtype=float)
    out = np.zeros((p1.shape[0], 4), dtype=float)
    is_dist = p1 >= float(t1)
    out[is_dist, stage1_id] = 1.0
    rest = ~is_dist
    is_wd = (p2 >= float(t2)) & rest
    out[is_wd, stage2_id] = 1.0
    remaining = rest & (~is_wd)
    p3_pos_adj = np.where(p3[remaining] >= float(t3), np.clip(p3[remaining], 1e-9, 1.0 - 1e-9), np.clip(p3[remaining], 1e-9, 1.0 - 1e-9))
    out[remaining, stage3_pos_id] = p3_pos_adj
    out[remaining, stage3_neg_id] = 1.0 - p3_pos_adj
    return out


def _is_better_candidate(args, cand, best):
    if best["seed"] is None:
        return True
    sel = str(args.select_by)
    if sel == "macro_f1":
        return (
            cand["val_macro_f1"],
            cand["val_bal_acc"],
            cand["val_stage3_f1"],
            cand["val_accuracy"],
            -cand["val_logloss"],
        ) > (
            best["val_macro_f1"],
            best["val_bal_acc"],
            best["val_stage3_f1"],
            best["val_accuracy"],
            -best["val_logloss"],
        )
    if sel == "accuracy":
        return (
            cand["val_accuracy"],
            cand["val_macro_f1"],
            cand["val_bal_acc"],
            cand["val_stage3_f1"],
            -cand["val_logloss"],
        ) > (
            best["val_accuracy"],
            best["val_macro_f1"],
            best["val_bal_acc"],
            best["val_stage3_f1"],
            -best["val_logloss"],
        )
    return (
        -cand["val_logloss"],
        cand["val_macro_f1"],
        cand["val_bal_acc"],
        cand["val_accuracy"],
        cand["val_stage3_f1"],
    ) > (
        -best["val_logloss"],
        best["val_macro_f1"],
        best["val_bal_acc"],
        best["val_accuracy"],
        best["val_stage3_f1"],
    )


def eval_binary_full(y_true_bin, y_pred_bin, neg_name, pos_name) -> dict:
    rep = classification_report(
        y_true_bin, y_pred_bin, labels=[0, 1], target_names=[neg_name, pos_name],
        digits=4, zero_division=0, output_dict=True,
    )
    pos_rep = rep.get(pos_name, {})
    return {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist(),
        "classification_report": rep,
        "labels": [neg_name, pos_name],
        "precision_pos": float(pos_rep.get("precision", 0.0)),
        "recall_pos": float(pos_rep.get("recall", 0.0)),
        "f1_pos": float(pos_rep.get("f1-score", 0.0)),
        "support_pos": int(pos_rep.get("support", 0)),
        "num_samples": int(len(y_true_bin)),
    }


def _build_stage_report_binary(y_true_bin, y_pred_bin, neg_name, pos_name):
    return eval_binary_full(y_true_bin, y_pred_bin, neg_name, pos_name)


def _binary_prf_from_predictions(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true_bin, dtype=int)
    y_pred = np.asarray(y_pred_bin, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    f1 = 0.0 if (precision + recall) == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    b = max(float(beta), 1e-8)
    b2 = b * b
    denom = b2 * float(precision) + float(recall)
    if denom <= 0.0:
        return 0.0
    return float((1.0 + b2) * float(precision) * float(recall) / denom)


def compute_stage3_threshold_metric(y_true_bin, y_pred_bin, metric_name: str) -> float:
    y_true = np.asarray(y_true_bin, dtype=int)
    y_pred = np.asarray(y_pred_bin, dtype=int)
    name = str(metric_name).strip().lower()
    if name == "pos_f1":
        return float(f1_score(y_true, y_pred, labels=[0, 1], pos_label=1, average="binary", zero_division=0))
    if name == "macro_f1":
        return float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported Stage3 threshold metric: {metric_name}")


def _compute_stage_aux_metric(metric_name: str, y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    name = str(metric_name).strip().lower()
    if name == "macro_f1":
        return float(f1_score(y_true_bin, y_pred_bin, labels=[0, 1], average="macro", zero_division=0))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true_bin, y_pred_bin))
    if name == "recall":
        return float(recall_for_class(np.asarray(y_true_bin, dtype=int), np.asarray(y_pred_bin, dtype=int), 1))
    raise ValueError(f"Unsupported threshold metric: {metric_name}")


def compute_global_selection_score(
    val_macro_f1,
    val_acc,
    val_bal_acc,
    dist_recall,
    withdrawn_recall,
    stage3_metric,
    args,
) -> float:
    if not bool(args.stage3_plus):
        return float(float(val_macro_f1) + 0.20 * float(val_acc))
    score = (
        1.00 * float(val_macro_f1)
        + 0.15 * float(val_bal_acc)
        + 0.20 * float(dist_recall)
        + 0.10 * float(withdrawn_recall)
        + 0.05 * float(val_acc)
    )
    score += float(args.stage3_score_weight) * float(stage3_metric)
    return float(score)


def compute_stage3_objective_score(
    args: argparse.Namespace,
    objective_name: str,
    val_macro_f1: float,
    val_bal_acc: float,
    dist_recall: float,
    withdrawn_recall: float,
    stage3_precision: float,
    stage3_recall: float,
    stage3_f1: float,
    stage3_fbeta: float,
) -> dict:
    objective = str(objective_name).strip().lower()
    if objective == "legacy":
        objective_score = compute_global_selection_score(
            val_macro_f1=val_macro_f1,
            val_acc=0.0,
            val_bal_acc=val_bal_acc,
            dist_recall=dist_recall,
            withdrawn_recall=withdrawn_recall,
            stage3_metric=stage3_f1,
            args=args,
        )
    elif objective == "macro_f1":
        objective_score = float(val_macro_f1)
    elif objective == "stage3_f1":
        objective_score = float(stage3_f1)
    elif objective == "stage3_fbeta":
        objective_score = float(stage3_fbeta)
    elif objective == "hybrid":
        objective_score = (
            float(args.stage3_threshold_macro_weight) * float(val_macro_f1)
            + float(args.stage3_threshold_stage3_weight) * float(stage3_fbeta)
            + 0.15 * float(val_bal_acc)
            + 0.10 * float(dist_recall)
            + 0.05 * float(withdrawn_recall)
        )
    else:
        raise ValueError(f"Unsupported stage3 threshold objective: {objective_name}")

    return {
        "objective_score": float(objective_score),
        "stage3_metric": float(stage3_f1),
        "stage3_precision": float(stage3_precision),
        "stage3_recall": float(stage3_recall),
        "stage3_f1": float(stage3_f1),
        "stage3_fbeta": float(stage3_fbeta),
    }


def compute_collapse_aware_score(
    val_macro_f1: float,
    val_acc: float,
    val_bal_acc: float,
    per_class_recall: dict[str, float],
    stage3_f1: float,
    args: argparse.Namespace,
) -> dict:
    base_score = (
        1.00 * float(val_macro_f1)
        + 0.15 * float(val_bal_acc)
        + 0.05 * float(val_acc)
        + 0.10 * float(stage3_f1)
    )
    recall_values = [float(v) for v in per_class_recall.values()]
    min_recall = min(recall_values) if recall_values else 0.0
    distinction_recall = float(per_class_recall.get("distinction", 0.0))

    collapse_penalty = 0.0
    if min_recall < float(args.v2_min_class_recall_hard):
        collapse_penalty += float(args.v2_collapse_penalty_weight)
    elif min_recall < float(args.v2_min_class_recall_soft):
        collapse_penalty += 0.5 * float(args.v2_collapse_penalty_weight)

    if distinction_recall < float(args.v2_min_class_recall_soft):
        collapse_penalty += 0.5 * float(args.v2_collapse_penalty_weight)

    distinction_bonus = 0.05 * distinction_recall
    routing_score = base_score + distinction_bonus - collapse_penalty
    return {
        "base_score": float(base_score),
        "collapse_penalty": float(collapse_penalty),
        "distinction_bonus": float(distinction_bonus),
        "routing_score": float(routing_score),
        "min_recall": float(min_recall),
        "distinction_recall": float(distinction_recall),
    }


def _better_threshold_payload(a: dict, b: dict | None) -> bool:
    if b is None:
        return True
    if "routing_score" in a and "routing_score" in b:
        return (
            float(a["routing_score"]),
            float(a["val_macro_f1"]),
            float(a["val_bal_acc"]),
            float(a["stage3_f1"]),
            float(a["val_acc"]),
            -float(a["val_logloss"]),
        ) > (
            float(b["routing_score"]),
            float(b["val_macro_f1"]),
            float(b["val_bal_acc"]),
            float(b["stage3_f1"]),
            float(b["val_acc"]),
            -float(b["val_logloss"]),
        )
    return (
        float(a["objective_score"]),
        float(a["val_macro_f1"]),
        float(a["val_bal_acc"]),
        float(a["stage3_f1"]),
        float(a["val_acc"]),
        -float(a["val_logloss"]),
    ) > (
        float(b["objective_score"]),
        float(b["val_macro_f1"]),
        float(b["val_bal_acc"]),
        float(b["stage3_f1"]),
        float(b["val_acc"]),
        -float(b["val_logloss"]),
    )


def search_best_threshold_triplet(
    p_val_stage1,
    p_val_stage2,
    p_val_stage3,
    y_val_arr,
    mask3_val,
    labels,
    s1_label,
    s2_label,
    s3_pos,
    s3_neg,
    thresholds_stage1,
    thresholds_stage2,
    thresholds_stage3,
    args,
) -> dict:
    s3_pos_id = labels.index(s3_pos)
    best_any = None
    best_constrained = None
    total_configs = 0
    feasible_configs = 0
    lightweight_summary = []

    s1_true_bin = make_binary_for_label(y_val_arr, labels, s1_label)
    s2_true_bin_all = make_binary_for_label(y_val_arr, labels, s2_label)

    for t1 in thresholds_stage1:
        for t2 in thresholds_stage2:
            for t3 in thresholds_stage3:
                total_configs += 1
                t1f, t2f, t3f = float(t1), float(t2), float(t3)
                val_pred = predict_3stage_from_proba(
                    p_val_stage1, p_val_stage2, p_val_stage3, labels, t1f, t2f, s1_label, s2_label, s3_pos, s3_neg, t3=t3f
                )
                val_proba4_hard = proba_3stage_hard(
                    p_val_stage1, p_val_stage2, p_val_stage3, labels, t1f, t2f, s1_label, s2_label, s3_pos, s3_neg, t3=t3f
                )
                val_eval = eval_4class_full(y_val_arr, val_pred, y_proba=val_proba4_hard, labels=labels)
                val_macro_f1 = float(val_eval["macro_f1"])
                val_acc = float(val_eval["accuracy"])
                val_bal_acc = float(val_eval["balanced_accuracy"])
                val_ll = float(val_eval["logloss"]) if val_eval["logloss"] is not None else float("inf")
                val_per_class_recall = val_eval["per_class_recall"]
                dist_rec = float(val_per_class_recall.get("distinction", 0.0))
                wd_rec = float(val_per_class_recall.get("withdrawn", 0.0))

                stage1_pred = (np.asarray(p_val_stage1, dtype=float) >= t1f).astype(int)
                stage1_metric = _compute_stage_aux_metric(str(args.stage1_threshold_metric), s1_true_bin, stage1_pred)

                stage2_mask_eval = stage1_pred == 0
                if np.any(stage2_mask_eval):
                    y_stage2_true = s2_true_bin_all[stage2_mask_eval]
                    y_stage2_pred = (np.asarray(p_val_stage2, dtype=float)[stage2_mask_eval] >= t2f).astype(int)
                    stage2_metric = _compute_stage_aux_metric(str(args.stage2_threshold_metric), y_stage2_true, y_stage2_pred)
                    stage2_recall = recall_for_class(y_stage2_true, y_stage2_pred, 1)
                else:
                    stage2_metric = 0.0
                    stage2_recall = 0.0

                mask_stage3_val_eval = (
                    (np.asarray(p_val_stage1, dtype=float) < t1f)
                    & (np.asarray(p_val_stage2, dtype=float) < t2f)
                    & np.asarray(mask3_val, dtype=bool)
                )
                if np.any(mask_stage3_val_eval):
                    y_stage3_val_true = (np.asarray(y_val_arr)[mask_stage3_val_eval] == s3_pos_id).astype(int)
                    y_stage3_val_pred = (np.asarray(p_val_stage3, dtype=float)[mask_stage3_val_eval] >= t3f).astype(int)
                    stage3_bin = _binary_prf_from_predictions(y_stage3_val_true, y_stage3_val_pred)
                    stage3_precision = float(stage3_bin["precision"])
                    stage3_recall = float(stage3_bin["recall"])
                    stage3_f1 = float(stage3_bin["f1"])
                    stage3_fbeta = _fbeta_from_pr(stage3_precision, stage3_recall, args.stage3_threshold_beta)
                else:
                    stage3_precision = 0.0
                    stage3_recall = 0.0
                    stage3_f1 = 0.0
                    stage3_fbeta = 0.0

                score_payload = compute_stage3_objective_score(
                    args=args,
                    objective_name=str(args.stage3_threshold_objective),
                    val_macro_f1=val_macro_f1,
                    val_bal_acc=val_bal_acc,
                    dist_recall=dist_rec,
                    withdrawn_recall=wd_rec,
                    stage3_precision=stage3_precision,
                    stage3_recall=stage3_recall,
                    stage3_f1=stage3_f1,
                    stage3_fbeta=stage3_fbeta,
                )
                if str(args.arch_version) == "v2" and str(args.v2_routing_score_mode) == "collapse_aware":
                    routing_payload = compute_collapse_aware_score(
                        val_macro_f1=val_macro_f1,
                        val_acc=val_acc,
                        val_bal_acc=val_bal_acc,
                        per_class_recall=val_per_class_recall,
                        stage3_f1=stage3_f1,
                        args=args,
                    )
                else:
                    min_recall = min((float(v) for v in val_per_class_recall.values()), default=0.0)
                    routing_payload = {
                        "base_score": float(score_payload["objective_score"]),
                        "collapse_penalty": 0.0,
                        "distinction_bonus": 0.0,
                        "routing_score": float(score_payload["objective_score"]),
                        "min_recall": float(min_recall),
                        "distinction_recall": float(val_per_class_recall.get("distinction", 0.0)),
                    }

                cand = {
                    "t1": t1f,
                    "t2": t2f,
                    "t3": t3f,
                    "val_macro_f1": val_macro_f1,
                    "val_acc": val_acc,
                    "val_bal_acc": val_bal_acc,
                    "val_logloss": val_ll,
                    "dist_recall": dist_rec,
                    "withdrawn_recall": wd_rec,
                    "per_class_recall": val_per_class_recall,
                    "stage1_metric": float(stage1_metric),
                    "stage2_metric": float(stage2_metric),
                    "stage2_recall": float(stage2_recall),
                    "objective_score": float(score_payload["objective_score"]),
                    "stage3_metric": float(score_payload["stage3_metric"]),
                    "stage3_precision": float(score_payload["stage3_precision"]),
                    "stage3_recall": float(score_payload["stage3_recall"]),
                    "stage3_f1": float(score_payload["stage3_f1"]),
                    "stage3_fbeta": float(score_payload["stage3_fbeta"]),
                    "collapse_penalty": float(routing_payload["collapse_penalty"]),
                    "distinction_bonus": float(routing_payload["distinction_bonus"]),
                    "min_recall": float(routing_payload["min_recall"]),
                    "routing_score": float(routing_payload["routing_score"]),
                    "val_eval_full": val_eval,
                }
                if _better_threshold_payload(cand, best_any):
                    best_any = cand
                is_feasible = (
                    cand["dist_recall"] >= float(args.min_dist_recall)
                    and cand["withdrawn_recall"] >= float(args.min_withdrawn_recall)
                    and cand["stage2_recall"] >= float(args.stage2_min_pos_recall)
                    and cand["stage3_recall"] >= float(args.stage3_min_pos_recall)
                )
                if is_feasible:
                    feasible_configs += 1
                    if _better_threshold_payload(cand, best_constrained):
                        best_constrained = cand
                if len(lightweight_summary) < 25:
                    lightweight_summary.append(
                        {
                            "t1": t1f,
                            "t2": t2f,
                            "t3": t3f,
                            "objective_score": float(cand["objective_score"]),
                            "val_macro_f1": val_macro_f1,
                            "val_bal_acc": val_bal_acc,
                            "dist_recall": dist_rec,
                            "withdrawn_recall": wd_rec,
                            "stage2_recall": float(stage2_recall),
                            "stage3_f1": float(stage3_f1),
                            "stage3_recall": float(stage3_recall),
                            "feasible": bool(is_feasible),
                        }
                    )

    chosen = best_constrained if best_constrained is not None else best_any
    return {
        "chosen": chosen,
        "best_any": best_any,
        "best_constrained": best_constrained,
        "used_constrained_candidate": bool(best_constrained is not None),
        "threshold_search_total_configs": int(total_configs),
        "threshold_search_feasible_configs": int(feasible_configs),
        "threshold_search_best_any_score": None if best_any is None else float(best_any["objective_score"]),
        "threshold_search_best_feasible_score": None if best_constrained is None else float(best_constrained["objective_score"]),
        "search_summary": lightweight_summary,
    }
