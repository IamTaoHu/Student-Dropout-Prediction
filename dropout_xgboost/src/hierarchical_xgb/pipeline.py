from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

from hierarchical_xgb.config import OUTPUT_DIR, ROUTING_MODE_SPECS
from hierarchical_xgb.data_utils import load_data, resolve_input_path, save_json
from hierarchical_xgb.feature_engineering import (
    augment_distinction_specialist_features,
    augment_stage3_features,
    build_purpose_feature_sets,
    print_feature_source_summary,
    resolve_mode_feature_sets,
)
from hierarchical_xgb.model_utils import (
    _predict_proba_binary,
    apply_stage3_fail_weight,
    build_specialist_model,
    build_stage1_model,
    build_stage2_model,
    build_stage3_model,
    calibrate_prefit,
    fit_compat,
    get_model_best_iteration,
    hybrid_balance_binary,
    make_binary_for_label,
    make_binary_for_two_labels,
    make_stage1_sample_weight_mode,
    make_specialist_sample_weight,
    make_stage3_sample_weight,
    oversample_binary_minority,
    parse_routing_modes,
    parse_threshold_grid,
    resolve_stage3_calibration,
    resolve_stage_calibration,
    split_train_val_test,
    undersample_binary_majority,
    validate_binary_labels,
    value_counts_binary,
)
from hierarchical_xgb.reporting import (
    _save_overall_report_artifacts,
    _save_stage_report_artifacts,
    print_end_summary,
    print_run_header,
    save_summary_confusion_matrices,
    save_summary_text_report,
    save_v2_summary_report,
    select_best_mode_row,
)
from hierarchical_xgb.threshold_utils import (
    _build_stage_report_binary,
    _is_better_candidate,
    compute_collapse_aware_score,
    eval_4class_full,
    predict_3stage_from_proba,
    proba_3stage_hard,
    search_best_threshold_triplet,
)


def _binary_specialist_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true_bin, dtype=int)
    y_pred = np.asarray(y_pred_bin, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    pos_f1 = 0.0 if (precision + recall) == 0.0 else float(2.0 * precision * recall / (precision + recall))
    neg_precision = 0.0 if (tn + fn) == 0 else float(tn / (tn + fn))
    neg_recall = 0.0 if (tn + fp) == 0 else float(tn / (tn + fp))
    neg_f1 = 0.0 if (neg_precision + neg_recall) == 0.0 else float(2.0 * neg_precision * neg_recall / (neg_precision + neg_recall))
    return {
        "precision": precision,
        "recall": recall,
        "pos_f1": pos_f1,
        "macro_f1": float((pos_f1 + neg_f1) / 2.0),
    }


def _select_specialist_threshold(
    p_val_specialist: np.ndarray,
    y_val_specialist: np.ndarray,
    thresholds: list[float],
    min_recall: float | None,
) -> tuple[float | None, dict[str, float] | None]:
    best_candidate: tuple[float, dict[str, float]] | None = None
    for threshold in thresholds:
        pred_bin = (np.asarray(p_val_specialist, dtype=float) >= float(threshold)).astype(int)
        metrics = _binary_specialist_metrics(y_val_specialist, pred_bin)
        payload = (float(metrics["pos_f1"]), float(metrics["macro_f1"]), -float(threshold))
        if min_recall is None or float(metrics["recall"]) >= float(min_recall):
            if best_candidate is None or payload > (best_candidate[1]["pos_f1"], best_candidate[1]["macro_f1"], -float(best_candidate[0])):
                best_candidate = (float(threshold), metrics)
    if best_candidate is not None:
        return best_candidate
    return None, None


def _apply_pass_to_distinction_fusion(
    pred: np.ndarray,
    proba4_hard: np.ndarray,
    specialist_proba: np.ndarray,
    labels: list[str],
    threshold: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    pred_out = np.asarray(pred, dtype=int).copy()
    proba_out = np.asarray(proba4_hard, dtype=float).copy()
    pass_id = labels.index("pass")
    distinction_id = labels.index("distinction")
    specialist_arr = np.asarray(specialist_proba, dtype=float)
    pass_before = int(np.sum(pred_out == pass_id))
    eligible_mask = (
        (pred_out == pass_id)
        & (specialist_arr >= float(threshold))
        & ((specialist_arr - 0.5) >= float(margin))
    )
    flip_mask = eligible_mask
    if np.any(flip_mask):
        pred_out[flip_mask] = distinction_id
        proba_out[flip_mask, :] = 0.0
        proba_out[flip_mask, distinction_id] = 1.0
    return pred_out, proba_out, pass_before, int(np.sum(eligible_mask)), int(np.sum(flip_mask))


def _train_candidate_for_mode(
    args,
    seed: int,
    routing_mode: str,
    labels: list[str],
    purpose_feature_sets: dict[str, list[str]],
    thresholds_stage1: list[float],
    thresholds_stage2: list[float],
    thresholds_stage3: list[float],
    stage3_threshold_objective: str,
    resolved_stage1_calibration: str,
    resolved_stage2_calibration: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_arr: np.ndarray,
    y_val_arr: np.ndarray,
    y_test_arr: np.ndarray,
):
    spec = ROUTING_MODE_SPECS[routing_mode]
    s1_label = spec["stage1_target"]
    s2_label = spec["stage2_target"]
    s3_pos = spec["stage3_pos"]
    s3_neg = spec["stage3_neg"]

    X1_train = X_train[purpose_feature_sets[spec["stage1_feature_group"]]]
    X1_val = X_val[purpose_feature_sets[spec["stage1_feature_group"]]]
    X1_test = X_test[purpose_feature_sets[spec["stage1_feature_group"]]]
    X2_train = X_train[purpose_feature_sets[spec["stage2_feature_group"]]]
    X2_val = X_val[purpose_feature_sets[spec["stage2_feature_group"]]]
    X2_test = X_test[purpose_feature_sets[spec["stage2_feature_group"]]]

    X3_train_full = augment_stage3_features(
        X_train[purpose_feature_sets[spec["stage3_feature_group"]]],
        mode=args.stage3_feature_engineering,
        stage3_plus=args.stage3_plus,
    )
    X3_val_full = augment_stage3_features(
        X_val[purpose_feature_sets[spec["stage3_feature_group"]]],
        mode=args.stage3_feature_engineering,
        stage3_plus=args.stage3_plus,
    )
    X3_test_full = augment_stage3_features(
        X_test[purpose_feature_sets[spec["stage3_feature_group"]]],
        mode=args.stage3_feature_engineering,
        stage3_plus=args.stage3_plus,
    )

    y1_train = make_binary_for_label(y_train_arr, labels, s1_label)
    y1_val = make_binary_for_label(y_val_arr, labels, s1_label)
    try:
        validate_binary_labels(y1_train, "stage1")
    except ValueError:
        print(f"seed={seed} mode={routing_mode} | skipped (stage1 missing positive/negative class)")
        return None
    w1_train, stage1_weight_info = make_stage1_sample_weight_mode(y1_train, args.stage1_class_weight_mode, args.stage1_pos_multiplier)
    m1 = fit_compat(build_stage1_model(args, seed), X1_train, y1_train, sample_weight=w1_train, X_val=X1_val, y_val=y1_val)
    can_cal_stage1 = len(np.unique(y1_val)) >= 2
    m1c = calibrate_prefit(m1, X1_val, y1_val, resolved_stage1_calibration) if (can_cal_stage1 and resolved_stage1_calibration != "none") else m1

    y2_train = make_binary_for_label(y_train_arr, labels, s2_label)
    y2_val = make_binary_for_label(y_val_arr, labels, s2_label)
    try:
        validate_binary_labels(y2_train, "stage2")
    except ValueError:
        print(f"seed={seed} mode={routing_mode} | skipped (stage2 missing positive/negative class)")
        return None
    w2_train, stage2_weight_info = make_stage3_sample_weight(
        y2_train,
        mode=args.stage2_class_weight_mode,
        pos_multiplier=args.stage2_pos_multiplier,
    )
    m2 = fit_compat(build_stage2_model(args, seed), X2_train, y2_train, sample_weight=w2_train, X_val=X2_val, y_val=y2_val)
    can_cal_stage2 = len(np.unique(y2_val)) >= 2
    m2c = calibrate_prefit(m2, X2_val, y2_val, resolved_stage2_calibration) if (can_cal_stage2 and resolved_stage2_calibration != "none") else m2

    s3_pos_id = labels.index(s3_pos)
    s3_neg_id = labels.index(s3_neg)
    mask3_train = np.isin(y_train_arr, [s3_pos_id, s3_neg_id])
    X3_train_raw = X3_train_full.loc[mask3_train].copy()
    if X3_train_raw.shape[0] == 0:
        print(f"seed={seed} mode={routing_mode} | skipped (no rows for stage3 train)")
        return None
    y3_train_raw = (y_train_arr[mask3_train] == s3_pos_id).astype(int)
    try:
        validate_binary_labels(y3_train_raw, "stage3")
    except ValueError:
        print(f"seed={seed} mode={routing_mode} | skipped (stage3 missing positive/negative class)")
        return None

    stage3_balance_report = {"strategy": "none", "applied": False}
    X3_train = X3_train_raw
    y3_train = y3_train_raw
    if bool(args.stage3_plus):
        stage3_balance_mode = str(args.stage3_balance_train).strip().lower()
        stage3_balance_report = {
            "strategy": stage3_balance_mode,
            "applied": False,
            "before_counts": value_counts_binary(y3_train_raw),
            "after_counts": value_counts_binary(y3_train_raw),
            "majority_label": None,
            "minority_label": None,
        }
        if stage3_balance_mode == "undersample_majority":
            X3_train, y3_train, stage3_balance_report = undersample_binary_majority(
                X3_train_raw,
                y3_train_raw,
                target_ratio=float(args.stage3_balance_ratio),
                random_state=int(seed),
            )
        elif stage3_balance_mode == "oversample_minority":
            X3_train, y3_train, stage3_balance_report = oversample_binary_minority(
                X3_train_raw,
                y3_train_raw,
                target_ratio=float(args.stage3_balance_ratio),
                random_state=int(seed),
            )
        elif stage3_balance_mode == "hybrid":
            X3_train, y3_train, stage3_balance_report = hybrid_balance_binary(
                X3_train_raw,
                y3_train_raw,
                ratio=float(args.stage3_balance_ratio),
                random_state=int(seed),
            )

    try:
        validate_binary_labels(y3_train, "stage3_balanced")
    except ValueError:
        print(f"seed={seed} mode={routing_mode} | skipped (stage3 balancing removed class diversity)")
        return None
    stage3_raw_counts = value_counts_binary(y3_train_raw)
    stage3_bal_counts = value_counts_binary(y3_train)
    stage3_balance_strategy = str(stage3_balance_report.get("strategy", "none"))
    w3_train, stage3_weight_info = make_stage3_sample_weight(
        y3_train,
        mode=args.stage3_class_weight_mode,
        pos_multiplier=args.stage3_pos_multiplier,
    )
    w3_train, fail_weight_info = apply_stage3_fail_weight(
        w3_train,
        y3_train,
        fail_weight=args.stage3_fail_weight,
        stage3_pos_label=s3_pos,
        stage3_neg_label=s3_neg,
    )
    stage3_weight_info["fail_weight"] = fail_weight_info["fail_weight"]
    stage3_weight_info["fail_weight_applied_to"] = fail_weight_info["applied_to"]

    mask3_val = np.isin(y_val_arr, [s3_pos_id, s3_neg_id])
    X3_val = X3_val_full.loc[mask3_val]
    y3_val = (y_val_arr[mask3_val] == s3_pos_id).astype(int)
    m3 = fit_compat(
        build_stage3_model(args, seed),
        X3_train,
        y3_train,
        sample_weight=w3_train,
        X_val=X3_val if len(X3_val) > 0 else None,
        y_val=y3_val if len(X3_val) > 0 else None,
    )
    can_calibrate_stage3 = len(X3_val) > 0 and len(np.unique(y3_val)) >= 2
    resolved_stage3_calibration = resolve_stage3_calibration(args)
    m3c = calibrate_prefit(m3, X3_val, y3_val, resolved_stage3_calibration) if can_calibrate_stage3 else m3

    specialist_enabled_requested = str(args.arch_version) == "v2" and bool(args.v2_specialist_enable)
    specialist_model = None
    specialist_feature_columns: list[str] = []
    v2_specialist_threshold = float(args.v2_specialist_threshold)
    v2_specialist_val_f1 = 0.0
    v2_specialist_val_recall = 0.0
    v2_specialist_test_f1 = 0.0
    v2_specialist_test_recall = 0.0
    v2_specialist_train_rows = 0
    v2_specialist_val_rows = 0
    v2_specialist_test_rows = 0
    v2_specialist_train_counts = {0: 0, 1: 0}
    v2_specialist_val_counts = {0: 0, 1: 0}
    v2_specialist_test_counts = {0: 0, 1: 0}
    v2_specialist_val_proba_min = 0.0
    v2_specialist_val_proba_max = 0.0
    v2_specialist_val_proba_mean = 0.0
    v2_specialist_test_proba_min = 0.0
    v2_specialist_test_proba_max = 0.0
    v2_specialist_test_proba_mean = 0.0
    p_val_specialist_full = np.zeros(shape=(len(X_val),), dtype=float)
    p_test_specialist_full = np.zeros(shape=(len(X_test),), dtype=float)

    if specialist_enabled_requested:
        specialist_base_cols = (
            purpose_feature_sets[spec["stage3_feature_group"]]
            if routing_mode == "fail_first"
            else purpose_feature_sets["distinction"]
        )
        Xsp_train_full = augment_distinction_specialist_features(
            X_train[specialist_base_cols],
            stage3_plus=args.stage3_plus,
        )
        Xsp_val_full = augment_distinction_specialist_features(
            X_val[specialist_base_cols],
            stage3_plus=args.stage3_plus,
        )
        Xsp_test_full = augment_distinction_specialist_features(
            X_test[specialist_base_cols],
            stage3_plus=args.stage3_plus,
        )
        specialist_feature_columns = Xsp_train_full.columns.tolist()

        distinction_id = labels.index("distinction")
        pass_id = labels.index("pass")
        mask_sp_train, ysp_train = make_binary_for_two_labels(y_train_arr, distinction_id, pass_id)
        mask_sp_val, ysp_val = make_binary_for_two_labels(y_val_arr, distinction_id, pass_id)
        mask_sp_test, ysp_test = make_binary_for_two_labels(y_test_arr, distinction_id, pass_id)
        v2_specialist_train_rows = int(np.sum(mask_sp_train))
        v2_specialist_val_rows = int(np.sum(mask_sp_val))
        v2_specialist_test_rows = int(np.sum(mask_sp_test))
        if v2_specialist_train_rows > 0:
            v2_specialist_train_counts = value_counts_binary(ysp_train)
        if v2_specialist_val_rows > 0:
            v2_specialist_val_counts = value_counts_binary(ysp_val)
        if v2_specialist_test_rows > 0:
            v2_specialist_test_counts = value_counts_binary(ysp_test)

        print(
            f"seed={seed} mode={routing_mode} | "
            f"specialist rows train={v2_specialist_train_rows}, val={v2_specialist_val_rows}, test={v2_specialist_test_rows}"
        )
        print(
            f"seed={seed} mode={routing_mode} | "
            f"specialist counts "
            f"train(0={v2_specialist_train_counts[0]},1={v2_specialist_train_counts[1]}) "
            f"val(0={v2_specialist_val_counts[0]},1={v2_specialist_val_counts[1]}) "
            f"test(0={v2_specialist_test_counts[0]},1={v2_specialist_test_counts[1]})"
        )
        if v2_specialist_train_rows == 0:
            print(f"seed={seed} mode={routing_mode} | specialist train mask has zero rows; skipping specialist training")
        if v2_specialist_val_rows == 0:
            print(f"seed={seed} mode={routing_mode} | specialist val mask has zero rows; threshold search will use fallback threshold")
        if v2_specialist_test_rows == 0:
            print(f"seed={seed} mode={routing_mode} | specialist test mask has zero rows; test specialist metrics remain default")

        if np.any(mask_sp_train) and len(np.unique(ysp_train)) >= 2:
            Xsp_train = Xsp_train_full.loc[mask_sp_train].copy()
            Xsp_val = Xsp_val_full.loc[mask_sp_val].copy()
            Xsp_test = Xsp_test_full.loc[mask_sp_test].copy()
            wsp_train, _specialist_weight_info = make_specialist_sample_weight(ysp_train, pos_multiplier=1.5)
            specialist_model_raw = fit_compat(
                build_specialist_model(args, seed, model_name=args.v2_specialist_model),
                Xsp_train,
                ysp_train,
                sample_weight=wsp_train,
                X_val=Xsp_val if len(Xsp_val) > 0 else None,
                y_val=ysp_val if len(Xsp_val) > 0 else None,
            )
            # Temporary debug: use raw specialist probabilities to avoid over-compressed calibrated scores
            specialist_model = specialist_model_raw

            if np.any(mask_sp_val):
                p_val_specialist_full[mask_sp_val] = _predict_proba_binary(specialist_model, Xsp_val_full.loc[mask_sp_val])
                val_specialist_proba = p_val_specialist_full[mask_sp_val]
                v2_specialist_val_proba_min = float(np.min(val_specialist_proba))
                v2_specialist_val_proba_max = float(np.max(val_specialist_proba))
                v2_specialist_val_proba_mean = float(np.mean(val_specialist_proba))
                print(
                    f"seed={seed} mode={routing_mode} | "
                    f"specialist val proba stats min={v2_specialist_val_proba_min:.4f} "
                    f"max={v2_specialist_val_proba_max:.4f} mean={v2_specialist_val_proba_mean:.4f}"
                )
            if np.any(mask_sp_test):
                p_test_specialist_full[mask_sp_test] = _predict_proba_binary(specialist_model, Xsp_test_full.loc[mask_sp_test])
                test_specialist_proba = p_test_specialist_full[mask_sp_test]
                v2_specialist_test_proba_min = float(np.min(test_specialist_proba))
                v2_specialist_test_proba_max = float(np.max(test_specialist_proba))
                v2_specialist_test_proba_mean = float(np.mean(test_specialist_proba))
                print(
                    f"seed={seed} mode={routing_mode} | "
                    f"specialist test proba stats min={v2_specialist_test_proba_min:.4f} "
                    f"max={v2_specialist_test_proba_max:.4f} mean={v2_specialist_test_proba_mean:.4f}"
                )

            specialist_thresholds = parse_threshold_grid(args.v2_specialist_threshold_grid, "--v2_specialist_threshold_grid")
            if specialist_thresholds and all(float(t) >= 0.5 for t in specialist_thresholds):
                specialist_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
            print(f"seed={seed} mode={routing_mode} | specialist threshold grid used={specialist_thresholds}")
            if np.any(mask_sp_val):
                v2_specialist_threshold_search, specialist_val_metrics = _select_specialist_threshold(
                    p_val_specialist_full[mask_sp_val],
                    ysp_val,
                    specialist_thresholds,
                    float(args.v2_specialist_min_recall),
                )
                threshold_source = "feasible"
                if v2_specialist_threshold_search is None or specialist_val_metrics is None:
                    v2_specialist_threshold_search, specialist_val_metrics = _select_specialist_threshold(
                        p_val_specialist_full[mask_sp_val],
                        ysp_val,
                        specialist_thresholds,
                        None,
                    )
                    threshold_source = "unconstrained"
                if v2_specialist_threshold_search is None or specialist_val_metrics is None:
                    v2_specialist_threshold = float(args.v2_specialist_threshold)
                    specialist_val_pred = (p_val_specialist_full[mask_sp_val] >= float(v2_specialist_threshold)).astype(int)
                    specialist_val_metrics = _binary_specialist_metrics(ysp_val, specialist_val_pred)
                    print(
                        f"seed={seed} mode={routing_mode} | "
                        f"specialist threshold search found no feasible candidate; fallback_threshold={v2_specialist_threshold:.4f}"
                    )
                    print(f"seed={seed} mode={routing_mode} | specialist threshold source=scalar_fallback")
                else:
                    v2_specialist_threshold = float(v2_specialist_threshold_search)
                    print(f"seed={seed} mode={routing_mode} | specialist threshold source={threshold_source}")
                v2_specialist_val_f1 = float(specialist_val_metrics["pos_f1"])
                v2_specialist_val_recall = float(specialist_val_metrics["recall"])
                print(
                    f"seed={seed} mode={routing_mode} | "
                    f"specialist threshold={v2_specialist_threshold:.4f} "
                    f"val_prec={float(specialist_val_metrics['precision']):.4f} "
                    f"val_rec={float(specialist_val_metrics['recall']):.4f} "
                    f"val_f1={float(specialist_val_metrics['pos_f1']):.4f}"
                )
            if np.any(mask_sp_test):
                specialist_test_pred = (p_test_specialist_full[mask_sp_test] >= float(v2_specialist_threshold)).astype(int)
                specialist_test_metrics = _binary_specialist_metrics(ysp_test, specialist_test_pred)
                v2_specialist_test_f1 = float(specialist_test_metrics["pos_f1"])
                v2_specialist_test_recall = float(specialist_test_metrics["recall"])
            if np.any(mask_sp_val) or np.any(mask_sp_test):
                val_positive_count = int(np.sum((p_val_specialist_full[mask_sp_val] >= float(v2_specialist_threshold)).astype(int))) if np.any(mask_sp_val) else 0
                test_positive_count = int(np.sum((p_test_specialist_full[mask_sp_test] >= float(v2_specialist_threshold)).astype(int))) if np.any(mask_sp_test) else 0
                print(
                    f"seed={seed} mode={routing_mode} | "
                    f"specialist positives val={val_positive_count} test={test_positive_count}"
                )
        else:
            print(
                f"seed={seed} mode={routing_mode} | "
                f"specialist train subset missing class diversity or zero rows; training skipped"
            )

    p_val_stage1 = _predict_proba_binary(m1c, X1_val)
    p_val_stage2 = _predict_proba_binary(m2c, X2_val)
    p_val_stage3 = np.full(shape=(len(X_val),), fill_value=0.5, dtype=float)
    if np.any(mask3_val):
        p_val_stage3[mask3_val] = _predict_proba_binary(m3c, X3_val_full.loc[mask3_val])

    threshold_result = search_best_threshold_triplet(
        p_val_stage1=p_val_stage1,
        p_val_stage2=p_val_stage2,
        p_val_stage3=p_val_stage3,
        y_val_arr=y_val_arr,
        mask3_val=mask3_val,
        labels=labels,
        s1_label=s1_label,
        s2_label=s2_label,
        s3_pos=s3_pos,
        s3_neg=s3_neg,
        thresholds_stage1=thresholds_stage1,
        thresholds_stage2=thresholds_stage2,
        thresholds_stage3=thresholds_stage3,
        args=args,
    )
    chosen = threshold_result["chosen"]
    if chosen is None:
        print(f"seed={seed} mode={routing_mode} | skipped (threshold search produced no candidate)")
        return None
    best_t1 = chosen["t1"]
    best_t2 = chosen["t2"]
    best_t3 = chosen["t3"]

    print(
        f"seed={seed} mode={routing_mode} | "
        f"s3_counts raw(0={stage3_raw_counts[0]},1={stage3_raw_counts[1]}) "
        f"bal(0={stage3_bal_counts[0]},1={stage3_bal_counts[1]}) "
        f"strategy={stage3_balance_strategy}"
    )
    print(
        f"seed={seed} mode={routing_mode} | "
        f"s3 model={args.stage3_model} calib={resolved_stage3_calibration} "
        f"metric={args.stage3_threshold_metric} t3={float(best_t3):.4f} "
        f"metric_val={float(chosen['stage3_metric']):.4f}"
    )

    p_test_stage1 = _predict_proba_binary(m1c, X1_test)
    p_test_stage2 = _predict_proba_binary(m2c, X2_test)
    p_test_stage3 = np.full(shape=(len(X_test),), fill_value=0.5, dtype=float)
    mask3_test = np.isin(y_test_arr, [s3_pos_id, s3_neg_id])
    if np.any(mask3_test):
        p_test_stage3[mask3_test] = _predict_proba_binary(m3c, X3_test_full.loc[mask3_test])

    test_pred = predict_3stage_from_proba(
        p_test_stage1, p_test_stage2, p_test_stage3, labels, best_t1, best_t2, s1_label, s2_label, s3_pos, s3_neg, t3=float(best_t3)
    )
    test_proba4_hard = proba_3stage_hard(
        p_test_stage1, p_test_stage2, p_test_stage3, labels, best_t1, best_t2, s1_label, s2_label, s3_pos, s3_neg, t3=float(best_t3)
    )
    test_eval_full = eval_4class_full(y_test_arr, test_pred, y_proba=test_proba4_hard, labels=labels)
    test_macro_f1 = float(test_eval_full["macro_f1"])
    test_acc = float(test_eval_full["accuracy"])
    test_bal_acc = float(test_eval_full["balanced_accuracy"])
    test_logloss = float(test_eval_full["logloss"]) if test_eval_full["logloss"] is not None else float("inf")
    test_rep = test_eval_full["classification_report"]

    stage1_pred = (p_test_stage1 >= float(best_t1)).astype(int)
    y_stage1_true = make_binary_for_label(y_test_arr, labels, s1_label)
    stage1_report = _build_stage_report_binary(y_stage1_true, stage1_pred, "rest", s1_label)

    mask_stage2_eval = stage1_pred == 0
    if np.any(mask_stage2_eval):
        y_stage2_true = make_binary_for_label(y_test_arr[mask_stage2_eval], labels, s2_label)
        stage2_pred = (p_test_stage2[mask_stage2_eval] >= float(best_t2)).astype(int)
        stage2_report = _build_stage_report_binary(y_stage2_true, stage2_pred, "rest", s2_label)
    else:
        stage2_report = {"classification_report": None, "confusion_matrix": None, "accuracy": None, "balanced_accuracy": None, "num_samples": 0}

    mask_stage3_eval = mask_stage2_eval & (p_test_stage2 < float(best_t2))
    if np.any(mask_stage3_eval):
        y_stage3_true = (y_test_arr[mask_stage3_eval] == s3_pos_id).astype(int)
        stage3_pred = (p_test_stage3[mask_stage3_eval] >= float(best_t3)).astype(int)
        stage3_report = _build_stage_report_binary(y_stage3_true, stage3_pred, s3_neg, s3_pos)
    else:
        stage3_report = {"classification_report": None, "confusion_matrix": None, "accuracy": None, "balanced_accuracy": None, "num_samples": 0}

    val_pred = predict_3stage_from_proba(
        p_val_stage1, p_val_stage2, p_val_stage3, labels, best_t1, best_t2, s1_label, s2_label, s3_pos, s3_neg, t3=float(best_t3)
    )
    val_proba4_hard = proba_3stage_hard(
        p_val_stage1, p_val_stage2, p_val_stage3, labels, best_t1, best_t2, s1_label, s2_label, s3_pos, s3_neg, t3=float(best_t3)
    )
    v2_fusion_enabled = (
        str(args.arch_version) == "v2"
        and bool(args.v2_specialist_enable)
        and bool(args.v2_pass_to_distinction_enable)
        and specialist_model is not None
    )
    v2_num_pass_predictions_before_fusion_val = 0
    v2_num_pass_predictions_before_fusion_test = 0
    v2_num_specialist_eligible_val = 0
    v2_num_specialist_eligible_test = 0
    v2_num_pass_to_distinction_flips_val = 0
    v2_num_pass_to_distinction_flips_test = 0
    if v2_fusion_enabled:
        fusion_threshold = float(min(float(v2_specialist_threshold), float(args.v2_pass_to_distinction_threshold)))
        if str(args.arch_version) == "v2":
            fusion_threshold = min(fusion_threshold, 0.30)
        print(f"seed={seed} mode={routing_mode} | fusion threshold used={fusion_threshold:.4f}")
        val_pred, val_proba4_hard, v2_num_pass_predictions_before_fusion_val, v2_num_specialist_eligible_val, v2_num_pass_to_distinction_flips_val = _apply_pass_to_distinction_fusion(
            val_pred,
            val_proba4_hard,
            p_val_specialist_full,
            labels,
            threshold=float(fusion_threshold),
            margin=float(args.v2_pass_to_distinction_margin),
        )
        test_pred, test_proba4_hard, v2_num_pass_predictions_before_fusion_test, v2_num_specialist_eligible_test, v2_num_pass_to_distinction_flips_test = _apply_pass_to_distinction_fusion(
            test_pred,
            test_proba4_hard,
            p_test_specialist_full,
            labels,
            threshold=float(fusion_threshold),
            margin=float(args.v2_pass_to_distinction_margin),
        )
        print(
            f"seed={seed} mode={routing_mode} | "
            f"fusion pass_before_val={v2_num_pass_predictions_before_fusion_val} "
            f"pass_before_test={v2_num_pass_predictions_before_fusion_test} "
            f"eligible_val={v2_num_specialist_eligible_val} "
            f"eligible_test={v2_num_specialist_eligible_test} "
            f"flips_val={v2_num_pass_to_distinction_flips_val} "
            f"flips_test={v2_num_pass_to_distinction_flips_test}"
        )

    val_eval_full = eval_4class_full(y_val_arr, val_pred, y_proba=val_proba4_hard, labels=labels)
    test_eval_full = eval_4class_full(y_test_arr, test_pred, y_proba=test_proba4_hard, labels=labels)
    val_macro_f1_final = float(val_eval_full["macro_f1"])
    val_acc_final = float(val_eval_full["accuracy"])
    val_bal_acc_final = float(val_eval_full["balanced_accuracy"])
    val_dist_recall_final = float(val_eval_full["per_class_recall"].get("distinction", 0.0))
    val_withdrawn_recall_final = float(val_eval_full["per_class_recall"].get("withdrawn", 0.0))
    test_macro_f1 = float(test_eval_full["macro_f1"])
    test_acc = float(test_eval_full["accuracy"])
    test_bal_acc = float(test_eval_full["balanced_accuracy"])
    test_logloss = float(test_eval_full["logloss"]) if test_eval_full["logloss"] is not None else float("inf")
    test_rep = test_eval_full["classification_report"]

    if str(args.arch_version) == "v2" and str(args.v2_routing_score_mode) == "collapse_aware":
        routing_payload = {
            **chosen,
            **compute_collapse_aware_score(
                val_macro_f1=val_macro_f1_final,
                val_acc=val_acc_final,
                val_bal_acc=val_bal_acc_final,
                per_class_recall=val_eval_full["per_class_recall"],
                stage3_f1=float(chosen["stage3_f1"]),
                args=args,
            ),
        }
    else:
        routing_payload = {
            "routing_score": float(chosen.get("routing_score", chosen["objective_score"])),
            "collapse_penalty": float(chosen.get("collapse_penalty", 0.0)),
            "distinction_bonus": float(chosen.get("distinction_bonus", 0.0)),
            "min_recall": float(chosen.get("min_recall", min(val_eval_full["per_class_recall"].values(), default=0.0))),
        }

    stage1_val_pred = (p_val_stage1 >= float(best_t1)).astype(int)
    y_stage1_val_true = make_binary_for_label(y_val_arr, labels, s1_label)
    stage1_report_val = _build_stage_report_binary(y_stage1_val_true, stage1_val_pred, "rest", s1_label)
    mask_stage2_val_eval = stage1_val_pred == 0
    if np.any(mask_stage2_val_eval):
        y_stage2_val_true = make_binary_for_label(y_val_arr[mask_stage2_val_eval], labels, s2_label)
        stage2_val_pred = (p_val_stage2[mask_stage2_val_eval] >= float(best_t2)).astype(int)
        stage2_report_val = _build_stage_report_binary(y_stage2_val_true, stage2_val_pred, "rest", s2_label)
    else:
        stage2_report_val = {"classification_report": None, "confusion_matrix": None, "accuracy": None, "balanced_accuracy": None, "num_samples": 0}
    mask_stage3_val_eval = mask_stage2_val_eval & (p_val_stage2 < float(best_t2))
    if np.any(mask_stage3_val_eval):
        y_stage3_val_true = (y_val_arr[mask_stage3_val_eval] == s3_pos_id).astype(int)
        stage3_val_pred = (p_val_stage3[mask_stage3_val_eval] >= float(best_t3)).astype(int)
        stage3_report_val = _build_stage_report_binary(y_stage3_val_true, stage3_val_pred, s3_neg, s3_pos)
    else:
        stage3_report_val = {"classification_report": None, "confusion_matrix": None, "accuracy": None, "balanced_accuracy": None, "num_samples": 0}

    routed_counts_val = {
        "stage1_positive_count": int(np.sum(stage1_val_pred == 1)),
        "stage1_negative_count": int(np.sum(stage1_val_pred == 0)),
        "stage2_routed_count": int(np.sum(mask_stage2_val_eval)),
        "stage3_routed_count": int(np.sum(mask_stage3_val_eval)),
        "stage3_positive_count": int(np.sum(y_val_arr[mask_stage3_val_eval] == s3_pos_id)) if np.any(mask_stage3_val_eval) else 0,
        "stage3_negative_count": int(np.sum(y_val_arr[mask_stage3_val_eval] == s3_neg_id)) if np.any(mask_stage3_val_eval) else 0,
    }
    routed_counts_test = {
        "stage1_positive_count": int(np.sum(stage1_pred == 1)),
        "stage1_negative_count": int(np.sum(stage1_pred == 0)),
        "stage2_routed_count": int(np.sum(mask_stage2_eval)),
        "stage3_routed_count": int(np.sum(mask_stage3_eval)),
        "stage3_positive_count": int(np.sum(y_test_arr[mask_stage3_eval] == s3_pos_id)) if np.any(mask_stage3_eval) else 0,
        "stage3_negative_count": int(np.sum(y_test_arr[mask_stage3_eval] == s3_neg_id)) if np.any(mask_stage3_eval) else 0,
    }

    return {
        "seed": int(seed),
        "routing_mode": routing_mode,
        "best_t1": float(best_t1),
        "best_t2": float(best_t2),
        "best_t3": float(best_t3),
        "val_macro_f1": float(val_macro_f1_final),
        "val_dist_recall": float(val_dist_recall_final),
        "val_withdrawn_recall": float(val_withdrawn_recall_final),
        "val_bal_acc": float(val_bal_acc_final),
        "val_acc": float(val_acc_final),
        "val_logloss": float(val_eval_full["logloss"]) if val_eval_full["logloss"] is not None else float("inf"),
        "val_stage3_precision": float(chosen["stage3_precision"]),
        "val_stage3_recall": float(chosen["stage3_recall"]),
        "val_stage3_f1": float(chosen["stage3_f1"]),
        "val_stage3_fbeta": float(chosen["stage3_fbeta"]),
        "val_stage3_metric": float(chosen["stage3_metric"]),
        "routing_score": float(routing_payload["routing_score"]),
        "collapse_penalty": float(routing_payload["collapse_penalty"]),
        "distinction_bonus": float(routing_payload["distinction_bonus"]),
        "min_recall": float(routing_payload["min_recall"]),
        "val_per_class_recall": val_eval_full["per_class_recall"],
        "val_per_class_precision": val_eval_full["per_class_precision"],
        "val_stage1_metrics": stage1_report_val,
        "val_stage2_metrics": stage2_report_val,
        "val_stage3_metrics": stage3_report_val,
        "stage3_threshold_objective": stage3_threshold_objective,
        "stage3_objective_score": float(chosen["objective_score"]),
        "used_constrained_candidate": bool(threshold_result["used_constrained_candidate"]),
        "threshold_search_total_configs": int(threshold_result["threshold_search_total_configs"]),
        "threshold_search_feasible_configs": int(threshold_result["threshold_search_feasible_configs"]),
        "threshold_search_best_any_score": threshold_result["threshold_search_best_any_score"],
        "threshold_search_best_feasible_score": threshold_result["threshold_search_best_feasible_score"],
        "threshold_search_summary": threshold_result["search_summary"],
        "stage3_calibrated": bool(can_calibrate_stage3 and str(resolved_stage3_calibration).lower() != "none"),
        "stage3_calibration_resolved": str(resolved_stage3_calibration),
        "stage1_calibration_resolved": str(resolved_stage1_calibration),
        "stage2_calibration_resolved": str(resolved_stage2_calibration),
        "test_macro_f1": float(test_macro_f1),
        "test_bal_acc": float(test_bal_acc),
        "test_acc": float(test_acc),
        "test_logloss": test_logloss,
        "report": test_rep,
        "confusion_matrix": confusion_matrix(y_test_arr, test_pred, labels=list(range(len(labels)))).tolist(),
        "v2_specialist_enabled": bool(specialist_model is not None),
        "v2_specialist_threshold": float(v2_specialist_threshold),
        "v2_specialist_val_f1": float(v2_specialist_val_f1),
        "v2_specialist_val_recall": float(v2_specialist_val_recall),
        "v2_specialist_test_f1": float(v2_specialist_test_f1),
        "v2_specialist_test_recall": float(v2_specialist_test_recall),
        "v2_specialist_train_rows": int(v2_specialist_train_rows),
        "v2_specialist_val_rows": int(v2_specialist_val_rows),
        "v2_specialist_test_rows": int(v2_specialist_test_rows),
        "v2_specialist_train_counts": v2_specialist_train_counts,
        "v2_specialist_val_counts": v2_specialist_val_counts,
        "v2_specialist_test_counts": v2_specialist_test_counts,
        "v2_specialist_val_proba_min": float(v2_specialist_val_proba_min),
        "v2_specialist_val_proba_max": float(v2_specialist_val_proba_max),
        "v2_specialist_val_proba_mean": float(v2_specialist_val_proba_mean),
        "v2_specialist_test_proba_min": float(v2_specialist_test_proba_min),
        "v2_specialist_test_proba_max": float(v2_specialist_test_proba_max),
        "v2_specialist_test_proba_mean": float(v2_specialist_test_proba_mean),
        "v2_num_pass_predictions_before_fusion_val": int(v2_num_pass_predictions_before_fusion_val),
        "v2_num_pass_predictions_before_fusion_test": int(v2_num_pass_predictions_before_fusion_test),
        "v2_num_specialist_eligible_val": int(v2_num_specialist_eligible_val),
        "v2_num_specialist_eligible_test": int(v2_num_specialist_eligible_test),
        "v2_num_pass_to_distinction_flips_val": int(v2_num_pass_to_distinction_flips_val),
        "v2_num_pass_to_distinction_flips_test": int(v2_num_pass_to_distinction_flips_test),
        "v2_fusion_enabled": bool(v2_fusion_enabled),
        "stage_reports": {
            "validation": {
                f"stage1_{s1_label}_vs_rest": stage1_report_val,
                f"stage2_{s2_label}_vs_rest": stage2_report_val,
                f"stage3_{s3_pos}_vs_{s3_neg}": stage3_report_val,
            },
            "test": {
                f"stage1_{s1_label}_vs_rest": stage1_report,
                f"stage2_{s2_label}_vs_rest": stage2_report,
                f"stage3_{s3_pos}_vs_{s3_neg}": stage3_report,
            },
        },
        "routed_counts": {
            "validation": routed_counts_val,
            "test": routed_counts_test,
        },
        "y_test": y_test_arr,
        "test_pred": test_pred,
        "test_proba4": test_proba4_hard,
        "p_test_stage1": p_test_stage1,
        "p_test_stage2": p_test_stage2,
        "p_test_stage3": p_test_stage3,
        "p_test_specialist": p_test_specialist_full,
        "stage1_model": m1c,
        "stage2_model": m2c,
        "stage3_model": m3c,
        "specialist_model": specialist_model,
        "stage1_iteration": int(get_model_best_iteration(m1, args.n_estimators)),
        "stage2_iteration": int(get_model_best_iteration(m2, args.n_estimators)),
        "stage3_iteration": int(get_model_best_iteration(m3, args.n_estimators)),
        "stage1_weight_info": stage1_weight_info,
        "stage2_weight_info": stage2_weight_info,
        "stage3_weight_info": stage3_weight_info,
        "stage3_balance_report": stage3_balance_report,
        "stage1_feature_columns": purpose_feature_sets[spec["stage1_feature_group"]],
        "stage2_feature_columns": purpose_feature_sets[spec["stage2_feature_group"]],
        "stage3_feature_columns": X3_train_full.columns.tolist(),
        "specialist_feature_columns": specialist_feature_columns,
        "stage1_target": s1_label,
        "stage2_target": s2_label,
        "stage3_pos": s3_pos,
        "stage3_neg": s3_neg,
        "selection_validation_metrics": val_eval_full,
    }


def run_hierarchical_training(args) -> dict:
    """Run the full hierarchical training pipeline and persist all artifacts."""
    resolved_input = resolve_input_path(args.input)
    X_raw, y, labels, label_mapping = load_data(resolved_input)
    if len(labels) != 4:
        raise RuntimeError(f"Hierarchical setup expects 4 classes, got {len(labels)}.")

    X, stage_feature_sets, feature_counts = resolve_mode_feature_sets(X_raw, args.mode)
    routing_modes = parse_routing_modes(args.routing_modes)
    purpose_feature_sets = build_purpose_feature_sets(stage_feature_sets)
    print_feature_source_summary(X_raw, X)
    print_run_header(labels, label_mapping, y, args.mode, routing_modes, purpose_feature_sets, feature_counts)

    thresholds_stage1 = parse_threshold_grid(args.stage1_threshold_grid, "--stage1_threshold_grid")
    thresholds_stage2 = parse_threshold_grid(args.stage2_threshold_grid, "--stage2_threshold_grid")
    use_stage3_threshold_optimization = bool(str(args.stage3_threshold_grid).strip())
    if use_stage3_threshold_optimization:
        thresholds_stage3 = parse_threshold_grid(args.stage3_threshold_grid, "--stage3_threshold_grid")
    else:
        if bool(args.stage3_plus):
            default_stage3_grid = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60"
            thresholds_stage3 = parse_threshold_grid(default_stage3_grid, "--stage3_threshold_grid(default_stage3_plus)")
        else:
            if float(args.stage3_threshold) < 0.0 or float(args.stage3_threshold) > 1.0:
                raise ValueError("--stage3-threshold must be within [0,1].")
            thresholds_stage3 = [float(args.stage3_threshold)]
    stage3_threshold_objective = str(args.stage3_threshold_objective).strip().lower()
    resolved_stage1_calibration = resolve_stage_calibration(args.stage1_calibration, args.calibration)
    resolved_stage2_calibration = resolve_stage_calibration(args.stage2_calibration, args.calibration)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))

    best_by_mode = {mode: None for mode in routing_modes}

    for seed in seeds:
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X=X, y=y, seed=seed, test_size=args.test_size, val_size=args.val_size
        )
        y_train_arr = y_train.to_numpy(dtype=int)
        y_val_arr = y_val.to_numpy(dtype=int)
        y_test_arr = y_test.to_numpy(dtype=int)

        for routing_mode in routing_modes:
            candidate = _train_candidate_for_mode(
                args=args,
                seed=seed,
                routing_mode=routing_mode,
                labels=labels,
                purpose_feature_sets=purpose_feature_sets,
                thresholds_stage1=thresholds_stage1,
                thresholds_stage2=thresholds_stage2,
                thresholds_stage3=thresholds_stage3,
                stage3_threshold_objective=stage3_threshold_objective,
                resolved_stage1_calibration=resolved_stage1_calibration,
                resolved_stage2_calibration=resolved_stage2_calibration,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train_arr=y_train_arr,
                y_val_arr=y_val_arr,
                y_test_arr=y_test_arr,
            )
            if candidate is None:
                continue

            current_best = best_by_mode[routing_mode]
            if current_best is None:
                best_by_mode[routing_mode] = candidate
                continue

            cand_cmp = {
                "seed": candidate["seed"],
                "val_macro_f1": candidate["val_macro_f1"],
                "val_bal_acc": candidate["val_bal_acc"],
                "val_stage3_f1": candidate["val_stage3_f1"],
                "val_accuracy": candidate["val_acc"],
                "val_logloss": candidate["val_logloss"],
            }
            best_cmp = {
                "seed": current_best["seed"],
                "val_macro_f1": current_best["val_macro_f1"],
                "val_bal_acc": current_best["val_bal_acc"],
                "val_stage3_f1": current_best["val_stage3_f1"],
                "val_accuracy": current_best["val_acc"],
                "val_logloss": current_best["val_logloss"],
            }
            if str(args.arch_version) == "v2" and ("routing_score" in candidate) and ("routing_score" in current_best):
                cand_route = (
                    float(candidate["routing_score"]),
                    float(candidate["val_macro_f1"]),
                    float(candidate["val_bal_acc"]),
                    float(candidate["val_stage3_f1"]),
                    float(candidate["val_acc"]),
                    -float(candidate["val_logloss"]),
                )
                best_route = (
                    float(current_best["routing_score"]),
                    float(current_best["val_macro_f1"]),
                    float(current_best["val_bal_acc"]),
                    float(current_best["val_stage3_f1"]),
                    float(current_best["val_acc"]),
                    -float(current_best["val_logloss"]),
                )
                if cand_route > best_route:
                    best_by_mode[routing_mode] = candidate
            elif _is_better_candidate(args, cand_cmp, best_cmp):
                best_by_mode[routing_mode] = candidate

    missing_modes = [m for m, best in best_by_mode.items() if best is None]
    if missing_modes:
        raise RuntimeError(f"No valid model trained for routing modes: {missing_modes}")

    output_root = OUTPUT_DIR / "xgboost_hierarchical"
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    mode_rows = []
    per_mode_results = []
    for routing_mode in routing_modes:
        best = best_by_mode[routing_mode]
        y_test = np.asarray(best["y_test"]).astype(int)
        y_pred = np.asarray(best["test_pred"]).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))

        print(f"\n=== Routing Mode: {routing_mode} ===")
        print(classification_report(y_test, y_pred, labels=list(range(len(labels))), target_names=labels, digits=4, zero_division=0))
        for stage_name, stage_payload in best["stage_reports"]["test"].items():
            if stage_payload.get("classification_report") is None:
                print(f"  {stage_name}: skipped (no routed samples)")
                continue
            s_labels = stage_payload.get("labels", ["neg", "pos"])
            s_rep = stage_payload["classification_report"]
            s_pos = s_labels[1] if len(s_labels) > 1 else "pos"
            s_pos_rep = s_rep.get(s_pos, {})
            print(
                f"  {stage_name}: "
                f"acc={stage_payload.get('accuracy', 0.0):.4f} "
                f"prec={float(s_pos_rep.get('precision', 0.0)):.4f} "
                f"rec={float(s_pos_rep.get('recall', 0.0)):.4f} "
                f"f1={float(s_pos_rep.get('f1-score', 0.0)):.4f} "
                f"support={int(s_pos_rep.get('support', 0))}"
            )

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(f"Confusion Matrix ({routing_mode})")
        plt.xticks(rotation=30)
        plt.tight_layout()

        mode_dir = output_root / routing_mode
        stage_dir = mode_dir / "stage_reports"
        stage_val_dir = stage_dir / "validation"
        stage_test_dir = stage_dir / "test"
        overall_dir = mode_dir / "overall"
        mode_dir.mkdir(parents=True, exist_ok=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_val_dir.mkdir(parents=True, exist_ok=True)
        stage_test_dir.mkdir(parents=True, exist_ok=True)
        overall_dir.mkdir(parents=True, exist_ok=True)
        cm_plot_path = overall_dir / "overall_confusion_matrix_plot.png"
        fig.savefig(cm_plot_path, dpi=150)
        if not args.no_plot:
            plt.show()
        else:
            plt.close(fig)

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        metrics = {
            "mode": f"hierarchical_3stage::{args.mode}::{routing_mode}",
            "dataset_path": str(resolved_input),
            "arch_version": str(args.arch_version),
            "routing_mode": routing_mode,
            "best_seed": int(best["seed"]),
            "selected_seed": int(best["seed"]),
            "best_t1": float(best["best_t1"]),
            "best_t2": float(best["best_t2"]),
            "best_t3": float(best["best_t3"]),
            "best_thresholds": {"t1": float(best["best_t1"]), "t2": float(best["best_t2"]), "t3": float(best["best_t3"])},
            "threshold_search": {
                "threshold_grids": {"stage1": thresholds_stage1, "stage2": thresholds_stage2, "stage3": thresholds_stage3},
                "used_constrained_candidate": bool(best.get("used_constrained_candidate", False)),
                "threshold_search_total_configs": int(best.get("threshold_search_total_configs", 0)),
                "threshold_search_feasible_configs": int(best.get("threshold_search_feasible_configs", 0)),
                "threshold_search_best_any_score": best.get("threshold_search_best_any_score"),
                "threshold_search_best_feasible_score": best.get("threshold_search_best_feasible_score"),
                "threshold_search_summary": best.get("threshold_search_summary", []),
            },
            "selection_metrics": {
                "val_macro_f1": float(best["val_macro_f1"]),
                "val_bal_acc": float(best["val_bal_acc"]),
                "val_acc": float(best["val_acc"]),
                "val_logloss": float(best["val_logloss"]),
                "val_stage3_f1": float(best["val_stage3_f1"]),
            },
            "final_validation_metrics": best.get("selection_validation_metrics"),
            "final_test_metrics": {
                "macro_f1": float(best["test_macro_f1"]),
                "balanced_accuracy": float(best["test_bal_acc"]),
                "accuracy": float(best["test_acc"]),
                "logloss": float(best["test_logloss"]),
            },
            "val_macro_f1": float(best["val_macro_f1"]),
            "val_dist_recall": float(best["val_dist_recall"]),
            "val_withdrawn_recall": float(best["val_withdrawn_recall"]),
            "val_stage3_precision": float(best["val_stage3_precision"]),
            "val_stage3_recall": float(best["val_stage3_recall"]),
            "val_stage3_f1": float(best["val_stage3_f1"]),
            "val_stage3_fbeta": float(best["val_stage3_fbeta"]),
            "val_stage3_metric": best.get("val_stage3_metric", None),
            "routing_score": best.get("routing_score"),
            "collapse_penalty": best.get("collapse_penalty"),
            "distinction_bonus": best.get("distinction_bonus"),
            "min_recall": best.get("min_recall"),
            "v2_specialist_enabled": bool(best.get("v2_specialist_enabled", False)),
            "v2_specialist_threshold": best.get("v2_specialist_threshold"),
            "v2_specialist_val_f1": best.get("v2_specialist_val_f1"),
            "v2_specialist_val_recall": best.get("v2_specialist_val_recall"),
            "v2_specialist_test_f1": best.get("v2_specialist_test_f1"),
            "v2_specialist_test_recall": best.get("v2_specialist_test_recall"),
            "v2_specialist_train_rows": int(best.get("v2_specialist_train_rows", 0)),
            "v2_specialist_val_rows": int(best.get("v2_specialist_val_rows", 0)),
            "v2_specialist_test_rows": int(best.get("v2_specialist_test_rows", 0)),
            "v2_specialist_train_counts": best.get("v2_specialist_train_counts", {0: 0, 1: 0}),
            "v2_specialist_val_counts": best.get("v2_specialist_val_counts", {0: 0, 1: 0}),
            "v2_specialist_test_counts": best.get("v2_specialist_test_counts", {0: 0, 1: 0}),
            "v2_specialist_val_proba_min": best.get("v2_specialist_val_proba_min", 0.0),
            "v2_specialist_val_proba_max": best.get("v2_specialist_val_proba_max", 0.0),
            "v2_specialist_val_proba_mean": best.get("v2_specialist_val_proba_mean", 0.0),
            "v2_specialist_test_proba_min": best.get("v2_specialist_test_proba_min", 0.0),
            "v2_specialist_test_proba_max": best.get("v2_specialist_test_proba_max", 0.0),
            "v2_specialist_test_proba_mean": best.get("v2_specialist_test_proba_mean", 0.0),
            "v2_fusion_enabled": bool(best.get("v2_fusion_enabled", False)),
            "v2_num_pass_predictions_before_fusion_val": int(best.get("v2_num_pass_predictions_before_fusion_val", 0)),
            "v2_num_pass_predictions_before_fusion_test": int(best.get("v2_num_pass_predictions_before_fusion_test", 0)),
            "v2_num_specialist_eligible_val": int(best.get("v2_num_specialist_eligible_val", 0)),
            "v2_num_specialist_eligible_test": int(best.get("v2_num_specialist_eligible_test", 0)),
            "v2_num_pass_to_distinction_flips_val": int(best.get("v2_num_pass_to_distinction_flips_val", 0)),
            "v2_num_pass_to_distinction_flips_test": int(best.get("v2_num_pass_to_distinction_flips_test", 0)),
            "stage3_plus": bool(args.stage3_plus),
            "stage3_threshold_metric": str(args.stage3_threshold_metric),
            "stage3_threshold_objective": best["stage3_threshold_objective"],
            "stage3_objective_score": float(best["stage3_objective_score"]),
            "stage3_calibrated": bool(best["stage3_calibrated"]),
            "stage3_calibration_resolved": best.get("stage3_calibration_resolved", None),
            "stage3_balance_report": best.get("stage3_balance_report", {}),
            "val_bal_acc": float(best["val_bal_acc"]),
            "val_acc": float(best["val_acc"]),
            "val_logloss": best["val_logloss"],
            "test_macro_f1": float(best["test_macro_f1"]),
            "test_bal_acc": float(best["test_bal_acc"]),
            "test_acc": float(best["test_acc"]),
            "test_logloss": best["test_logloss"],
            "val_recall_distinction": float(best.get("val_per_class_recall", {}).get("distinction", 0.0)),
            "val_recall_fail": float(best.get("val_per_class_recall", {}).get("fail", 0.0)),
            "val_recall_pass": float(best.get("val_per_class_recall", {}).get("pass", 0.0)),
            "val_recall_withdrawn": float(best.get("val_per_class_recall", {}).get("withdrawn", 0.0)),
            "test_recall_distinction": float(report_dict.get("distinction", {}).get("recall", 0.0)),
            "test_recall_fail": float(report_dict.get("fail", {}).get("recall", 0.0)),
            "test_recall_pass": float(report_dict.get("pass", {}).get("recall", 0.0)),
            "test_recall_withdrawn": float(report_dict.get("withdrawn", {}).get("recall", 0.0)),
            "labels": labels,
            "label_mapping": label_mapping,
            "num_class": 4,
            "feature_mode": args.mode,
            "feature_counts": feature_counts,
            "stage_feature_columns": {
                "stage1": best["stage1_feature_columns"],
                "stage2": best["stage2_feature_columns"],
                "stage3": best["stage3_feature_columns"],
            },
            "stage_task_labels": {
                "stage1_target": best["stage1_target"],
                "stage2_target": best["stage2_target"],
                "stage3_pos": best["stage3_pos"],
                "stage3_neg": best["stage3_neg"],
            },
            "stage_weight_info": {
                "stage1": best.get("stage1_weight_info"),
                "stage2": best.get("stage2_weight_info"),
                "stage3": best.get("stage3_weight_info"),
            },
            "calibration_info": {
                "global": str(args.calibration),
                "stage1": best.get("stage1_calibration_resolved"),
                "stage2": best.get("stage2_calibration_resolved"),
                "stage3": best.get("stage3_calibration_resolved"),
            },
            "final_retrain_used": False,
            "final_training_strategy": "selection_models_for_deployment",
            "final_stage_iterations": {
                "stage1": int(best.get("stage1_iteration", args.n_estimators)),
                "stage2": int(best.get("stage2_iteration", args.n_estimators)),
                "stage3": int(best.get("stage3_iteration", args.n_estimators)),
            },
            "final_eval_source": "test",
            "params": vars(args),
            "classification_report": report_dict,
            "stage_reports": best["stage_reports"],
            "routed_counts": best.get("routed_counts", {}),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_plot": str(cm_plot_path),
            "prediction_csv_path": str(overall_dir / "final_predictions.csv"),
        }

        model_artifact = {
            "mode": f"hierarchical_3stage::{args.mode}::{routing_mode}",
            "arch_version": str(args.arch_version),
            "labels": labels,
            "label_mapping": label_mapping,
            "routing_mode": routing_mode,
            "stage_task_labels": {
                "stage1_target": best["stage1_target"],
                "stage2_target": best["stage2_target"],
                "stage3_pos": best["stage3_pos"],
                "stage3_neg": best["stage3_neg"],
            },
            "stage1_model": best["stage1_model"],
            "stage2_model": best["stage2_model"],
            "stage3_model": best["stage3_model"],
            "specialist_model": best.get("specialist_model"),
            "stage1_threshold": float(best["best_t1"]),
            "stage2_threshold": float(best["best_t2"]),
            "stage3_threshold": float(best["best_t3"]),
            "v2_specialist_enabled": bool(best.get("v2_specialist_enabled", False)),
            "v2_specialist_model_name": str(args.v2_specialist_model),
            "v2_specialist_threshold": best.get("v2_specialist_threshold"),
            "stage2_weight_info": best["stage2_weight_info"],
            "stage3_weight_info": best["stage3_weight_info"],
            "stage3_balance_report": best["stage3_balance_report"],
            "stage3_plus": bool(args.stage3_plus),
            "stage3_threshold_metric": str(args.stage3_threshold_metric),
            "stage3_threshold_objective": best["stage3_threshold_objective"],
            "stage3_objective_score": float(best["stage3_objective_score"]),
            "stage3_calibrated": bool(best["stage3_calibrated"]),
            "stage3_calibration_resolved": best["stage3_calibration_resolved"],
            "stage1_feature_columns": best["stage1_feature_columns"],
            "stage2_feature_columns": best["stage2_feature_columns"],
            "stage3_feature_columns": best["stage3_feature_columns"],
            "specialist_feature_columns": best.get("specialist_feature_columns", []),
            "feature_mode": args.mode,
            "params": vars(args),
            "best_seed": int(best["seed"]),
        }

        model_path = mode_dir / "model.joblib"
        metrics_path = mode_dir / "metrics.json"
        joblib.dump(model_artifact, model_path)
        save_json(metrics, metrics_path)
        for stage_name, stage_payload in best["stage_reports"]["validation"].items():
            _save_stage_report_artifacts(stage_val_dir, stage_name, stage_payload)
        for stage_name, stage_payload in best["stage_reports"]["test"].items():
            _save_stage_report_artifacts(stage_test_dir, stage_name, stage_payload)
        overall_acc = float(accuracy_score(y_test, y_pred))
        route_stage1_positive = (np.asarray(best["p_test_stage1"], dtype=float) >= float(best["best_t1"])).astype(int)
        route_stage2_positive = (np.asarray(best["p_test_stage2"], dtype=float) >= float(best["best_t2"])).astype(int)
        routed_to_stage3 = ((route_stage1_positive == 0) & (route_stage2_positive == 0)).astype(int)
        _save_overall_report_artifacts(
            overall_dir,
            labels,
            report_dict,
            cm,
            overall_acc,
            y_test,
            y_pred,
            p_stage1=np.asarray(best["p_test_stage1"], dtype=float),
            p_stage2=np.asarray(best["p_test_stage2"], dtype=float),
            p_stage3=np.asarray(best["p_test_stage3"], dtype=float),
            route_stage1_positive=route_stage1_positive,
            route_stage2_positive=route_stage2_positive,
            routed_to_stage3=routed_to_stage3,
            final_t1=float(best["best_t1"]),
            final_t2=float(best["best_t2"]),
            final_t3=float(best["best_t3"]),
            proba4=np.asarray(best["test_proba4"], dtype=float),
        )
        macro = report_dict.get("macro avg", {})
        weighted = report_dict.get("weighted avg", {})
        print(
            f"  overall: acc={overall_acc:.4f} "
            f"macro_prec={float(macro.get('precision', 0.0)):.4f} "
            f"macro_rec={float(macro.get('recall', 0.0)):.4f} "
            f"macro_f1={float(macro.get('f1-score', 0.0)):.4f} "
            f"weighted_f1={float(weighted.get('f1-score', 0.0)):.4f}"
        )
        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved overall confusion matrix plot to: {cm_plot_path}")
        print(f"Saved stage validation reports to: {stage_val_dir}")
        print(f"Saved stage test reports to: {stage_test_dir}")
        print(f"Saved overall reports to: {overall_dir}")

        mode_row = {
            "routing_mode": routing_mode,
            "best_seed": int(best["seed"]),
            "selected_t1": float(best["best_t1"]),
            "selected_t2": float(best["best_t2"]),
            "selected_t3": float(best["best_t3"]),
            "val_macro_f1": float(best["val_macro_f1"]),
            "val_acc": float(best["val_acc"]),
            "val_bal_acc": float(best["val_bal_acc"]),
            "val_stage3_f1": float(best["val_stage3_f1"]),
            "routing_score": float(best["routing_score"]) if best.get("routing_score") is not None else None,
            "collapse_penalty": float(best["collapse_penalty"]) if best.get("collapse_penalty") is not None else None,
            "distinction_bonus": float(best["distinction_bonus"]) if best.get("distinction_bonus") is not None else None,
            "min_recall": float(best["min_recall"]) if best.get("min_recall") is not None else None,
            "v2_specialist_threshold": float(best["v2_specialist_threshold"]) if best.get("v2_specialist_threshold") is not None else None,
            "v2_specialist_val_f1": float(best.get("v2_specialist_val_f1", 0.0)),
            "v2_specialist_test_f1": float(best.get("v2_specialist_test_f1", 0.0)),
            "v2_num_pass_to_distinction_flips_test": int(best.get("v2_num_pass_to_distinction_flips_test", 0)),
            "test_macro_f1": float(best["test_macro_f1"]),
            "test_acc": float(best["test_acc"]),
            "test_bal_acc": float(best["test_bal_acc"]),
            "test_logloss": float(best["test_logloss"]),
            "test_recall_distinction": float(best["report"].get("distinction", {}).get("recall", 0.0)),
            "test_recall_fail": float(best["report"].get("fail", {}).get("recall", 0.0)),
            "test_recall_pass": float(best["report"].get("pass", {}).get("recall", 0.0)),
            "test_recall_withdrawn": float(best["report"].get("withdrawn", {}).get("recall", 0.0)),
        }
        mode_rows.append(mode_row)
        per_mode_results.append({**mode_row, "confusion_matrix": cm.tolist()})

    summary_df = pd.DataFrame(mode_rows)
    if str(args.arch_version) == "v2" and "routing_score" in summary_df.columns:
        summary_df = summary_df.sort_values(by=["routing_score", "test_macro_f1", "test_acc"], ascending=[False, False, False]).reset_index(drop=True)
    else:
        summary_df = summary_df.sort_values(by=["test_macro_f1", "test_acc"], ascending=[False, False]).reset_index(drop=True)
    print("\nRouting Mode Comparison")
    print(summary_df.to_string(index=False))

    summary_csv = summary_dir / "mode_comparison.csv"
    summary_txt = summary_dir / "mode_comparison.txt"
    summary_df.to_csv(summary_csv, index=False)
    summary_txt.write_text(summary_df.to_string(index=False) + "\n", encoding="utf-8")

    best_mode_row = select_best_mode_row(summary_df)
    best_mode_name = str(best_mode_row["routing_mode"])
    best_routing_score = None
    if str(args.arch_version) == "v2" and "routing_score" in summary_df.columns:
        best_routing_row = summary_df.sort_values(by=["routing_score", "test_macro_f1", "test_acc"], ascending=[False, False, False]).iloc[0]
        best_routing_score = float(best_routing_row["routing_score"])
    summary_confusion_paths = save_summary_confusion_matrices(summary_dir, per_mode_results, labels, best_mode_name)
    summary_report = save_summary_text_report(summary_dir, summary_df, best_mode_row)
    v2_summary_report = save_v2_summary_report(
        summary_dir=summary_dir,
        summary_df=summary_df,
        best_mode_row=best_mode_row,
        arch_version=str(args.arch_version),
    )

    print_end_summary(resolved_input, routing_modes, output_root, summary_df)
    print(f"Architecture version: {args.arch_version}")
    if best_routing_score is not None:
        print(f"Best routing_score: {best_routing_score:.4f}")
    print(f"Saved routing summary to: {summary_csv}")
    print(f"Saved routing summary to: {summary_txt}")
    print(f"Saved summary report to: {summary_report}")
    if v2_summary_report is not None:
        print(f"Saved V2 summary report to: {v2_summary_report}")
    print(f"Saved summary confusion matrices to: {summary_confusion_paths['summary_confusion_json']}")

    return {
        "arch_version": str(args.arch_version),
        "resolved_input": resolved_input,
        "output_root": output_root,
        "summary_dir": summary_dir,
        "summary_csv": summary_csv,
        "summary_txt": summary_txt,
        "summary_report": summary_report,
        "v2_summary_report": v2_summary_report,
        "summary_confusion_json": summary_dir / "summary_confusion_matrices.json",
        "best_mode_name": best_mode_name,
        "best_routing_score": best_routing_score,
        "best_mode_metrics": best_mode_row.to_dict(),
        "summary_df": summary_df,
    }
