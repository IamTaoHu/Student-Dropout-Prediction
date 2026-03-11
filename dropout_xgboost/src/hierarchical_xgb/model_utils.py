from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from hierarchical_xgb.config import ROUTING_MODES_SUPPORTED

try:
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None


def split_train_val_test(X, y, seed, test_size, val_size):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed), stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=float(val_size), random_state=int(seed), stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def parse_threshold_grid(raw: str, arg_name: str) -> list[float]:
    values_raw = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not values_raw:
        raise ValueError(f"{arg_name} cannot be empty.")
    vals = [float(x) for x in values_raw]
    if not vals:
        raise ValueError(f"{arg_name} cannot be empty.")
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"All thresholds in {arg_name} must be within [0,1].")
    return sorted(set(float(v) for v in vals))


def validate_binary_labels(y_bin, stage_name: str) -> bool:
    arr = np.asarray(y_bin, dtype=int)
    uniq = np.unique(arr)
    if len(uniq) < 2:
        raise ValueError(f"{stage_name} missing positive/negative class.")
    return True


def parse_routing_modes(raw: str) -> list[str]:
    modes = [m.strip() for m in str(raw).split(",") if m.strip()]
    if not modes:
        raise ValueError("--routing_modes cannot be empty.")
    invalid = [m for m in modes if m not in ROUTING_MODES_SUPPORTED]
    if invalid:
        raise ValueError(f"Unsupported routing modes: {invalid}. Supported: {ROUTING_MODES_SUPPORTED}")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def build_stage1_model(args, seed, n_estimators_override: int | None = None):
    n_estimators = int(n_estimators_override) if n_estimators_override is not None else int(args.n_estimators)
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=n_estimators,
        learning_rate=args.learning_rate, max_depth=args.max_depth, min_child_weight=args.min_child_weight,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree, reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha, gamma=args.gamma, max_delta_step=args.max_delta_step,
        tree_method="hist", n_jobs=args.n_jobs, random_state=seed, verbosity=args.verbosity,
    )


def build_stage2_model(args, seed, n_estimators_override: int | None = None):
    return build_stage1_model(args, seed, n_estimators_override=n_estimators_override)


def build_stage3_model(args, seed, n_estimators_override: int | None = None):
    if str(args.stage3_model) == "logreg":
        return LogisticRegression(
            C=float(args.stage3_logreg_c),
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
    n_estimators = int(n_estimators_override) if n_estimators_override is not None else int(args.n_estimators)
    if bool(args.stage3_plus):
        stage3_max_depth = int(min(int(args.max_depth), int(args.stage3_xgb_max_depth)))
        stage3_min_child_weight = float(max(float(args.min_child_weight), float(args.stage3_xgb_min_child_weight)))
        stage3_reg_lambda = float(max(float(args.reg_lambda), float(args.stage3_xgb_reg_lambda)))
        stage3_reg_alpha = float(max(float(args.reg_alpha), float(args.stage3_xgb_reg_alpha)))
        stage3_gamma = float(max(float(args.gamma), float(args.stage3_xgb_gamma)))
        stage3_max_delta_step = float(max(float(args.max_delta_step), float(args.stage3_xgb_max_delta_step)))
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=args.learning_rate,
            max_depth=stage3_max_depth,
            min_child_weight=stage3_min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=stage3_reg_lambda,
            reg_alpha=stage3_reg_alpha,
            gamma=stage3_gamma,
            max_delta_step=stage3_max_delta_step,
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=seed,
            verbosity=args.verbosity,
        )
    return build_stage1_model(args, seed, n_estimators_override=n_estimators_override)


def build_specialist_model(args, seed, model_name: str, n_estimators_override: int | None = None):
    model_name_l = str(model_name).strip().lower()
    if model_name_l == "logreg":
        return LogisticRegression(
            C=float(args.stage3_logreg_c),
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
    if model_name_l == "xgb":
        n_estimators = int(n_estimators_override) if n_estimators_override is not None else int(args.n_estimators)
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=args.learning_rate,
            max_depth=int(min(int(args.max_depth), 4)),
            min_child_weight=float(max(float(args.min_child_weight), 4.0)),
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=float(max(float(args.reg_lambda), 4.0)),
            reg_alpha=float(max(float(args.reg_alpha), 0.2)),
            gamma=float(max(float(args.gamma), 0.1)),
            max_delta_step=float(max(float(args.max_delta_step), 1.0)),
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=seed,
            verbosity=args.verbosity,
        )
    raise ValueError(f"Unsupported specialist model: {model_name}")


def fit_compat(model, X_train, y_train, sample_weight=None, X_val=None, y_val=None):
    try:
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    except TypeError:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_train, y_train)
    return model


def make_stage1_binary(y_int, distinction_id=0):
    return (y_int == distinction_id).astype(int)


def compute_neg_pos_ratio(y_bin):
    pos = int(np.sum(y_bin == 1))
    neg = int(np.sum(y_bin == 0))
    return 1.0 if pos == 0 or neg == 0 else neg / max(pos, 1)


def make_stage1_sample_weight(y_bin, pos_multiplier):
    base = compute_neg_pos_ratio(y_bin)
    final = base * float(pos_multiplier)
    w = np.ones_like(y_bin, dtype=float)
    w[y_bin == 1] = final
    return w, float(final)


def make_stage1_sample_weight_mode(y_bin: np.ndarray, mode: str, pos_multiplier: float) -> tuple[np.ndarray, dict]:
    mode_l = str(mode).strip().lower()
    y_arr = np.asarray(y_bin, dtype=int)
    if mode_l == "legacy":
        w, pos_w = make_stage1_sample_weight(y_arr, pos_multiplier)
        return w, {"mode": mode_l, "pos_weight": float(pos_w), "neg_weight": 1.0}
    return make_stage3_sample_weight(y_arr, mode=mode_l, pos_multiplier=pos_multiplier)


def make_binary_for_label(y_int, labels, positive_label):
    return (y_int == labels.index(positive_label)).astype(int)


def make_binary_for_two_labels(y_int, pos_id: int, neg_id: int) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_int, dtype=int)
    mask = np.isin(y_arr, [int(pos_id), int(neg_id)])
    y_bin = (y_arr[mask] == int(pos_id)).astype(int)
    return mask, y_bin


def make_sample_weight_binary(y_bin, pos_multiplier):
    return make_stage1_sample_weight(y_bin, pos_multiplier)


def value_counts_binary(y_bin) -> dict:
    y_arr = np.asarray(y_bin, dtype=int)
    return {
        0: int(np.sum(y_arr == 0)),
        1: int(np.sum(y_arr == 1)),
    }


def _resolve_majority_minority_labels(y_bin: np.ndarray) -> tuple[int | None, int | None]:
    counts = value_counts_binary(y_bin)
    if counts[0] == 0 and counts[1] == 0:
        return None, None
    if counts[0] >= counts[1]:
        return 0, 1
    return 1, 0


def undersample_binary_majority(X_df, y_bin, target_ratio=1.0, random_state=42):
    X_in = pd.DataFrame(X_df).copy()
    y_arr = np.asarray(y_bin, dtype=int)
    before = value_counts_binary(y_arr)
    maj_label, min_label = _resolve_majority_minority_labels(y_arr)
    report = {
        "before_counts": before,
        "after_counts": before.copy(),
        "majority_label": maj_label,
        "minority_label": min_label,
        "strategy": "undersample_majority",
        "applied": False,
    }
    if maj_label is None or min_label is None or before[min_label] == 0 or before[maj_label] == 0:
        return X_in.reset_index(drop=True), y_arr.copy(), report

    ratio = max(float(target_ratio), 1e-9)
    target_majority = int(np.ceil(before[min_label] * ratio))
    target_majority = max(1, min(target_majority, before[maj_label]))
    if target_majority >= before[maj_label]:
        return X_in.reset_index(drop=True), y_arr.copy(), report

    rng = np.random.RandomState(int(random_state))
    idx_maj = np.where(y_arr == maj_label)[0]
    idx_min = np.where(y_arr == min_label)[0]
    keep_maj = rng.choice(idx_maj, size=target_majority, replace=False)
    idx_keep = np.concatenate([keep_maj, idx_min])
    idx_keep = idx_keep[rng.permutation(len(idx_keep))]

    X_res = X_in.iloc[idx_keep].reset_index(drop=True)
    y_res = y_arr[idx_keep]
    report["after_counts"] = value_counts_binary(y_res)
    report["applied"] = True
    return X_res, y_res, report


def oversample_binary_minority(X_df, y_bin, target_ratio=1.0, random_state=42):
    X_in = pd.DataFrame(X_df).copy()
    y_arr = np.asarray(y_bin, dtype=int)
    before = value_counts_binary(y_arr)
    maj_label, min_label = _resolve_majority_minority_labels(y_arr)
    report = {
        "before_counts": before,
        "after_counts": before.copy(),
        "majority_label": maj_label,
        "minority_label": min_label,
        "strategy": "oversample_minority",
        "applied": False,
    }
    if maj_label is None or min_label is None or before[min_label] == 0 or before[maj_label] == 0:
        return X_in.reset_index(drop=True), y_arr.copy(), report

    ratio = float(target_ratio)
    if ratio < 1.0:
        target_minority = int(np.ceil(before[maj_label] / max(ratio, 1e-9)))
    else:
        target_minority = int(before[maj_label])
    target_minority = max(before[min_label], target_minority)
    if target_minority <= before[min_label]:
        return X_in.reset_index(drop=True), y_arr.copy(), report

    rng = np.random.RandomState(int(random_state))
    idx_maj = np.where(y_arr == maj_label)[0]
    idx_min = np.where(y_arr == min_label)[0]
    extra_n = int(target_minority - len(idx_min))
    extra_idx = rng.choice(idx_min, size=extra_n, replace=True)
    idx_all = np.concatenate([idx_maj, idx_min, extra_idx])
    idx_all = idx_all[rng.permutation(len(idx_all))]

    X_res = X_in.iloc[idx_all].reset_index(drop=True)
    y_res = y_arr[idx_all]
    report["after_counts"] = value_counts_binary(y_res)
    report["applied"] = True
    return X_res, y_res, report


def hybrid_balance_binary(X_df, y_bin, ratio=1.0, random_state=42):
    X_in = pd.DataFrame(X_df).copy()
    y_arr = np.asarray(y_bin, dtype=int)
    before = value_counts_binary(y_arr)
    maj_label, min_label = _resolve_majority_minority_labels(y_arr)
    report = {
        "before_counts": before,
        "after_counts": before.copy(),
        "majority_label": maj_label,
        "minority_label": min_label,
        "strategy": "hybrid",
        "applied": False,
    }
    if maj_label is None or min_label is None or before[min_label] == 0 or before[maj_label] == 0:
        return X_in.reset_index(drop=True), y_arr.copy(), report

    first_ratio = max(1.2, float(ratio))
    X_mid, y_mid, rep_under = undersample_binary_majority(
        X_in, y_arr, target_ratio=first_ratio, random_state=int(random_state)
    )
    X_res, y_res, rep_over = oversample_binary_minority(
        X_mid, y_mid, target_ratio=1.0, random_state=int(random_state) + 1
    )
    report["after_counts"] = value_counts_binary(y_res)
    report["applied"] = bool(rep_under.get("applied", False) or rep_over.get("applied", False))
    return X_res, y_res, report


def make_stage3_sample_weight(y_bin: np.ndarray, mode: str, pos_multiplier: float) -> tuple[np.ndarray, dict]:
    mode = str(mode).strip().lower()
    y_arr = np.asarray(y_bin, dtype=int)
    pos = int(np.sum(y_arr == 1))
    neg = int(np.sum(y_arr == 0))
    if pos == 0 or neg == 0:
        return np.ones_like(y_arr, dtype=float), {"mode": mode, "pos_weight": 1.0, "neg_weight": 1.0}

    if mode == "legacy":
        w, final_pos = make_sample_weight_binary(y_arr, pos_multiplier)
        return w, {"mode": mode, "pos_weight": float(final_pos), "neg_weight": 1.0}
    if mode == "none":
        return np.ones_like(y_arr, dtype=float), {"mode": mode, "pos_weight": 1.0, "neg_weight": 1.0}

    n = float(len(y_arr))
    inv_pos = n / (2.0 * pos)
    inv_neg = n / (2.0 * neg)
    if mode == "sqrt_inv":
        inv_pos = float(np.sqrt(inv_pos))
        inv_neg = float(np.sqrt(inv_neg))

    pos_w = float(inv_pos * float(pos_multiplier))
    neg_w = float(inv_neg)
    w = np.where(y_arr == 1, pos_w, neg_w).astype(float)
    return w, {"mode": mode, "pos_weight": pos_w, "neg_weight": neg_w}


def make_specialist_sample_weight(y_bin: np.ndarray, pos_multiplier: float = 1.5) -> tuple[np.ndarray, dict]:
    y_arr = np.asarray(y_bin, dtype=int)
    pos = int(np.sum(y_arr == 1))
    neg = int(np.sum(y_arr == 0))
    if pos == 0 or neg == 0:
        return np.ones_like(y_arr, dtype=float), {"pos_weight": 1.0, "neg_weight": 1.0}

    n = float(len(y_arr))
    pos_w = float((n / (2.0 * pos)) * float(pos_multiplier))
    neg_w = float(n / (2.0 * neg))
    weights = np.where(y_arr == 1, pos_w, neg_w).astype(float)
    return weights, {"pos_weight": pos_w, "neg_weight": neg_w}


def apply_stage3_fail_weight(
    weights: np.ndarray,
    y_stage3_bin: np.ndarray,
    fail_weight: float,
    stage3_pos_label: str,
    stage3_neg_label: str,
) -> tuple[np.ndarray, dict]:
    w = np.asarray(weights, dtype=float).copy()
    yb = np.asarray(y_stage3_bin, dtype=int)
    fw = float(fail_weight)
    if fw <= 0:
        raise ValueError("--stage3-fail-weight must be > 0.")

    applied_to = "none"
    if stage3_pos_label == "fail":
        w[yb == 1] *= fw
        applied_to = "pos"
    elif stage3_neg_label == "fail":
        w[yb == 0] *= fw
        applied_to = "neg"

    return w, {"fail_weight": fw, "applied_to": applied_to}


def resolve_stage_calibration(stage_mode: str, global_mode: str) -> str:
    stage_m = str(stage_mode).strip().lower()
    if stage_m == "auto":
        return str(global_mode).strip().lower()
    return stage_m


def get_model_best_iteration(model, fallback_n_estimators: int) -> int:
    best_iter = None
    if hasattr(model, "best_iteration"):
        best_iter = getattr(model, "best_iteration", None)
    if best_iter is None and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            best_iter = getattr(booster, "best_iteration", None)
        except Exception:
            best_iter = None
    if best_iter is not None and int(best_iter) > 0:
        return int(best_iter)
    if hasattr(model, "n_estimators"):
        try:
            n_est = int(getattr(model, "n_estimators"))
            if n_est > 0:
                return n_est
        except Exception:
            pass
    return int(fallback_n_estimators)


def calibrate_prefit(model, X_calib, y_calib, method):
    if method == "none":
        return model
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method)
        cal.fit(X_calib, y_calib)
        return cal
    cal = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    cal.fit(X_calib, y_calib)
    return cal


def resolve_stage3_calibration(args) -> str:
    stage3_mode = str(args.stage3_calibration).strip().lower()
    if stage3_mode == "auto":
        return str(args.calibration).strip().lower()
    return stage3_mode


def _predict_proba_binary(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
