from __future__ import annotations

import re

import numpy as np
import pandas as pd

from hierarchical_xgb.config import (
    LOCKED_42_FEATURES,
    STAGE1_FEATURES_42,
    STAGE2_FEATURES_42,
    STAGE3_FEATURES_42,
)
from hierarchical_xgb.data_utils import _is_feature_identifier


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(float(default)).astype(float)
    return pd.Series(float(default), index=df.index, dtype=float)


def _safe_divide_series(num, den, default: float = 0.0) -> pd.Series:
    if isinstance(num, pd.Series):
        num_s = num.copy()
    elif isinstance(den, pd.Series):
        num_s = pd.Series(num, index=den.index)
    else:
        num_s = pd.Series(num)
    if isinstance(den, pd.Series):
        den_s = den.copy()
    elif isinstance(num, pd.Series):
        den_s = pd.Series(den, index=num.index)
    else:
        den_s = pd.Series(den)
    num_s = pd.to_numeric(num_s, errors="coerce").astype(float)
    den_s = pd.to_numeric(den_s, errors="coerce").astype(float)
    out = num_s.divide(den_s.replace(0.0, np.nan))
    out = out.replace([np.inf, -np.inf], np.nan).fillna(float(default))
    return out.astype(float)


def _clip_nonnegative(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)


def _safe_log1p(series) -> pd.Series:
    values = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0.0).astype(float)
    return np.log1p(values.clip(lower=0.0)).astype(float)


def _cleanup_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    return X


def _infer_assessment_score_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "kuz_score_mean",
        "kuz_score_std",
        "kuz_score_min",
        "kuz_score_max",
        "kuz_score_skew_proxy",
        "kuz_score_iqr_proxy",
        "kuz_score_per_assess",
        "kuz_score_per_weight_sum",
        "kuz_score_per_module",
        "kuz_score_cv",
        "kuz_score_range",
        "score_std",
        "score_range",
        "score_density",
        "score_weight_density",
    }
    score_cols: list[str] = []
    for col in df.select_dtypes(include=[np.number]).columns.tolist():
        col_l = str(col).strip().lower()
        if col in excluded:
            continue
        if "score" not in col_l:
            continue
        if col_l.startswith("kuz_"):
            continue
        score_cols.append(col)
    return score_cols


def _infer_assessment_weight_columns(df: pd.DataFrame) -> list[str]:
    weight_cols: list[str] = []
    for col in df.select_dtypes(include=[np.number]).columns.tolist():
        col_l = str(col).strip().lower()
        if "weight" not in col_l:
            continue
        if col_l in {"kuz_weight_mean", "kuz_weight_sum"}:
            continue
        weight_cols.append(col)
    return weight_cols


def _ordered_assessment_score_columns(df: pd.DataFrame) -> list[str]:
    score_cols = _infer_assessment_score_columns(df)
    if not score_cols:
        return []

    def _sort_key(col: str) -> tuple[int, int, str]:
        col_l = str(col).strip().lower()
        digits = re.findall(r"\d+", col_l)
        idx = int(digits[-1]) if digits else 10**6
        exam_bias = 1 if "exam" in col_l else 0
        return (idx, exam_bias, col_l)

    return sorted(score_cols, key=_sort_key)


def _longest_true_streak(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for item in np.asarray(mask, dtype=bool).tolist():
        if item:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def _first_long_gap_index(completed_mask: np.ndarray, min_gap: int = 2) -> int:
    run = 0
    for i, is_completed in enumerate(np.asarray(completed_mask, dtype=bool).tolist()):
        if not is_completed:
            run += 1
            if run >= int(min_gap):
                # 1-based index for easier interpretation in reports.
                return int(i - run + 2)
        else:
            run = 0
    return 0


def _add_score_band_count_features(X: pd.DataFrame, score_cols: list[str]) -> list[str]:
    if not score_cols:
        return []
    scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores)
    denom = np.maximum(valid.sum(axis=1), 1)
    num_high = ((scores >= 70.0) & valid).sum(axis=1).astype(float)
    num_mid = ((scores >= 40.0) & (scores < 70.0) & valid).sum(axis=1).astype(float)
    num_low = ((scores < 40.0) & valid).sum(axis=1).astype(float)
    num_zero = ((scores == 0.0) & valid).sum(axis=1).astype(float)
    X["num_high_scores"] = num_high
    X["num_mid_scores"] = num_mid
    X["num_low_scores"] = num_low
    X["num_zero_scores"] = num_zero
    X["high_score_ratio"] = num_high / denom
    X["low_score_ratio"] = num_low / denom
    return [
        "num_high_scores",
        "num_mid_scores",
        "num_low_scores",
        "num_zero_scores",
        "high_score_ratio",
        "low_score_ratio",
    ]


def _add_trend_features(X: pd.DataFrame, score_cols: list[str]) -> list[str]:
    if not score_cols:
        return []
    scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_cols = scores.shape[1]
    split_idx = max(1, n_cols // 2)
    early_scores = scores[:, :split_idx]
    late_scores = scores[:, split_idx:] if split_idx < n_cols else scores[:, :split_idx]
    recent_window = scores[:, max(0, n_cols - min(3, n_cols)) :]
    early_mean = np.nanmean(early_scores, axis=1)
    late_mean = np.nanmean(late_scores, axis=1)
    weights = np.linspace(1.0, 2.0, recent_window.shape[1], dtype=float)
    weighted_recent = np.divide(
        np.nansum(recent_window * weights, axis=1),
        np.maximum(np.nansum(np.isfinite(recent_window) * weights, axis=1), 1e-6),
    )
    overall_mean = np.nanmean(scores, axis=1)
    early_mean = np.where(np.isfinite(early_mean), early_mean, 0.0)
    late_mean = np.where(np.isfinite(late_mean), late_mean, 0.0)
    weighted_recent = np.where(np.isfinite(weighted_recent), weighted_recent, 0.0)
    overall_mean = np.where(np.isfinite(overall_mean), overall_mean, 0.0)
    X["early_mean_score"] = early_mean
    X["late_mean_score"] = late_mean
    X["score_trend_delta"] = late_mean - early_mean
    X["score_trend"] = X["score_trend_delta"]
    X["recent_weighted_score_mean"] = weighted_recent
    X["recent_weighted_mean"] = X["recent_weighted_score_mean"]
    X["recent_vs_overall_gap"] = weighted_recent - overall_mean
    recent_max = np.nanmax(recent_window, axis=1)
    recent_min = np.nanmin(recent_window, axis=1)
    X["recent_max_score"] = np.where(np.isfinite(recent_max), recent_max, 0.0)
    X["recent_min_score"] = np.where(np.isfinite(recent_min), recent_min, 0.0)
    return [
        "early_mean_score",
        "late_mean_score",
        "score_trend_delta",
        "score_trend",
        "recent_weighted_score_mean",
        "recent_weighted_mean",
        "recent_vs_overall_gap",
        "recent_max_score",
        "recent_min_score",
    ]


def _add_stability_spread_features(X: pd.DataFrame, score_cols: list[str]) -> list[str]:
    created: list[str] = []
    if score_cols:
        scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        score_max = np.nanmax(scores, axis=1)
        score_min = np.nanmin(scores, axis=1)
        score_mean = np.nanmean(scores, axis=1)
        score_std = np.nanstd(scores, axis=1)
        score_var = np.nanvar(scores, axis=1)
        score_max = np.where(np.isfinite(score_max), score_max, 0.0)
        score_min = np.where(np.isfinite(score_min), score_min, 0.0)
        score_mean = np.where(np.isfinite(score_mean), score_mean, 0.0)
        score_std = np.where(np.isfinite(score_std), score_std, 0.0)
        score_var = np.where(np.isfinite(score_var), score_var, 0.0)
        X["score_range"] = score_max - score_min
        X["score_variance"] = score_var
        X["coeff_var_score"] = _safe_divide_series(score_std, np.clip(np.abs(score_mean), 1e-6, None))
        X["consistency_index"] = _safe_divide_series(1.0, 1.0 + score_std)
        X["high_low_gap"] = score_max - score_min
        created.extend(["score_range", "score_variance", "coeff_var_score", "consistency_index", "high_low_gap"])
    else:
        score_range = _safe_series(X, "kuz_score_max") - _safe_series(X, "kuz_score_min")
        score_std = np.abs(_safe_series(X, "kuz_score_std"))
        score_mean = _safe_series(X, "kuz_score_mean")
        X["score_variance"] = score_std.pow(2)
        X["score_range"] = score_range
        X["coeff_var_score"] = _safe_divide_series(score_std, np.abs(score_mean).clip(lower=1e-6))
        X["consistency_index"] = _safe_divide_series(1.0, 1.0 + score_std)
        X["high_low_gap"] = score_range
        created.extend(["score_range", "score_variance", "coeff_var_score", "consistency_index", "high_low_gap"])
    return created


def _add_assessment_composition_features(X: pd.DataFrame, score_cols: list[str], weight_cols: list[str]) -> list[str]:
    created: list[str] = []
    assessed_count = _clip_nonnegative(_safe_series(X, "kuz_assess_count"))
    X["assessed_count"] = assessed_count
    created.append("assessed_count")

    expected_count = assessed_count.copy()
    if score_cols:
        expected_count = pd.Series(float(len(score_cols)), index=X.index, dtype=float).clip(lower=1.0)
        missing_count = expected_count - _clip_nonnegative(pd.Series(np.isfinite(X[score_cols].to_numpy(dtype=float)).sum(axis=1), index=X.index))
    else:
        type_total = _clip_nonnegative(_safe_series(X, "kuz_type_cma")) + _clip_nonnegative(_safe_series(X, "kuz_type_exam")) + _clip_nonnegative(_safe_series(X, "kuz_type_tma"))
        expected_count = pd.concat([assessed_count, type_total], axis=1).max(axis=1).clip(lower=1.0)
        missing_count = (expected_count - assessed_count).clip(lower=0.0)

    X["missing_assessment_count"] = _clip_nonnegative(missing_count)
    X["completion_ratio"] = _safe_divide_series(assessed_count, expected_count, default=0.0).clip(lower=0.0, upper=1.0)
    X["assessment_completion_ratio"] = X["completion_ratio"]
    X["exam_vs_tma_gap"] = _safe_series(X, "kuz_exam_ratio") - _safe_series(X, "kuz_tma_ratio")
    created.extend(["missing_assessment_count", "completion_ratio", "assessment_completion_ratio", "exam_vs_tma_gap"])

    if weight_cols and score_cols and len(weight_cols) == len(score_cols):
        weights = X[weight_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        weight_sum = np.maximum(np.nansum(np.where(np.isfinite(weights), weights, 0.0), axis=1), 1e-6)
        weighted_mean = np.nansum(np.where(np.isfinite(scores), scores, 0.0) * np.where(np.isfinite(weights), weights, 0.0), axis=1) / weight_sum
        X["exam_weighted_mean_if_available"] = np.where(np.isfinite(weighted_mean), weighted_mean, 0.0)
        X["tma_weighted_mean_if_available"] = np.where(np.isfinite(weighted_mean), weighted_mean, 0.0)
    else:
        overall_mean = _safe_series(X, "kuz_score_mean")
        X["exam_weighted_mean_if_available"] = overall_mean * _safe_series(X, "kuz_exam_ratio")
        X["tma_weighted_mean_if_available"] = overall_mean * _safe_series(X, "kuz_tma_ratio")
    created.extend(["exam_weighted_mean_if_available", "tma_weighted_mean_if_available"])
    return created


def _add_interaction_features(X: pd.DataFrame) -> list[str]:
    X["mean_x_completion"] = _safe_series(X, "kuz_score_mean") * _safe_series(X, "completion_ratio")
    X["recent_x_stability"] = _safe_series(X, "recent_weighted_score_mean") * _safe_series(X, "consistency_index", default=1.0)
    X["high_score_ratio_x_exam_ratio"] = _safe_series(X, "high_score_ratio") * _safe_series(X, "kuz_exam_ratio")
    X["low_score_ratio_x_missing_count"] = _safe_series(X, "low_score_ratio") * _safe_series(X, "missing_assessment_count")
    return [
        "mean_x_completion",
        "recent_x_stability",
        "high_score_ratio_x_exam_ratio",
        "low_score_ratio_x_missing_count",
    ]


def add_flat_feature_families(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    before_cols = set(X.columns)
    score_cols = _infer_assessment_score_columns(X)
    weight_cols = _infer_assessment_weight_columns(X)
    created: list[str] = []
    created.extend(_add_score_band_count_features(X, score_cols))
    created.extend(_add_trend_features(X, score_cols))
    created.extend(_add_stability_spread_features(X, score_cols))
    created.extend(_add_assessment_composition_features(X, score_cols, weight_cols))
    created.extend(_add_interaction_features(X))
    X = _cleanup_feature_frame(X)
    new_cols = [col for col in X.columns if col not in before_cols]
    X.attrs["flat_feature_family_added"] = sorted(set(new_cols))
    X.attrs["flat_feature_family_count"] = int(len(new_cols))
    X.attrs["flat_feature_family_requested"] = sorted(set(created))
    return X


def _binary_entropy_from_three_counts(a, b, c) -> pd.Series:
    counts = pd.concat(
        [
            pd.to_numeric(pd.Series(a), errors="coerce").fillna(0.0).clip(lower=0.0),
            pd.to_numeric(pd.Series(b), errors="coerce").fillna(0.0).clip(lower=0.0),
            pd.to_numeric(pd.Series(c), errors="coerce").fillna(0.0).clip(lower=0.0),
        ],
        axis=1,
    ).astype(float)
    total = counts.sum(axis=1).replace(0.0, np.nan)
    probs = counts.divide(total, axis=0)
    probs = probs.where(probs > 0.0, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    return entropy.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def normalize_kuzilek_base_columns(X_df: pd.DataFrame) -> pd.DataFrame:
    X = X_df.copy()
    X.columns = [str(c).replace("\ufeff", "").strip() for c in X.columns]

    alias_map = {
        "kuz_type_CMA": "kuz_type_cma",
        "kuz_type_Exam": "kuz_type_exam",
        "kuz_type_TMA": "kuz_type_tma",
    }
    alias_created: list[str] = []
    for raw_col, canonical_col in alias_map.items():
        if raw_col in X.columns and canonical_col not in X.columns:
            X[canonical_col] = X[raw_col]
            alias_created.append(f"{raw_col}->{canonical_col}")

    numeric_pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"
    for col in X.columns:
        if _is_feature_identifier(col):
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        non_null = X[col].dropna()
        if non_null.empty:
            continue
        as_str = non_null.astype(str).str.strip()
        if bool(as_str.str.fullmatch(numeric_pattern).all()):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X.attrs["kuz_aliases_created"] = alias_created
    return X


def build_locked42_from_base_kuzilek(X_df: pd.DataFrame) -> pd.DataFrame:
    """Build the locked feature set from raw Kuzilek aggregate columns when needed."""
    X = normalize_kuzilek_base_columns(X_df.copy())
    alias_created = list(X.attrs.get("kuz_aliases_created", []))

    assess_count = _clip_nonnegative(_safe_series(X, "kuz_assess_count"))
    weight_mean = _safe_series(X, "kuz_weight_mean")
    weight_sum = _clip_nonnegative(_safe_series(X, "kuz_weight_sum"))
    _date_min = X.get("kuz_date_min")
    _date_max = X.get("kuz_date_max")
    modules_nunique = _clip_nonnegative(_safe_series(X, "kuz_modules_nunique"))
    score_mean = _safe_series(X, "kuz_score_mean")
    score_std = _safe_series(X, "kuz_score_std")
    score_min = _safe_series(X, "kuz_score_min")
    score_max = _safe_series(X, "kuz_score_max")
    type_cma = _clip_nonnegative(_safe_series(X, "kuz_type_cma"))
    type_exam = _clip_nonnegative(_safe_series(X, "kuz_type_exam"))
    type_tma = _clip_nonnegative(_safe_series(X, "kuz_type_tma"))

    raw_passthrough = {
        "kuz_weight_mean": weight_mean,
        "kuz_weight_sum": weight_sum,
        "kuz_modules_nunique": modules_nunique,
        "kuz_score_mean": score_mean,
        "kuz_score_std": score_std,
        "kuz_score_min": score_min,
        "kuz_score_max": score_max,
        "kuz_type_cma": type_cma,
        "kuz_type_exam": type_exam,
        "kuz_type_tma": type_tma,
    }
    for col, values in raw_passthrough.items():
        if col not in X.columns:
            X[col] = values

    type_total = type_cma + type_exam + type_tma
    score_range = score_max - score_min
    score_delta = score_mean - 40.0
    abs_score_std = np.abs(score_std)
    abs_score_max = np.abs(score_max)
    abs_score_mean = np.abs(score_mean)
    exam_ratio = _safe_divide_series(type_exam, type_total)
    tma_ratio = _safe_divide_series(type_tma, type_total)
    cma_ratio = _safe_divide_series(type_cma, type_total)
    type_entropy = _binary_entropy_from_three_counts(type_cma, type_exam, type_tma)
    type_dom_ratio = pd.concat([exam_ratio, tma_ratio, cma_ratio], axis=1).max(axis=1).astype(float)

    derived = {
        "kuz_score_per_assess": _safe_divide_series(score_mean, assess_count.replace(0.0, 1.0)),
        "kuz_score_per_weight_sum": _safe_divide_series(score_mean, weight_sum.clip(lower=1e-6)),
        "kuz_score_per_module": _safe_divide_series(score_mean, modules_nunique.replace(0.0, 1.0)),
        "kuz_type_total": type_total,
        "kuz_exam_ratio": exam_ratio,
        "kuz_tma_ratio": tma_ratio,
        "kuz_cma_ratio": cma_ratio,
        "kuz_score_skew_proxy": score_mean - ((score_min + score_max) / 2.0),
        "kuz_score_iqr_proxy": score_range,
        "kuz_type_entropy": type_entropy,
        "kuz_exam_vs_tma": exam_ratio - tma_ratio,
        "kuz_tma_vs_cma": tma_ratio - cma_ratio,
        "log1p_kuz_weight_sum": np.log1p(_clip_nonnegative(weight_sum)),
        "v3_mean_above_min": score_mean - score_min,
        "v3_headroom_ratio": _safe_divide_series(score_max - score_mean, abs_score_max.clip(lower=1.0)),
        "v3_ceiling_proximity": 100.0 - score_max,
        "v3_mean_over_std": _safe_divide_series(score_mean, abs_score_std.clip(lower=1e-6)),
        "v3_z_quality": _safe_divide_series(score_delta, abs_score_std.clip(lower=1e-6)),
        "v3_range_over_mean": _safe_divide_series(score_range, abs_score_mean.clip(lower=1e-6)),
        "v3_range_over_std": _safe_divide_series(score_range, abs_score_std.clip(lower=1e-6)),
        "v3_quality_times_count": score_delta * assess_count,
        "v3_quality_times_weight": score_delta * weight_sum,
        "v3_quality_density": _safe_divide_series(score_delta, assess_count.replace(0.0, 1.0)),
        "v3_stability_density": _safe_divide_series(1.0, abs_score_std.clip(lower=1e-6)),
        "v3_type_dom_ratio": type_dom_ratio,
        "v3_exam_heavy": ((exam_ratio >= tma_ratio) & (exam_ratio >= cma_ratio)).astype(float),
        "v3_entropy_x_count": type_entropy * assess_count,
        "v4_mean_minus_std": score_mean - score_std,
        "v4_max_minus_std": score_max - score_std,
        "v4_mean_over_range": _safe_divide_series(score_mean, score_range.clip(lower=1e-6)),
        "v4_ceiling_minus_mean": 100.0 - score_mean,
        "v4_ceiling_minus_min": 100.0 - score_min,
    }
    for col, values in derived.items():
        if col not in X.columns:
            X[col] = values

    X = X.loc[:, ~X.columns.duplicated(keep="first")].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)

    missing_locked = [c for c in LOCKED_42_FEATURES if c not in X.columns]
    if missing_locked:
        base_candidates = [
            "kuz_assess_count",
            "kuz_weight_mean",
            "kuz_weight_sum",
            "kuz_date_min",
            "kuz_date_max",
            "kuz_modules_nunique",
            "kuz_score_mean",
            "kuz_score_std",
            "kuz_score_min",
            "kuz_score_max",
            "kuz_type_cma",
            "kuz_type_exam",
            "kuz_type_tma",
        ]
        missing_base = [c for c in base_candidates if c not in X.columns]
        raise ValueError(
            "Failed to build the full locked 42-feature set from the provided Kuzilek columns. "
            f"Missing locked features: {missing_locked}. Missing base candidates: {missing_base}."
        )

    X.attrs["feature_source"] = "auto_built_locked42"
    X.attrs["kuz_aliases_created"] = alias_created
    X.attrs["base_columns_seen"] = sorted([c for c in X_df.columns if str(c).startswith("kuz_")])
    if _date_min is not None:
        X.attrs["kuz_date_min_seen"] = True
    if _date_max is not None:
        X.attrs["kuz_date_max_seen"] = True
    return X


def resolve_mode_feature_sets(X_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    X_norm = normalize_kuzilek_base_columns(X_df)
    if all(c in X_norm.columns for c in LOCKED_42_FEATURES):
        X_base = X_norm.copy()
        X_base.attrs["feature_source"] = "direct_locked42"
    else:
        try:
            X_base = build_locked42_from_base_kuzilek(X_norm)
        except Exception as exc:
            raw_base_candidates = [
                "kuz_assess_count",
                "kuz_weight_mean",
                "kuz_weight_sum",
                "kuz_date_min",
                "kuz_date_max",
                "kuz_modules_nunique",
                "kuz_score_mean",
                "kuz_score_std",
                "kuz_score_min",
                "kuz_score_max",
                "kuz_type_cma",
                "kuz_type_exam",
                "kuz_type_tma",
                "kuz_type_CMA",
                "kuz_type_Exam",
                "kuz_type_TMA",
            ]
            present_base = [c for c in raw_base_candidates if c in X_norm.columns]
            missing_base = [c for c in raw_base_candidates if c not in X_norm.columns]
            raise ValueError(
                "Unable to resolve locked42 features from input. "
                f"Detected {len(present_base)} raw/canonical Kuzilek base columns and still failed during engineering. "
                f"Present base columns: {present_base}. Missing base columns: {missing_base}. "
                f"Original error: {exc}"
            ) from exc

    stage_lists = {
        "stage1": STAGE1_FEATURES_42,
        "stage2": STAGE2_FEATURES_42,
        "stage3": STAGE3_FEATURES_42,
    }
    locked_set = set(LOCKED_42_FEATURES)
    for stage_name, cols in stage_lists.items():
        outside = [c for c in cols if c not in locked_set]
        if outside:
            raise ValueError(f"Stage feature list contains columns outside locked master set ({stage_name}): {outside}")

    if mode == "locked42_shared":
        stage_cols = {
            "stage1": LOCKED_42_FEATURES.copy(),
            "stage2": LOCKED_42_FEATURES.copy(),
            "stage3": LOCKED_42_FEATURES.copy(),
        }
    else:
        missing = {
            "stage1": [c for c in STAGE1_FEATURES_42 if c not in X_base.columns],
            "stage2": [c for c in STAGE2_FEATURES_42 if c not in X_base.columns],
            "stage3": [c for c in STAGE3_FEATURES_42 if c not in X_base.columns],
        }
        bad = {k: v for k, v in missing.items() if v}
        if bad:
            raise ValueError(f"Missing required stage-specific columns for mode '{mode}': {bad}")
        stage_cols = {
            "stage1": STAGE1_FEATURES_42.copy(),
            "stage2": STAGE2_FEATURES_42.copy(),
            "stage3": STAGE3_FEATURES_42.copy(),
        }

    union_cols = []
    for key in ["stage1", "stage2", "stage3"]:
        for col in stage_cols[key]:
            if col not in union_cols:
                union_cols.append(col)

    X_use = X_base[union_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    X_use.attrs["feature_source"] = X_base.attrs.get("feature_source", "direct_locked42")
    X_use.attrs["kuz_aliases_created"] = X_base.attrs.get("kuz_aliases_created", X_norm.attrs.get("kuz_aliases_created", []))
    counts = {
        "stage1": len(stage_cols["stage1"]),
        "stage2": len(stage_cols["stage2"]),
        "stage3": len(stage_cols["stage3"]),
        "union": len(union_cols),
    }
    return X_use, stage_cols, counts


def _safe_divide_num(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.divide(a_arr, b_arr + eps)


def build_stage3_features(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = df.copy()
    X = X.loc[:, ~X.columns.duplicated(keep="first")].copy()
    if X.shape[0] == 0:
        return X

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    score_cols_all = [c for c in numeric_cols if "score" in c.lower() and not c.lower().startswith("s3_")]

    excluded_aggregates = {
        "kuz_score_mean",
        "kuz_score_std",
        "kuz_score_min",
        "kuz_score_max",
        "kuz_score_skew_proxy",
        "kuz_score_iqr_proxy",
        "kuz_score_per_assess",
        "kuz_score_per_weight_sum",
        "kuz_score_per_module",
    }
    assessment_score_cols = [c for c in score_cols_all if c not in excluded_aggregates]

    if score_cols_all:
        all_scores = X[score_cols_all].to_numpy(dtype=float)
        valid_all = np.isfinite(all_scores)
        denom_all = np.maximum(valid_all.sum(axis=1), 1)
        below_all = ((all_scores < 40.0) & valid_all).sum(axis=1)
        X["fail_zone_ratio"] = below_all / denom_all

    if assessment_score_cols:
        assess_scores = X[assessment_score_cols].to_numpy(dtype=float)
        valid_assess = np.isfinite(assess_scores)
        denom_assess = np.maximum(valid_assess.sum(axis=1), 1)
        below_assess = ((assess_scores < 40.0) & valid_assess).sum(axis=1)
        X["score_below_40_ratio"] = below_assess / denom_assess

        assess_std = np.nanstd(assess_scores, axis=1)
        X["score_std"] = np.where(np.isfinite(assess_std), assess_std, 0.0)

        split_idx = max(1, len(assessment_score_cols) // 2)
        early_mean = np.nanmean(assess_scores[:, :split_idx], axis=1)
        late_mean = np.nanmean(assess_scores[:, split_idx:], axis=1) if split_idx < len(assessment_score_cols) else early_mean
        early_mean = np.where(np.isfinite(early_mean), early_mean, 0.0)
        late_mean = np.where(np.isfinite(late_mean), late_mean, 0.0)
        X["score_trend"] = late_mean - early_mean
    else:
        if "kuz_score_std" in X.columns:
            X["score_std"] = X["kuz_score_std"].astype(float)
        else:
            X["score_std"] = _safe_series(X, "kuz_score_std")

    if "kuz_score_mean" in X.columns:
        X["pass_fail_margin"] = X["kuz_score_mean"].astype(float) - 40.0
    elif assessment_score_cols:
        assess_mean = np.nanmean(X[assessment_score_cols].to_numpy(dtype=float), axis=1)
        assess_mean = np.where(np.isfinite(assess_mean), assess_mean, 0.0)
        X["pass_fail_margin"] = assess_mean - 40.0
    else:
        X["pass_fail_margin"] = _safe_series(X, "kuz_score_mean") - 40.0

    if bool(stage3_plus):
        if "kuz_score_max" in X.columns:
            X["score_ceiling_gap"] = 100.0 - X["kuz_score_max"].astype(float)
        if "kuz_score_mean" in X.columns and "kuz_score_min" in X.columns:
            X["score_floor_gap"] = X["kuz_score_mean"].astype(float) - X["kuz_score_min"].astype(float)
        if "kuz_score_max" in X.columns and "kuz_score_mean" in X.columns:
            X["score_peak_margin"] = X["kuz_score_max"].astype(float) - X["kuz_score_mean"].astype(float)
        if "kuz_score_max" in X.columns and "kuz_score_min" in X.columns:
            spread = X["kuz_score_max"].astype(float) - X["kuz_score_min"].astype(float)
            X["score_spread_proxy"] = spread
            if "kuz_score_mean" in X.columns:
                X["score_relative_spread"] = _safe_divide_num(spread.to_numpy(dtype=float), X["kuz_score_mean"].astype(float).to_numpy(dtype=float))
        if "kuz_score_mean" in X.columns and "kuz_score_std" in X.columns:
            X["score_consistency_proxy"] = _safe_divide_num(
                X["kuz_score_mean"].astype(float).to_numpy(dtype=float),
                X["kuz_score_std"].astype(float).to_numpy(dtype=float),
            )
            X["distinction_margin_proxy"] = X["kuz_score_mean"].astype(float) - 70.0
            X["fail_margin_proxy"] = 40.0 - X["kuz_score_mean"].astype(float)
        elif "kuz_score_mean" in X.columns:
            X["distinction_margin_proxy"] = X["kuz_score_mean"].astype(float) - 70.0
            X["fail_margin_proxy"] = 40.0 - X["kuz_score_mean"].astype(float)

        if "score_ceiling_gap" in X.columns and "kuz_type_entropy" in X.columns:
            X["ceiling_x_entropy"] = X["score_ceiling_gap"].astype(float) * X["kuz_type_entropy"].astype(float)
        if "score_consistency_proxy" in X.columns and "v3_quality_times_weight" in X.columns:
            X["stability_x_quality"] = X["score_consistency_proxy"].astype(float) * X["v3_quality_times_weight"].astype(float)
        if "score_trend" in X.columns and "pass_fail_margin" in X.columns:
            X["trend_x_margin"] = X["score_trend"].astype(float) * X["pass_fail_margin"].astype(float)
        if "score_below_40_ratio" in X.columns and "score_std" in X.columns:
            X["below40_x_std"] = X["score_below_40_ratio"].astype(float) * X["score_std"].astype(float)
        if "score_below_40_ratio" in X.columns and "fail_margin_proxy" in X.columns:
            X["below40_x_margin"] = X["score_below_40_ratio"].astype(float) * X["fail_margin_proxy"].astype(float)
        if "kuz_score_mean" in X.columns and "kuz_score_per_assess" in X.columns:
            X["mean_x_assess_density"] = X["kuz_score_mean"].astype(float) * X["kuz_score_per_assess"].astype(float)
        if "fail_zone_ratio" in X.columns and "kuz_type_entropy" in X.columns:
            X["fail_zone_x_entropy"] = X["fail_zone_ratio"].astype(float) * X["kuz_type_entropy"].astype(float)

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    return X


def build_stage3_features_enhanced(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        return X

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    score_cols_all = [c for c in numeric_cols if "score" in c.lower() and not c.lower().startswith("s3_")]
    if score_cols_all:
        all_scores = X[score_cols_all].to_numpy(dtype=float)
        valid_all = np.isfinite(all_scores)
        denom_all = np.maximum(valid_all.sum(axis=1), 1)
        below_30 = ((all_scores < 30.0) & valid_all).sum(axis=1)
        band_40_55 = ((all_scores >= 40.0) & (all_scores < 55.0) & valid_all).sum(axis=1)
        above_70 = ((all_scores >= 70.0) & valid_all).sum(axis=1)
        X["score_below_30_ratio"] = below_30 / denom_all
        X["score_40_55_ratio"] = band_40_55 / denom_all
        X["score_above_70_ratio"] = above_70 / denom_all

    if "kuz_score_mean" in X.columns:
        mean_score = X["kuz_score_mean"].astype(float)
        X["distinction_margin"] = mean_score - 70.0
        X["mean_times_entropy"] = mean_score * X["kuz_type_entropy"].astype(float) if "kuz_type_entropy" in X.columns else mean_score
        X["mean_over_std_stable"] = _safe_divide_num(mean_score, np.abs(_safe_series(X, "kuz_score_std")))
    if "kuz_score_std" in X.columns:
        X["std_over_mean_stable"] = _safe_divide_num(np.abs(X["kuz_score_std"].astype(float)), np.abs(_safe_series(X, "kuz_score_mean")))

    if "kuz_exam_ratio" in X.columns and "kuz_score_mean" in X.columns:
        X["exam_ratio_x_mean"] = X["kuz_exam_ratio"].astype(float) * X["kuz_score_mean"].astype(float)
    if "kuz_tma_ratio" in X.columns and "kuz_score_mean" in X.columns:
        X["tma_ratio_x_mean"] = X["kuz_tma_ratio"].astype(float) * X["kuz_score_mean"].astype(float)

    # ===== NEW SCORE BAND FEATURES =====
    if "kuz_score_mean" in X.columns:
        mean_score = X["kuz_score_mean"].astype(float)
        X["score_band_fail"] = (mean_score < 40).astype(float)
        X["score_band_pass_low"] = ((mean_score >= 40) & (mean_score < 55)).astype(float)
        X["score_band_pass_high"] = ((mean_score >= 55) & (mean_score < 70)).astype(float)
        X["score_band_distinction"] = (mean_score >= 70).astype(float)

    # ===== INTERACTION FEATURES =====
    if "kuz_exam_ratio" in X.columns and "kuz_score_mean" in X.columns:
        X["exam_ratio_x_score"] = X["kuz_exam_ratio"] * X["kuz_score_mean"]
    if "kuz_tma_ratio" in X.columns and "kuz_score_mean" in X.columns:
        X["tma_ratio_x_score"] = X["kuz_tma_ratio"] * X["kuz_score_mean"]
    if "kuz_type_entropy" in X.columns and "kuz_score_mean" in X.columns:
        X["entropy_x_score"] = X["kuz_type_entropy"] * X["kuz_score_mean"]

    # ===== STABILITY FEATURES =====
    if "kuz_score_std" in X.columns and "kuz_assess_count" in X.columns:
        X["std_per_assess"] = X["kuz_score_std"] / (X["kuz_assess_count"] + 1e-6)
    if "kuz_score_max" in X.columns and "kuz_score_min" in X.columns and "kuz_assess_count" in X.columns:
        X["range_per_assess"] = (X["kuz_score_max"] - X["kuz_score_min"]) / (X["kuz_assess_count"] + 1e-6)

    # ===== DENSITY FEATURES =====
    if "kuz_score_mean" in X.columns and "kuz_assess_count" in X.columns:
        X["score_density"] = X["kuz_score_mean"] / (X["kuz_assess_count"] + 1e-6)
    if "kuz_score_mean" in X.columns and "kuz_weight_sum" in X.columns:
        X["score_weight_density"] = X["kuz_score_mean"] / (X["kuz_weight_sum"] + 1e-6)

    # ===== DISTANCE FEATURES =====
    if "kuz_score_mean" in X.columns:
        X["pass_center_distance"] = np.abs(X["kuz_score_mean"] - 55)

    X = add_flat_feature_families(X)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    return X


def build_stage3_features_phase3(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features_enhanced(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        X.attrs["feature_set_version"] = "phase3"
        return X

    score_cols = _ordered_assessment_score_columns(X)
    n_rows = int(X.shape[0])

    phase3_feature_names = [
        "early_mean_score",
        "late_mean_score",
        "score_trend_slope",
        "score_drop_flag",
        "late_peak_score",
        "score_std",
        "score_volatility",
        "high_score_ratio",
        "top_quartile_count",
        "assessment_completion_ratio",
        "late_submission_ratio",
        "missing_assessment_count",
        "completion_gap_max",
        "inactivity_weeks",
        "submission_gap_mean",
        "first_inactivity_week",
        "recent_activity_density",
        "exam_vs_coursework_ratio",
        "consistency_high_score",
        "performance_momentum",
    ]

    if not score_cols:
        for feature_name in phase3_feature_names:
            X[feature_name] = 0.0
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X.attrs["feature_set_version"] = "phase3"
        X.attrs["phase3_features_added"] = phase3_feature_names
        return X

    scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores)
    safe_scores = np.where(valid, scores, np.nan)
    n_cols = int(scores.shape[1])
    if n_cols <= 0:
        for feature_name in phase3_feature_names:
            X[feature_name] = 0.0
        X.attrs["feature_set_version"] = "phase3"
        X.attrs["phase3_features_added"] = phase3_feature_names
        return X

    third = max(1, int(np.ceil(float(n_cols) * 0.30)))
    early_scores = safe_scores[:, :third]
    late_scores = safe_scores[:, max(0, n_cols - third) :]
    early_mean = np.nanmean(early_scores, axis=1)
    late_mean = np.nanmean(late_scores, axis=1)
    late_peak = np.nanmax(late_scores, axis=1)

    early_mean = np.where(np.isfinite(early_mean), early_mean, 0.0)
    late_mean = np.where(np.isfinite(late_mean), late_mean, 0.0)
    late_peak = np.where(np.isfinite(late_peak), late_peak, 0.0)

    weights = valid.astype(float)
    x_idx = np.arange(n_cols, dtype=float)
    y = np.where(valid, scores, 0.0)
    count = np.maximum(weights.sum(axis=1), 1.0)
    sum_x = (weights * x_idx.reshape(1, -1)).sum(axis=1)
    sum_y = (weights * y).sum(axis=1)
    sum_xx = (weights * (x_idx.reshape(1, -1) ** 2)).sum(axis=1)
    sum_xy = (weights * y * x_idx.reshape(1, -1)).sum(axis=1)
    den = count * sum_xx - np.square(sum_x)
    slope = np.where(np.abs(den) > 1e-9, (count * sum_xy - sum_x * sum_y) / den, 0.0)
    slope = np.clip(np.where(np.isfinite(slope), slope, 0.0), -100.0, 100.0)

    drop_flag = (late_mean < (early_mean * 0.85)).astype(float)
    score_std = np.nanstd(safe_scores, axis=1)
    score_std = np.where(np.isfinite(score_std), score_std, 0.0)
    diff = np.abs(np.diff(scores, axis=1))
    valid_diff = valid[:, 1:] & valid[:, :-1]
    diff_sum = np.where(valid_diff, diff, 0.0).sum(axis=1)
    diff_den = np.maximum(valid_diff.sum(axis=1), 1.0)
    score_volatility = np.where(np.isfinite(diff_sum / diff_den), diff_sum / diff_den, 0.0)

    score_count = np.maximum(valid.sum(axis=1), 1.0)
    high_ratio = ((scores >= 80.0) & valid).sum(axis=1) / score_count
    high_ratio = np.clip(np.where(np.isfinite(high_ratio), high_ratio, 0.0), 0.0, 1.0)
    completion_ratio = valid.sum(axis=1) / float(max(n_cols, 1))
    completion_ratio = np.clip(np.where(np.isfinite(completion_ratio), completion_ratio, 0.0), 0.0, 1.0)
    missing_count = float(n_cols) - valid.sum(axis=1).astype(float)
    missing_count = np.where(np.isfinite(missing_count), np.clip(missing_count, 0.0, None), 0.0)

    late_window = valid[:, max(0, n_cols - third) :]
    late_completed = late_window.sum(axis=1).astype(float)
    total_completed = np.maximum(valid.sum(axis=1).astype(float), 1.0)
    late_submission_ratio = np.clip(late_completed / total_completed, 0.0, 1.0)
    recent_activity_density = np.clip(late_completed / float(max(third, 1)), 0.0, 1.0)

    top_quartile_count = np.zeros(shape=(n_rows,), dtype=float)
    completion_gap_max = np.zeros(shape=(n_rows,), dtype=float)
    inactivity_weeks = np.zeros(shape=(n_rows,), dtype=float)
    submission_gap_mean = np.zeros(shape=(n_rows,), dtype=float)
    first_inactivity_week = np.zeros(shape=(n_rows,), dtype=float)
    consistency_high_score = np.zeros(shape=(n_rows,), dtype=float)
    performance_momentum = np.zeros(shape=(n_rows,), dtype=float)

    for i in range(n_rows):
        row_scores = safe_scores[i]
        row_valid = valid[i]
        valid_scores = row_scores[row_valid]
        if valid_scores.size > 0:
            p75 = float(np.nanpercentile(valid_scores, 75))
            top_quartile_count[i] = float((valid_scores >= p75).sum())
            valid_pos = np.where(row_valid)[0]
            last_val = float(row_scores[valid_pos[-1]])
            performance_momentum[i] = float(last_val - early_mean[i])

            high_mask = (valid_scores >= 75.0).astype(bool)
            consistency_high_score[i] = float(_longest_true_streak(high_mask))

            if valid_pos.size > 1:
                submission_gap_mean[i] = float(np.diff(valid_pos).mean())

        missing_mask = ~row_valid
        max_gap = float(_longest_true_streak(missing_mask))
        completion_gap_max[i] = max_gap
        inactivity_weeks[i] = max_gap
        first_inactivity_week[i] = float(_first_long_gap_index(row_valid, min_gap=2))

    exam_cols = [c for c in score_cols if "exam" in str(c).strip().lower()]
    coursework_cols = [
        c
        for c in score_cols
        if any(token in str(c).strip().lower() for token in ["coursework", "tma", "cma", "cw"])
    ]
    if exam_cols and coursework_cols:
        exam_mean = np.nanmean(X[exam_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), axis=1)
        coursework_mean = np.nanmean(X[coursework_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), axis=1)
        exam_vs_coursework_ratio = _safe_divide_series(
            pd.Series(np.where(np.isfinite(exam_mean), exam_mean, 0.0), index=X.index),
            pd.Series(np.where(np.isfinite(coursework_mean), coursework_mean, 0.0), index=X.index).clip(lower=1e-6),
            default=0.0,
        )
    else:
        exam_proxy = _safe_series(X, "exam_weighted_mean_if_available")
        coursework_proxy = _safe_series(X, "tma_weighted_mean_if_available").clip(lower=1e-6)
        exam_vs_coursework_ratio = _safe_divide_series(exam_proxy, coursework_proxy, default=0.0)
    exam_vs_coursework_ratio = exam_vs_coursework_ratio.clip(lower=0.0, upper=10.0)

    X["early_mean_score"] = np.clip(early_mean, 0.0, 100.0)
    X["late_mean_score"] = np.clip(late_mean, 0.0, 100.0)
    X["score_trend_slope"] = slope
    X["score_drop_flag"] = drop_flag
    X["late_peak_score"] = np.clip(late_peak, 0.0, 100.0)
    X["score_std"] = np.clip(score_std, 0.0, 100.0)
    X["score_volatility"] = np.clip(score_volatility, 0.0, 100.0)
    X["high_score_ratio"] = high_ratio
    X["top_quartile_count"] = np.clip(top_quartile_count, 0.0, float(n_cols))
    X["assessment_completion_ratio"] = completion_ratio
    X["late_submission_ratio"] = late_submission_ratio
    X["missing_assessment_count"] = missing_count
    X["completion_gap_max"] = np.clip(completion_gap_max, 0.0, float(n_cols))
    X["inactivity_weeks"] = np.clip(inactivity_weeks, 0.0, float(n_cols))
    X["submission_gap_mean"] = np.clip(submission_gap_mean, 0.0, float(n_cols))
    X["first_inactivity_week"] = np.clip(first_inactivity_week, 0.0, float(n_cols))
    X["recent_activity_density"] = recent_activity_density
    X["exam_vs_coursework_ratio"] = exam_vs_coursework_ratio
    X["consistency_high_score"] = np.clip(consistency_high_score, 0.0, float(n_cols))
    X["performance_momentum"] = np.clip(performance_momentum, -100.0, 100.0)

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.attrs["feature_set_version"] = "phase3"
    X.attrs["phase3_features_added"] = phase3_feature_names
    return X


def build_stage3_features_phase4(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features_phase3(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        X.attrs["feature_set_version"] = "phase4"
        return X

    phase4_feature_names = [
        "score_above_85_ratio",
        "score_above_90_ratio",
        "high_score_streak",
        "exam_mean_score",
        "coursework_mean_score",
        "exam_coursework_gap",
        "score_trend_strength",
        "score_improvement_last3",
        "final_vs_mean_score_gap",
        "score_zscore_per_module",
        "relative_rank_in_module",
    ]
    for feature_name in phase4_feature_names:
        if feature_name not in X.columns:
            X[feature_name] = 0.0

    score_cols = _ordered_assessment_score_columns(X)
    if score_cols:
        scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(scores)
        safe_scores = np.where(valid, scores, np.nan)
        n_rows = int(scores.shape[0])
        n_cols = int(scores.shape[1])
        valid_count = np.maximum(valid.sum(axis=1).astype(float), 1.0)

        score_above_85_ratio = ((scores >= 85.0) & valid).sum(axis=1).astype(float) / valid_count
        score_above_90_ratio = ((scores >= 90.0) & valid).sum(axis=1).astype(float) / valid_count
        score_above_85_ratio = np.clip(np.where(np.isfinite(score_above_85_ratio), score_above_85_ratio, 0.0), 0.0, 1.0)
        score_above_90_ratio = np.clip(np.where(np.isfinite(score_above_90_ratio), score_above_90_ratio, 0.0), 0.0, 1.0)

        high_score_streak = np.zeros(shape=(n_rows,), dtype=float)
        score_improvement_last3 = np.zeros(shape=(n_rows,), dtype=float)
        final_vs_mean_score_gap = np.zeros(shape=(n_rows,), dtype=float)

        for i in range(n_rows):
            row_scores = safe_scores[i]
            row_valid = valid[i]
            valid_scores = row_scores[row_valid]
            if valid_scores.size <= 0:
                continue

            high_score_streak[i] = float(_longest_true_streak((valid_scores >= 85.0).astype(bool)))

            if valid_scores.size >= 2:
                final_vs_mean_score_gap[i] = float(valid_scores[-1] - np.nanmean(valid_scores))
            else:
                final_vs_mean_score_gap[i] = 0.0

            if valid_scores.size >= 4:
                tail = valid_scores[-3:]
                prev = valid_scores[-6:-3] if valid_scores.size >= 6 else valid_scores[: min(3, valid_scores.size - 1)]
                score_improvement_last3[i] = float(np.nanmean(tail) - np.nanmean(prev))
            elif valid_scores.size >= 2:
                score_improvement_last3[i] = float(valid_scores[-1] - valid_scores[0])
            else:
                score_improvement_last3[i] = 0.0

        exam_cols = [c for c in score_cols if "exam" in str(c).strip().lower()]
        coursework_cols = [
            c
            for c in score_cols
            if any(token in str(c).strip().lower() for token in ["coursework", "tma", "cma", "cw"])
        ]
        if exam_cols:
            exam_scores = X[exam_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            exam_mean_score = np.nanmean(np.where(np.isfinite(exam_scores), exam_scores, np.nan), axis=1)
            exam_mean_score = np.where(np.isfinite(exam_mean_score), exam_mean_score, 0.0)
        else:
            exam_mean_score = _safe_series(X, "exam_weighted_mean_if_available").to_numpy(dtype=float)

        if coursework_cols:
            coursework_scores = X[coursework_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            coursework_mean_score = np.nanmean(np.where(np.isfinite(coursework_scores), coursework_scores, np.nan), axis=1)
            coursework_mean_score = np.where(np.isfinite(coursework_mean_score), coursework_mean_score, 0.0)
        else:
            coursework_mean_score = _safe_series(X, "tma_weighted_mean_if_available").to_numpy(dtype=float)

        exam_coursework_gap = np.where(
            np.isfinite(exam_mean_score - coursework_mean_score),
            exam_mean_score - coursework_mean_score,
            0.0,
        )

        x_idx = np.arange(n_cols, dtype=float)
        y = np.where(valid, scores, 0.0)
        weights = valid.astype(float)
        count = np.maximum(weights.sum(axis=1), 1.0)
        sum_x = (weights * x_idx.reshape(1, -1)).sum(axis=1)
        sum_y = (weights * y).sum(axis=1)
        sum_xx = (weights * (x_idx.reshape(1, -1) ** 2)).sum(axis=1)
        sum_xy = (weights * y * x_idx.reshape(1, -1)).sum(axis=1)
        den = count * sum_xx - np.square(sum_x)
        slope = np.where(np.abs(den) > 1e-9, (count * sum_xy - sum_x * sum_y) / den, 0.0)
        score_trend_strength = np.clip(np.abs(np.where(np.isfinite(slope), slope, 0.0)), 0.0, 100.0)

        module_signal = _safe_series(X, "kuz_score_per_module")
        if float(np.nanstd(module_signal.to_numpy(dtype=float))) <= 1e-9:
            module_signal = _safe_series(X, "kuz_score_mean")
        module_mean = float(module_signal.mean())
        module_std = float(module_signal.std(ddof=0))
        if module_std <= 1e-9:
            score_zscore_per_module = np.zeros(shape=(X.shape[0],), dtype=float)
        else:
            score_zscore_per_module = ((module_signal.to_numpy(dtype=float) - module_mean) / module_std).astype(float)
        score_zscore_per_module = np.clip(np.where(np.isfinite(score_zscore_per_module), score_zscore_per_module, 0.0), -10.0, 10.0)
        relative_rank_in_module = module_signal.rank(pct=True, method="average").fillna(0.0).to_numpy(dtype=float)
        relative_rank_in_module = np.clip(np.where(np.isfinite(relative_rank_in_module), relative_rank_in_module, 0.0), 0.0, 1.0)

        X["score_above_85_ratio"] = score_above_85_ratio
        X["score_above_90_ratio"] = score_above_90_ratio
        X["high_score_streak"] = np.clip(high_score_streak, 0.0, float(max(n_cols, 1)))
        X["exam_mean_score"] = np.clip(np.where(np.isfinite(exam_mean_score), exam_mean_score, 0.0), 0.0, 100.0)
        X["coursework_mean_score"] = np.clip(np.where(np.isfinite(coursework_mean_score), coursework_mean_score, 0.0), 0.0, 100.0)
        X["exam_coursework_gap"] = np.clip(exam_coursework_gap, -100.0, 100.0)
        X["score_trend_strength"] = score_trend_strength
        X["score_improvement_last3"] = np.clip(np.where(np.isfinite(score_improvement_last3), score_improvement_last3, 0.0), -100.0, 100.0)
        X["final_vs_mean_score_gap"] = np.clip(np.where(np.isfinite(final_vs_mean_score_gap), final_vs_mean_score_gap, 0.0), -100.0, 100.0)
        X["score_zscore_per_module"] = score_zscore_per_module
        X["relative_rank_in_module"] = relative_rank_in_module

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.attrs["feature_set_version"] = "phase4"
    X.attrs["phase4_features_added"] = phase4_feature_names
    return X


def build_stage3_features_phase5(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features_phase4(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        X.attrs["feature_set_version"] = "phase5"
        return X

    phase5_feature_names = [
        "score_above_95_ratio",
        "high_score_streak_90",
        "consistency_above_88",
        "peak_minus_p90_gap",
        "exam_top_tail_gap",
        "coursework_top_tail_gap",
        "final_third_above_85_ratio",
        "late_vs_mid_high_score_gap",
        "module_top_score_std",
        "module_distinction_hit_ratio",
    ]
    for feature_name in phase5_feature_names:
        if feature_name not in X.columns:
            X[feature_name] = 0.0

    score_cols = _ordered_assessment_score_columns(X)
    if score_cols:
        scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(scores)
        safe_scores = np.where(valid, scores, np.nan)
        n_rows = int(scores.shape[0])
        n_cols = int(scores.shape[1])
        valid_count = np.maximum(valid.sum(axis=1).astype(float), 1.0)

        score_above_95_ratio = ((scores >= 95.0) & valid).sum(axis=1).astype(float) / valid_count
        consistency_above_88 = ((scores >= 88.0) & valid).sum(axis=1).astype(float) / valid_count
        score_above_95_ratio = np.clip(np.where(np.isfinite(score_above_95_ratio), score_above_95_ratio, 0.0), 0.0, 1.0)
        consistency_above_88 = np.clip(np.where(np.isfinite(consistency_above_88), consistency_above_88, 0.0), 0.0, 1.0)

        high_score_streak_90 = np.zeros(shape=(n_rows,), dtype=float)
        peak_minus_p90_gap = np.zeros(shape=(n_rows,), dtype=float)
        final_third_above_85_ratio = np.zeros(shape=(n_rows,), dtype=float)
        late_vs_mid_high_score_gap = np.zeros(shape=(n_rows,), dtype=float)

        third_n = max(1, int(np.ceil(float(n_cols) / 3.0)))
        final_start = max(0, n_cols - third_n)
        mid_start = third_n
        mid_end = min(n_cols, third_n * 2)
        if mid_start >= mid_end:
            mid_start = 0
            mid_end = max(1, n_cols // 2)

        for i in range(n_rows):
            row_scores = safe_scores[i]
            row_valid = valid[i]
            valid_scores = row_scores[row_valid]
            if valid_scores.size > 0:
                high_score_streak_90[i] = float(_longest_true_streak((valid_scores >= 90.0).astype(bool)))
                row_max = float(np.nanmax(valid_scores))
                row_p90 = float(np.nanpercentile(valid_scores, 90))
                peak_minus_p90_gap[i] = float(row_max - row_p90) if np.isfinite(row_max - row_p90) else 0.0

            final_mask = row_valid[final_start:]
            final_scores = row_scores[final_start:]
            final_den = max(int(final_mask.sum()), 1)
            final_num = int(((final_scores >= 85.0) & final_mask).sum())
            final_ratio = float(final_num) / float(final_den)
            final_third_above_85_ratio[i] = final_ratio if np.isfinite(final_ratio) else 0.0

            mid_mask = row_valid[mid_start:mid_end]
            mid_scores = row_scores[mid_start:mid_end]
            mid_den = max(int(mid_mask.sum()), 1)
            mid_num = int(((mid_scores >= 85.0) & mid_mask).sum())
            mid_ratio = float(mid_num) / float(mid_den)
            mid_ratio = mid_ratio if np.isfinite(mid_ratio) else 0.0
            late_vs_mid_high_score_gap[i] = final_third_above_85_ratio[i] - mid_ratio

        exam_cols = [c for c in score_cols if "exam" in str(c).strip().lower()]
        coursework_cols = [
            c
            for c in score_cols
            if any(token in str(c).strip().lower() for token in ["coursework", "tma", "cma", "cw"])
        ]

        def _top_tail_gap(columns: list[str]) -> np.ndarray:
            if not columns:
                return np.zeros(shape=(n_rows,), dtype=float)
            values = X[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, np.nan)
            out = np.zeros(shape=(n_rows,), dtype=float)
            for i in range(n_rows):
                row_values = values[i]
                row_values = row_values[np.isfinite(row_values)]
                if row_values.size <= 0:
                    continue
                p95 = float(np.nanpercentile(row_values, 95))
                p75 = float(np.nanpercentile(row_values, 75))
                gap = p95 - p75
                out[i] = gap if np.isfinite(gap) else 0.0
            return out

        exam_top_tail_gap = _top_tail_gap(exam_cols)
        coursework_top_tail_gap = _top_tail_gap(coursework_cols)

        module_top_score_std = np.zeros(shape=(n_rows,), dtype=float)
        module_distinction_hit_ratio = np.zeros(shape=(n_rows,), dtype=float)
        module_groups: dict[str, list[int]] = {}
        for idx, col in enumerate(score_cols):
            col_l = str(col).strip().lower()
            match = re.match(r"([^_]+)", col_l)
            module_key = str(match.group(1)) if match else col_l
            module_groups.setdefault(module_key, []).append(idx)

        if len(module_groups) > 1:
            group_indices = list(module_groups.values())
            for i in range(n_rows):
                row_scores = safe_scores[i]
                per_module_max: list[float] = []
                per_module_mean: list[float] = []
                for idxs in group_indices:
                    vals = row_scores[idxs]
                    vals = vals[np.isfinite(vals)]
                    if vals.size <= 0:
                        continue
                    per_module_max.append(float(np.nanmax(vals)))
                    per_module_mean.append(float(np.nanmean(vals)))
                if per_module_max:
                    std_val = float(np.std(per_module_max, ddof=0))
                    module_top_score_std[i] = std_val if np.isfinite(std_val) else 0.0
                if per_module_mean:
                    ratio = float(np.mean(np.asarray(per_module_mean, dtype=float) >= 85.0))
                    module_distinction_hit_ratio[i] = ratio if np.isfinite(ratio) else 0.0
        else:
            base_std = _safe_series(X, "kuz_score_std")
            module_top_score_std = np.clip(base_std.to_numpy(dtype=float), 0.0, 100.0)
            module_distinction_hit_ratio = np.clip(_safe_divide_series(_safe_series(X, "kuz_score_mean"), 100.0), 0.0, 1.0).to_numpy(dtype=float)

        X["score_above_95_ratio"] = score_above_95_ratio
        X["high_score_streak_90"] = np.clip(high_score_streak_90, 0.0, float(max(n_cols, 1)))
        X["consistency_above_88"] = consistency_above_88
        X["peak_minus_p90_gap"] = np.clip(np.where(np.isfinite(peak_minus_p90_gap), peak_minus_p90_gap, 0.0), 0.0, 100.0)
        X["exam_top_tail_gap"] = np.clip(np.where(np.isfinite(exam_top_tail_gap), exam_top_tail_gap, 0.0), 0.0, 100.0)
        X["coursework_top_tail_gap"] = np.clip(np.where(np.isfinite(coursework_top_tail_gap), coursework_top_tail_gap, 0.0), 0.0, 100.0)
        X["final_third_above_85_ratio"] = np.clip(np.where(np.isfinite(final_third_above_85_ratio), final_third_above_85_ratio, 0.0), 0.0, 1.0)
        X["late_vs_mid_high_score_gap"] = np.clip(np.where(np.isfinite(late_vs_mid_high_score_gap), late_vs_mid_high_score_gap, 0.0), -1.0, 1.0)
        X["module_top_score_std"] = np.clip(np.where(np.isfinite(module_top_score_std), module_top_score_std, 0.0), 0.0, 100.0)
        X["module_distinction_hit_ratio"] = np.clip(np.where(np.isfinite(module_distinction_hit_ratio), module_distinction_hit_ratio, 0.0), 0.0, 1.0)

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.attrs["feature_set_version"] = "phase5"
    X.attrs["phase5_features_added"] = phase5_feature_names
    return X


def build_stage3_features_phase6(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features_phase5(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        X.attrs["feature_set_version"] = "phase6"
        return X

    phase6_feature_names = [
        "interaction_high90_x_rank",
        "interaction_finalgap_x_streak",
        "interaction_examgap_x_trend",
        "interaction_high85_x_module_hit",
    ]

    score_above_90_ratio = _safe_series(X, "score_above_90_ratio")
    relative_rank_in_module = _safe_series(X, "relative_rank_in_module")
    final_vs_mean_score_gap = _safe_series(X, "final_vs_mean_score_gap")
    high_score_streak = _safe_series(X, "high_score_streak")
    exam_coursework_gap = _safe_series(X, "exam_coursework_gap")
    score_trend_strength = _safe_series(X, "score_trend_strength")
    score_above_85_ratio = _safe_series(X, "score_above_85_ratio")
    module_distinction_hit_ratio = _safe_series(X, "module_distinction_hit_ratio")

    X["interaction_high90_x_rank"] = np.clip(
        np.where(
            np.isfinite(score_above_90_ratio * relative_rank_in_module),
            score_above_90_ratio * relative_rank_in_module,
            0.0,
        ),
        0.0,
        1.0,
    )
    X["interaction_finalgap_x_streak"] = np.clip(
        np.where(
            np.isfinite(final_vs_mean_score_gap * high_score_streak),
            final_vs_mean_score_gap * high_score_streak,
            0.0,
        ),
        -1000.0,
        1000.0,
    )
    X["interaction_examgap_x_trend"] = np.clip(
        np.where(
            np.isfinite(exam_coursework_gap * score_trend_strength),
            exam_coursework_gap * score_trend_strength,
            0.0,
        ),
        -10000.0,
        10000.0,
    )
    X["interaction_high85_x_module_hit"] = np.clip(
        np.where(
            np.isfinite(score_above_85_ratio * module_distinction_hit_ratio),
            score_above_85_ratio * module_distinction_hit_ratio,
            0.0,
        ),
        0.0,
        1.0,
    )

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.attrs["feature_set_version"] = "phase6"
    X.attrs["phase6_features_added"] = phase6_feature_names
    return X


def build_stage3_features_phase10(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features_phase6(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        X.attrs["feature_set_version"] = "phase10"
        return X

    phase10_feature_names = [
        "score_trend_slope",
        "score_improvement_rate",
        "score_last_vs_mean",
        "score_tail_mean",
        "score_head_mean",
        "score_tail_minus_head",
        "score_peak_vs_mean",
        "score_volatility",
        "score_consistency_high_band",
        "score_recovery_after_drop",
        "activity_head_mean",
        "activity_tail_mean",
        "activity_decay_ratio",
        "activity_tail_minus_head",
        "activity_last_vs_first",
        "engagement_volatility",
        "late_submission_ratio",
        "missing_submission_ratio",
        "ontime_submission_ratio",
        "avg_submission_delay",
        "max_submission_delay",
        "completion_rate",
        "completed_count",
        "missing_count",
        "assessment_density",
        "engagement_x_score",
        "completion_x_score",
        "activity_decay_x_score",
        "late_ratio_x_score",
        "volatility_x_score",
        "recovery_x_tail",
        "consistency_x_tail",
    ]

    score_cols = _ordered_assessment_score_columns(X)
    if not score_cols:
        for feature_name in phase10_feature_names:
            if feature_name not in X.columns:
                X[feature_name] = 0.0
        X = X.loc[:, ~X.columns.duplicated(keep="first")]
        X = X.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X.attrs["feature_set_version"] = "phase10"
        X.attrs["phase10_features_added"] = phase10_feature_names
        return X

    scores = X[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores)
    safe_scores = np.where(valid, scores, np.nan)
    n_rows = int(scores.shape[0])
    n_cols = int(scores.shape[1])
    valid_count = np.maximum(valid.sum(axis=1).astype(float), 1.0)
    overall_mean = np.nanmean(safe_scores, axis=1)
    overall_mean = np.where(np.isfinite(overall_mean), overall_mean, 0.0)
    score_peak = np.nanmax(safe_scores, axis=1)
    score_peak = np.where(np.isfinite(score_peak), score_peak, 0.0)

    # Trajectory windows (head/tail = first/last third).
    third = max(1, int(np.ceil(float(n_cols) / 3.0)))
    head_slice = slice(0, third)
    tail_slice = slice(max(0, n_cols - third), n_cols)
    head_scores = safe_scores[:, head_slice]
    tail_scores = safe_scores[:, tail_slice]
    score_head_mean = np.nanmean(head_scores, axis=1)
    score_tail_mean = np.nanmean(tail_scores, axis=1)
    score_head_mean = np.where(np.isfinite(score_head_mean), score_head_mean, 0.0)
    score_tail_mean = np.where(np.isfinite(score_tail_mean), score_tail_mean, 0.0)

    # Linear slope over assessment order.
    x_idx = np.arange(n_cols, dtype=float)
    w = valid.astype(float)
    y = np.where(valid, scores, 0.0)
    sum_w = np.maximum(w.sum(axis=1), 1.0)
    sum_x = (w * x_idx.reshape(1, -1)).sum(axis=1)
    sum_y = (w * y).sum(axis=1)
    sum_xx = (w * (x_idx.reshape(1, -1) ** 2)).sum(axis=1)
    sum_xy = (w * y * x_idx.reshape(1, -1)).sum(axis=1)
    den = sum_w * sum_xx - np.square(sum_x)
    score_trend_slope = np.where(np.abs(den) > 1e-9, (sum_w * sum_xy - sum_x * sum_y) / den, 0.0)
    score_trend_slope = np.clip(np.where(np.isfinite(score_trend_slope), score_trend_slope, 0.0), -100.0, 100.0)

    score_last = np.where(np.isfinite(tail_scores[:, -1]), tail_scores[:, -1], score_tail_mean)
    score_first = np.where(np.isfinite(head_scores[:, 0]), head_scores[:, 0], score_head_mean)
    score_last_vs_mean = np.where(np.isfinite(score_last - overall_mean), score_last - overall_mean, 0.0)
    score_improvement_rate = _safe_divide_series(
        pd.Series(score_last - score_first, index=X.index),
        pd.Series(np.maximum(np.abs(score_first), 1.0), index=X.index),
        default=0.0,
    ).to_numpy(dtype=float)
    score_tail_minus_head = np.where(np.isfinite(score_tail_mean - score_head_mean), score_tail_mean - score_head_mean, 0.0)
    score_peak_vs_mean = np.where(np.isfinite(score_peak - overall_mean), score_peak - overall_mean, 0.0)

    diff = np.abs(np.diff(scores, axis=1))
    valid_diff = valid[:, 1:] & valid[:, :-1]
    diff_sum = np.where(valid_diff, diff, 0.0).sum(axis=1)
    diff_den = np.maximum(valid_diff.sum(axis=1), 1.0)
    score_volatility = np.where(np.isfinite(diff_sum / diff_den), diff_sum / diff_den, 0.0)
    score_consistency_high_band = ((scores >= 75.0) & valid).sum(axis=1).astype(float) / valid_count
    score_consistency_high_band = np.clip(np.where(np.isfinite(score_consistency_high_band), score_consistency_high_band, 0.0), 0.0, 1.0)

    score_recovery_after_drop = np.zeros(shape=(n_rows,), dtype=float)
    for i in range(n_rows):
        row = safe_scores[i]
        row_valid = np.isfinite(row)
        vals = row[row_valid]
        if vals.size < 3:
            continue
        row_diff = np.diff(vals)
        drop_idx = int(np.argmin(row_diff))
        largest_drop = float(row_diff[drop_idx])
        post = vals[drop_idx + 1 :]
        if post.size <= 1:
            continue
        trough = float(np.min(post))
        rebound = float(np.max(post) - trough)
        recovery = rebound / max(abs(largest_drop), 1.0)
        score_recovery_after_drop[i] = recovery if np.isfinite(recovery) else 0.0

    # Engagement/activity columns (fallback to completion mask by assessment position).
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    score_col_set = set(score_cols)
    activity_cols = [
        c
        for c in numeric_cols
        if c not in score_col_set
        and any(tok in str(c).strip().lower() for tok in ["activity", "engagement", "click", "interaction"])
    ]
    if activity_cols:
        activity = X[activity_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        activity_valid = np.isfinite(activity)
        safe_activity = np.where(activity_valid, activity, np.nan)
    else:
        safe_activity = np.where(valid, 1.0, 0.0)
        activity_valid = np.isfinite(safe_activity)

    act_cols = int(safe_activity.shape[1]) if safe_activity.ndim == 2 else 0
    if act_cols <= 0:
        activity_head_mean = np.zeros(shape=(n_rows,), dtype=float)
        activity_tail_mean = np.zeros(shape=(n_rows,), dtype=float)
        activity_last_vs_first = np.zeros(shape=(n_rows,), dtype=float)
        engagement_volatility = np.zeros(shape=(n_rows,), dtype=float)
    else:
        act_third = max(1, int(np.ceil(float(act_cols) / 3.0)))
        act_head = safe_activity[:, :act_third]
        act_tail = safe_activity[:, max(0, act_cols - act_third) :]
        activity_head_mean = np.nanmean(act_head, axis=1)
        activity_tail_mean = np.nanmean(act_tail, axis=1)
        activity_head_mean = np.where(np.isfinite(activity_head_mean), activity_head_mean, 0.0)
        activity_tail_mean = np.where(np.isfinite(activity_tail_mean), activity_tail_mean, 0.0)
        activity_first = np.where(np.isfinite(act_head[:, 0]), act_head[:, 0], activity_head_mean)
        activity_last = np.where(np.isfinite(act_tail[:, -1]), act_tail[:, -1], activity_tail_mean)
        activity_last_vs_first = np.where(np.isfinite(activity_last - activity_first), activity_last - activity_first, 0.0)

        act_diff = np.abs(np.diff(np.where(np.isfinite(safe_activity), safe_activity, 0.0), axis=1))
        act_valid_diff = activity_valid[:, 1:] & activity_valid[:, :-1]
        act_diff_sum = np.where(act_valid_diff, act_diff, 0.0).sum(axis=1)
        act_diff_den = np.maximum(act_valid_diff.sum(axis=1), 1.0)
        engagement_volatility = np.where(np.isfinite(act_diff_sum / act_diff_den), act_diff_sum / act_diff_den, 0.0)

    activity_decay_ratio = _safe_divide_series(
        pd.Series(activity_tail_mean, index=X.index),
        pd.Series(np.maximum(np.abs(activity_head_mean), 1e-6), index=X.index),
        default=0.0,
    ).to_numpy(dtype=float)
    activity_tail_minus_head = np.where(np.isfinite(activity_tail_mean - activity_head_mean), activity_tail_mean - activity_head_mean, 0.0)

    # Submission-delay features (fallback to existing completion-derived proxies).
    delay_cols = [
        c
        for c in numeric_cols
        if any(tok in str(c).strip().lower() for tok in ["delay", "days_late", "late_days", "submission_delay", "lag"])
    ]
    if delay_cols:
        delays = X[delay_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        delay_valid = np.isfinite(delays)
        safe_delays = np.where(delay_valid, delays, 0.0)
        completed_count = delay_valid.sum(axis=1).astype(float)
        avg_submission_delay = _safe_divide_series(
            pd.Series(np.where(delay_valid, safe_delays, 0.0).sum(axis=1), index=X.index),
            pd.Series(np.maximum(completed_count, 1.0), index=X.index),
            default=0.0,
        ).to_numpy(dtype=float)
        max_submission_delay = np.nanmax(np.where(delay_valid, safe_delays, np.nan), axis=1)
        max_submission_delay = np.where(np.isfinite(max_submission_delay), max_submission_delay, 0.0)
        late_submission_ratio = _safe_divide_series(
            pd.Series(((safe_delays > 0.0) & delay_valid).sum(axis=1).astype(float), index=X.index),
            pd.Series(np.maximum(completed_count, 1.0), index=X.index),
            default=0.0,
        ).to_numpy(dtype=float)
    else:
        completed_count = valid.sum(axis=1).astype(float)
        avg_submission_delay = np.zeros(shape=(n_rows,), dtype=float)
        max_submission_delay = np.zeros(shape=(n_rows,), dtype=float)
        late_submission_ratio = _safe_series(X, "late_submission_ratio").to_numpy(dtype=float)
        late_submission_ratio = np.clip(np.where(np.isfinite(late_submission_ratio), late_submission_ratio, 0.0), 0.0, 1.0)

    total_assess = float(max(n_cols, 1))
    completion_rate = np.clip(completed_count / total_assess, 0.0, 1.0)
    missing_count = np.clip(total_assess - completed_count, 0.0, None)
    missing_submission_ratio = np.clip(missing_count / total_assess, 0.0, 1.0)
    ontime_submission_ratio = np.clip(1.0 - late_submission_ratio - missing_submission_ratio, 0.0, 1.0)
    assessment_density = np.clip(completed_count / total_assess, 0.0, 1.0)

    # Interactions (bounded to avoid outliers dominating trees).
    engagement_x_score = activity_tail_mean * (score_tail_mean / 100.0)
    completion_x_score = completion_rate * (score_tail_mean / 100.0)
    activity_decay_x_score = activity_decay_ratio * score_trend_slope
    late_ratio_x_score = late_submission_ratio * (score_tail_mean / 100.0)
    volatility_x_score = score_volatility * (score_tail_mean / 100.0)
    recovery_x_tail = score_recovery_after_drop * (score_tail_mean / 100.0)
    consistency_x_tail = score_consistency_high_band * (score_tail_mean / 100.0)

    X["score_trend_slope"] = np.clip(np.where(np.isfinite(score_trend_slope), score_trend_slope, 0.0), -100.0, 100.0)
    X["score_improvement_rate"] = np.clip(np.where(np.isfinite(score_improvement_rate), score_improvement_rate, 0.0), -5.0, 5.0)
    X["score_last_vs_mean"] = np.clip(np.where(np.isfinite(score_last_vs_mean), score_last_vs_mean, 0.0), -100.0, 100.0)
    X["score_tail_mean"] = np.clip(np.where(np.isfinite(score_tail_mean), score_tail_mean, 0.0), 0.0, 100.0)
    X["score_head_mean"] = np.clip(np.where(np.isfinite(score_head_mean), score_head_mean, 0.0), 0.0, 100.0)
    X["score_tail_minus_head"] = np.clip(np.where(np.isfinite(score_tail_minus_head), score_tail_minus_head, 0.0), -100.0, 100.0)
    X["score_peak_vs_mean"] = np.clip(np.where(np.isfinite(score_peak_vs_mean), score_peak_vs_mean, 0.0), 0.0, 100.0)
    X["score_volatility"] = np.clip(np.where(np.isfinite(score_volatility), score_volatility, 0.0), 0.0, 100.0)
    X["score_consistency_high_band"] = np.clip(np.where(np.isfinite(score_consistency_high_band), score_consistency_high_band, 0.0), 0.0, 1.0)
    X["score_recovery_after_drop"] = np.clip(np.where(np.isfinite(score_recovery_after_drop), score_recovery_after_drop, 0.0), 0.0, 10.0)

    X["activity_head_mean"] = np.clip(np.where(np.isfinite(activity_head_mean), activity_head_mean, 0.0), 0.0, 1e6)
    X["activity_tail_mean"] = np.clip(np.where(np.isfinite(activity_tail_mean), activity_tail_mean, 0.0), 0.0, 1e6)
    X["activity_decay_ratio"] = np.clip(np.where(np.isfinite(activity_decay_ratio), activity_decay_ratio, 0.0), 0.0, 10.0)
    X["activity_tail_minus_head"] = np.clip(np.where(np.isfinite(activity_tail_minus_head), activity_tail_minus_head, 0.0), -1e6, 1e6)
    X["activity_last_vs_first"] = np.clip(np.where(np.isfinite(activity_last_vs_first), activity_last_vs_first, 0.0), -1e6, 1e6)
    X["engagement_volatility"] = np.clip(np.where(np.isfinite(engagement_volatility), engagement_volatility, 0.0), 0.0, 1e6)

    X["late_submission_ratio"] = np.clip(np.where(np.isfinite(late_submission_ratio), late_submission_ratio, 0.0), 0.0, 1.0)
    X["missing_submission_ratio"] = np.clip(np.where(np.isfinite(missing_submission_ratio), missing_submission_ratio, 0.0), 0.0, 1.0)
    X["ontime_submission_ratio"] = np.clip(np.where(np.isfinite(ontime_submission_ratio), ontime_submission_ratio, 0.0), 0.0, 1.0)
    X["avg_submission_delay"] = np.clip(np.where(np.isfinite(avg_submission_delay), avg_submission_delay, 0.0), 0.0, 365.0)
    X["max_submission_delay"] = np.clip(np.where(np.isfinite(max_submission_delay), max_submission_delay, 0.0), 0.0, 365.0)

    X["completion_rate"] = np.clip(np.where(np.isfinite(completion_rate), completion_rate, 0.0), 0.0, 1.0)
    X["completed_count"] = np.clip(np.where(np.isfinite(completed_count), completed_count, 0.0), 0.0, float(max(n_cols, 1)))
    X["missing_count"] = np.clip(np.where(np.isfinite(missing_count), missing_count, 0.0), 0.0, float(max(n_cols, 1)))
    X["assessment_density"] = np.clip(np.where(np.isfinite(assessment_density), assessment_density, 0.0), 0.0, 1.0)

    X["engagement_x_score"] = np.clip(np.where(np.isfinite(engagement_x_score), engagement_x_score, 0.0), 0.0, 1e6)
    X["completion_x_score"] = np.clip(np.where(np.isfinite(completion_x_score), completion_x_score, 0.0), 0.0, 1e6)
    X["activity_decay_x_score"] = np.clip(np.where(np.isfinite(activity_decay_x_score), activity_decay_x_score, 0.0), -1e6, 1e6)
    X["late_ratio_x_score"] = np.clip(np.where(np.isfinite(late_ratio_x_score), late_ratio_x_score, 0.0), 0.0, 1e6)
    X["volatility_x_score"] = np.clip(np.where(np.isfinite(volatility_x_score), volatility_x_score, 0.0), 0.0, 1e6)
    X["recovery_x_tail"] = np.clip(np.where(np.isfinite(recovery_x_tail), recovery_x_tail, 0.0), 0.0, 1e6)
    X["consistency_x_tail"] = np.clip(np.where(np.isfinite(consistency_x_tail), consistency_x_tail, 0.0), 0.0, 1e6)

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.attrs["feature_set_version"] = "phase10"
    X.attrs["phase10_features_added"] = phase10_feature_names
    return X


def _build_stage3_features_phase5_ablation(
    df: pd.DataFrame,
    remove_features: list[str],
    feature_set_version: str,
    stage3_plus: bool = False,
) -> pd.DataFrame:
    X = build_stage3_features_phase5(df, stage3_plus=stage3_plus)
    if X.shape[0] == 0:
        X.attrs["feature_set_version"] = str(feature_set_version)
        X.attrs["phase5_ablation_removed"] = list(remove_features)
        return X

    drop_cols = [col for col in remove_features if col in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.attrs["feature_set_version"] = str(feature_set_version)
    X.attrs["phase5_ablation_removed"] = list(remove_features)
    return X


def build_stage3_features_phase5_no_module_features(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    return _build_stage3_features_phase5_ablation(
        df,
        remove_features=[
            "module_top_score_std",
            "module_distinction_hit_ratio",
        ],
        feature_set_version="phase5_no_module_features",
        stage3_plus=stage3_plus,
    )


def build_stage3_features_phase5_no_tail_features(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    return _build_stage3_features_phase5_ablation(
        df,
        remove_features=[
            "exam_top_tail_gap",
            "coursework_top_tail_gap",
            "peak_minus_p90_gap",
        ],
        feature_set_version="phase5_no_tail_features",
        stage3_plus=stage3_plus,
    )


def build_stage3_features_phase5_no_late_features(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    return _build_stage3_features_phase5_ablation(
        df,
        remove_features=[
            "final_third_above_85_ratio",
            "late_vs_mid_high_score_gap",
        ],
        feature_set_version="phase5_no_late_features",
        stage3_plus=stage3_plus,
    )


def build_stage3_features_phase5_no_consistency_features(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    return _build_stage3_features_phase5_ablation(
        df,
        remove_features=[
            "score_above_95_ratio",
            "high_score_streak_90",
            "consistency_above_88",
        ],
        feature_set_version="phase5_no_consistency_features",
        stage3_plus=stage3_plus,
    )


def build_distinction_specialist_features(df: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    X = build_stage3_features_enhanced(df, stage3_plus=stage3_plus)
    if X.empty:
        return X

    if "kuz_score_mean" in X.columns:
        mean_score = X["kuz_score_mean"].astype(float)
        X["score_band_fail"] = (mean_score < 40.0).astype(float)
        X["score_band_pass_low"] = ((mean_score >= 40.0) & (mean_score < 55.0)).astype(float)
        X["score_band_pass_high"] = ((mean_score >= 55.0) & (mean_score < 70.0)).astype(float)
        X["score_band_distinction"] = (mean_score >= 70.0).astype(float)

        X["distinction_margin_strict"] = mean_score - 75.0
        X["pass_upper_margin"] = 70.0 - mean_score
        X["pass_center_distance"] = np.abs(mean_score - 55.0)
        X["distinction_center_distance"] = np.abs(mean_score - 75.0)

        if "kuz_score_max" in X.columns:
            X["ceiling_to_distinction"] = X["kuz_score_max"].astype(float) - 70.0
            X["normalized_ceiling_gap"] = (100.0 - X["kuz_score_max"].astype(float)) / (np.abs(mean_score) + 1e-6)
        if "kuz_score_min" in X.columns:
            X["min_to_pass_floor"] = X["kuz_score_min"].astype(float) - 40.0
        if "kuz_exam_ratio" in X.columns:
            X["exam_ratio_x_score"] = X["kuz_exam_ratio"].astype(float) * mean_score
        if "kuz_tma_ratio" in X.columns:
            X["tma_ratio_x_score"] = X["kuz_tma_ratio"].astype(float) * mean_score
        if "kuz_type_entropy" in X.columns:
            X["entropy_x_score"] = X["kuz_type_entropy"].astype(float) * mean_score
        if "kuz_score_per_assess" in X.columns:
            X["assess_density_x_mean"] = X["kuz_score_per_assess"].astype(float) * mean_score
        if "kuz_score_per_weight_sum" in X.columns:
            X["weight_density_x_mean"] = X["kuz_score_per_weight_sum"].astype(float) * mean_score

    if "kuz_type_entropy" in X.columns and "kuz_score_std" in X.columns:
        X["entropy_x_std"] = X["kuz_type_entropy"].astype(float) * X["kuz_score_std"].astype(float)

    if "kuz_assess_count" in X.columns:
        assess_count = X["kuz_assess_count"].astype(float) + 1e-6
        if "kuz_score_std" in X.columns:
            X["std_per_assess"] = X["kuz_score_std"].astype(float) / assess_count
        if "kuz_score_max" in X.columns and "kuz_score_min" in X.columns:
            X["range_per_assess"] = (X["kuz_score_max"].astype(float) - X["kuz_score_min"].astype(float)) / assess_count

    if "score_band_distinction" in X.columns and "kuz_exam_ratio" in X.columns:
        X["above70_x_exam_ratio"] = X["score_band_distinction"].astype(float) * X["kuz_exam_ratio"].astype(float)
    if "score_band_pass_high" in X.columns and "kuz_tma_ratio" in X.columns:
        X["passhigh_x_tma_ratio"] = X["score_band_pass_high"].astype(float) * X["kuz_tma_ratio"].astype(float)

    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    return X


def augment_stage3_features(X_in: pd.DataFrame, mode: str, stage3_plus: bool = False) -> pd.DataFrame:
    mode_l = str(mode).strip().lower()
    if mode_l == "none":
        return X_in
    if mode_l == "enhanced":
        return build_stage3_features_enhanced(X_in, stage3_plus=stage3_plus)
    return build_stage3_features(X_in, stage3_plus=stage3_plus)


def augment_distinction_specialist_features(X_in: pd.DataFrame, stage3_plus: bool = False) -> pd.DataFrame:
    return build_distinction_specialist_features(X_in, stage3_plus=stage3_plus)


def build_purpose_feature_sets(stage_feature_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        "distinction": stage_feature_sets["stage1"],
        "withdrawn": stage_feature_sets["stage2"],
        "failpass": stage_feature_sets["stage3"],
    }


def print_feature_source_summary(X_raw: pd.DataFrame, X_final: pd.DataFrame) -> None:
    feature_source = str(X_final.attrs.get("feature_source", "unknown"))
    aliases_created = list(X_final.attrs.get("kuz_aliases_created", []))
    print(
        "Feature source summary: "
        f"raw_cols={int(X_raw.shape[1])} "
        f"final_cols={int(X_final.shape[1])} "
        f"locked42={'detected' if feature_source == 'direct_locked42' else 'auto_built'} "
        f"aliases_created={'none' if not aliases_created else ','.join(aliases_created)}"
    )
