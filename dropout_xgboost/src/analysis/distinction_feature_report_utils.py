from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_feature_importance(
    df: pd.DataFrame,
    value_col: str,
    std_col_name: str,
    mean_col_name: str,
    rank_col_name: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["feature", mean_col_name, std_col_name, "n_seeds", rank_col_name])
    agg = (
        df.groupby("feature", as_index=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": mean_col_name,
                "std": std_col_name,
                "count": "n_seeds",
            }
        )
    )
    agg[std_col_name] = agg[std_col_name].fillna(0.0)
    agg = agg.sort_values(mean_col_name, ascending=False).reset_index(drop=True)
    agg[rank_col_name] = np.arange(1, len(agg) + 1, dtype=int)
    return agg


def compute_effect_size(mean_pos: float, std_pos: float, mean_neg: float, std_neg: float) -> float:
    pooled = float(np.sqrt(((std_pos**2) + (std_neg**2)) / 2.0))
    if pooled <= 1e-12:
        return 0.0
    return float(abs(mean_pos - mean_neg) / pooled)


def classify_feature_stability(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    if out.empty:
        out["stability_label"] = []
        return out
    for col in ["gain_mean", "gain_std", "perm_mean", "perm_std"]:
        if col not in out.columns:
            out[col] = 0.0
    out["gain_cv"] = out["gain_std"] / out["gain_mean"].abs().clip(lower=1e-9)
    out["perm_cv"] = out["perm_std"] / out["perm_mean"].abs().clip(lower=1e-9)
    out["stability_label"] = "weak_or_redundant"
    strong_mask = (out["gain_rank"] <= 25) & (out["perm_rank"] <= 25) & (out["gain_cv"] <= 1.0) & (out["perm_cv"] <= 1.0)
    unstable_mask = (out["gain_rank"] <= 30) & ((out["gain_cv"] > 1.0) | (out["perm_cv"] > 1.0))
    out.loc[strong_mask, "stability_label"] = "strong_and_useful"
    out.loc[unstable_mask, "stability_label"] = "noisy_or_unstable"
    return out


def build_candidate_features(top_features: list[str]) -> dict:
    candidate_new_features = [
        {"name": "score_above_95_ratio", "group": "high_score_consistency", "rationale": "Sharper top-end density than >=90 ratio."},
        {"name": "high_score_streak_90", "group": "high_score_consistency", "rationale": "Sustained elite streaks can separate distinction from strong pass."},
        {"name": "exam_top_tail_gap", "group": "exam_vs_coursework", "rationale": "Measures excellence concentration in exam-heavy profiles."},
        {"name": "coursework_top_tail_gap", "group": "exam_vs_coursework", "rationale": "Separates consistent coursework distinction candidates."},
        {"name": "final_third_above_85_ratio", "group": "late_performance", "rationale": "Captures sustained late-stage high performance."},
        {"name": "late_vs_mid_high_score_gap", "group": "late_performance", "rationale": "Detects acceleration into distinction territory."},
        {"name": "module_top_score_std", "group": "module_stability", "rationale": "High achievers tend to sustain quality across modules."},
        {"name": "module_distinction_hit_ratio", "group": "module_stability", "rationale": "Fraction of modules with distinction-level aggregate."},
        {"name": "peak_minus_p90_gap", "group": "peak_vs_average", "rationale": "Disentangles one-off peaks from stable excellence."},
        {"name": "consistency_above_88", "group": "high_score_consistency", "rationale": "Measures robust distinction-like consistency."},
        {"name": "distinction_margin_x_rank", "group": "distinction_interactions", "rationale": "Interaction of margin and relative rank signals."},
        {"name": "exam_coursework_gap_x_streak", "group": "distinction_interactions", "rationale": "Combines excellence gap with sustained high streaks."},
    ]
    recommended_phase5_features = [item["name"] for item in candidate_new_features[:10]]
    recommended_interactions = [
        "score_above_90_ratio * relative_rank_in_module",
        "final_vs_mean_score_gap * high_score_streak",
        "exam_coursework_gap * score_trend_strength",
        "score_above_85_ratio * module_distinction_hit_ratio",
    ]
    return {
        "top_stable_features": top_features[:15],
        "candidate_new_features": candidate_new_features,
        "recommended_phase5_features": recommended_phase5_features,
        "recommended_interactions": recommended_interactions,
    }


def write_candidates_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(
    out_path: Path,
    feature_set: str,
    metrics_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    candidates: dict,
    shap_available: bool,
    shap_path: Path | None,
) -> None:
    lines: list[str] = []
    lines.append("# Distinction Feature Discovery Report")
    lines.append("")
    lines.append("## 1. Analysis scope")
    lines.append("- Objective: identify distinction-vs-pass signals using a binary probe model.")
    lines.append("- Scope: analysis only; no replacement of main flat_xgb architecture.")
    lines.append("")
    lines.append("## 2. Dataset used")
    lines.append(f"- Feature set used: `{feature_set}`")
    lines.append("- Rows filtered to labels: `distinction` and `pass` only.")
    lines.append("- Label mapping: `distinction=1`, `pass=0`.")
    lines.append("")
    lines.append("## 3. Probe-model results across seeds")
    if metrics_df.empty:
        lines.append("- No metrics available.")
    else:
        means = metrics_df[["accuracy", "f1", "roc_auc", "precision", "recall"]].mean()
        stds = metrics_df[["accuracy", "f1", "roc_auc", "precision", "recall"]].std().fillna(0.0)
        lines.append(
            f"- Mean metrics: accuracy={means['accuracy']:.4f}, f1={means['f1']:.4f}, roc_auc={means['roc_auc']:.4f}, "
            f"precision={means['precision']:.4f}, recall={means['recall']:.4f}"
        )
        lines.append(
            f"- Std metrics: accuracy={stds['accuracy']:.4f}, f1={stds['f1']:.4f}, roc_auc={stds['roc_auc']:.4f}, "
            f"precision={stds['precision']:.4f}, recall={stds['recall']:.4f}"
        )
    lines.append("")
    lines.append("## 4. Top features by gain importance")
    for _, row in gain_df.head(15).iterrows():
        lines.append(f"- `{row['feature']}`: gain_mean={row['gain_mean']:.6f}, gain_std={row['gain_std']:.6f}")
    lines.append("")
    lines.append("## 5. Top features by permutation importance")
    for _, row in perm_df.head(15).iterrows():
        lines.append(f"- `{row['feature']}`: perm_mean={row['perm_mean']:.6f}, perm_std={row['perm_std']:.6f}")
    lines.append("")
    lines.append("## 6. Stable features across seeds")
    stable = summary_df[summary_df["stability_label"] == "strong_and_useful"].head(20)
    if stable.empty:
        lines.append("- No strongly stable features identified with current thresholds.")
    else:
        for _, row in stable.iterrows():
            lines.append(
                f"- `{row['feature']}` (gain_rank={int(row['gain_rank'])}, perm_rank={int(row['perm_rank'])}, "
                f"gain_cv={row['gain_cv']:.3f}, perm_cv={row['perm_cv']:.3f})"
            )
    lines.append("")
    lines.append("## 7. Features that separate distinction from pass most clearly")
    sep_df = dist_df.sort_values("effect_size", ascending=False).head(15)
    for _, row in sep_df.iterrows():
        lines.append(
            f"- `{row['feature']}`: effect_size={row['effect_size']:.4f}, "
            f"mean_distinction={row['distinction_mean']:.4f}, mean_pass={row['pass_mean']:.4f}"
        )
    lines.append("")
    lines.append("## 8. Suspected weak or redundant features")
    weak = summary_df[summary_df["stability_label"] == "weak_or_redundant"].sort_values("gain_rank").head(15)
    if weak.empty:
        lines.append("- No weak/redundant group identified.")
    else:
        for _, row in weak.iterrows():
            lines.append(
                f"- `{row['feature']}` (gain_mean={row['gain_mean']:.6f}, perm_mean={row['perm_mean']:.6f})"
            )
    lines.append("")
    lines.append("## 9. Recommended next feature batch")
    for item in candidates.get("candidate_new_features", [])[:15]:
        lines.append(f"- `{item['name']}` ({item['group']}): {item['rationale']}")
    lines.append("")
    lines.append("## 10. Recommended next experiment order")
    lines.append("1. Phase4 + distinction probe threshold study.")
    lines.append("2. Implement top phase5 candidates and run matched flat sweep.")
    lines.append("3. Compare `flat_xgb_phase4` vs `flat_xgb_phase5_candidate` under seeds 42..46.")
    lines.append("4. Re-run confusion analysis focused on distinction->pass reduction.")
    lines.append("")
    lines.append("## SHAP availability")
    if shap_available and shap_path is not None:
        lines.append(f"- SHAP summary was generated at `{shap_path}`.")
    else:
        lines.append("- SHAP was unavailable or failed in this environment; analysis used gain and permutation importance.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

