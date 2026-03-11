from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.distinction_feature_report_utils import (
    aggregate_feature_importance,
    build_candidate_features,
    classify_feature_stability,
    compute_effect_size,
    write_candidates_json,
    write_report,
)
from config.paths import OUTPUT_DIR, PROJECT_ROOT
from models.flat_xgb.train import DATA_PATH, _detect_target_column, _infer_label_mapping, _prepare_features


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distinction-vs-pass feature discovery for flat_xgb pipeline.")
    p.add_argument("--input", type=str, default=str(DATA_PATH), help="Input CSV path.")
    p.add_argument("--feature_set", type=str, default="phase4", choices=["existing", "phase3", "phase4"], help="Flat-compatible feature set.")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated seed list.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation split fraction inside train.")
    p.add_argument("--n_estimators", type=int, default=400, help="Probe model n_estimators.")
    p.add_argument("--learning_rate", type=float, default=0.05, help="Probe model learning_rate.")
    p.add_argument("--max_depth", type=int, default=6, help="Probe model max_depth.")
    p.add_argument("--subsample", type=float, default=0.8, help="Probe model subsample.")
    p.add_argument("--colsample_bytree", type=float, default=0.8, help="Probe model colsample_bytree.")
    p.add_argument("--min_child_weight", type=float, default=3.0, help="Probe model min_child_weight.")
    p.add_argument("--reg_lambda", type=float, default=2.0, help="Probe model reg_lambda.")
    p.add_argument("--reg_alpha", type=float, default=0.0, help="Probe model reg_alpha.")
    p.add_argument("--gamma", type=float, default=0.0, help="Probe model gamma.")
    p.add_argument("--early_stopping_rounds", type=int, default=80, help="Probe model early stopping rounds.")
    p.add_argument("--perm_repeats", type=int, default=5, help="Permutation importance repeats.")
    p.add_argument("--top_n", type=int, default=20, help="Top-N features for distribution summary.")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers.")
    p.add_argument("--output_dir", type=str, default=str((OUTPUT_DIR / "analysis").resolve()), help="Output analysis directory.")
    return p


def _parse_seed_list(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    seeds = [int(item) for item in values]
    if not seeds:
        raise ValueError("No seeds provided.")
    return seeds


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def _build_distinction_dataset(input_path: Path, feature_set: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    target_column = _detect_target_column(df.columns)
    y_raw = df[target_column]
    y_map, _, labels = _infer_label_mapping(y_raw)
    if "distinction" not in labels or "pass" not in labels:
        raise ValueError(f"Required labels not found in target. Observed labels={labels}")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    valid_mask = y_map.notna()
    y_valid = y_map.loc[valid_mask].reset_index(drop=True)
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    keep_mask = y_valid.isin([label_to_id["distinction"], label_to_id["pass"]])
    df_subset = df_valid.loc[keep_mask].reset_index(drop=True)
    y_subset = y_valid.loc[keep_mask].reset_index(drop=True)
    y_bin = (y_subset == label_to_id["distinction"]).astype(int).reset_index(drop=True)
    X, _ = _prepare_features(df_subset, target_column, feature_set=str(feature_set))
    X = X.astype(float)
    return X, y_bin


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser().resolve()
    seeds = _parse_seed_list(args.seeds)
    X, y = _build_distinction_dataset(input_path, feature_set=str(args.feature_set))

    metrics_rows: list[dict] = []
    gain_seed_rows: list[dict] = []
    perm_seed_rows: list[dict] = []
    shap_seed_rows: list[dict] = []
    shap_available = False
    shap_path: Path | None = None

    for seed in seeds:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=float(args.test_size),
            random_state=int(seed),
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=float(args.val_size),
            random_state=int(seed),
            stratify=y_train_full,
        )
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            min_child_weight=float(args.min_child_weight),
            reg_lambda=float(args.reg_lambda),
            reg_alpha=float(args.reg_alpha),
            gamma=float(args.gamma),
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=int(args.early_stopping_rounds),
            n_jobs=int(args.n_jobs),
            random_state=int(seed),
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics_rows.append(
            {
                "seed": int(seed),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": _safe_auc(y_test.to_numpy(dtype=int), y_prob),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            }
        )

        booster = model.get_booster()
        gain_map = booster.get_score(importance_type="gain")
        weight_map = booster.get_score(importance_type="weight")
        for feature in X.columns:
            gain_seed_rows.append(
                {
                    "seed": int(seed),
                    "feature": feature,
                    "gain": float(gain_map.get(feature, 0.0)),
                    "weight": float(weight_map.get(feature, 0.0)),
                }
            )

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=int(args.perm_repeats),
            random_state=int(seed),
            n_jobs=int(args.n_jobs),
        )
        for idx, feature in enumerate(X.columns):
            perm_seed_rows.append(
                {
                    "seed": int(seed),
                    "feature": feature,
                    "perm_mean": float(perm.importances_mean[idx]),
                    "perm_std": float(perm.importances_std[idx]),
                }
            )

        try:
            import shap  # type: ignore

            shap_available = True
            sample_n = int(min(len(X_test), 1200))
            X_shap = X_test.sample(n=sample_n, random_state=int(seed)) if len(X_test) > sample_n else X_test
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_arr = np.asarray(shap_values[-1], dtype=float)
            else:
                shap_arr = np.asarray(shap_values, dtype=float)
            mean_abs = np.abs(shap_arr).mean(axis=0)
            for idx, feature in enumerate(X.columns):
                shap_seed_rows.append({"seed": int(seed), "feature": feature, "shap_mean_abs": float(mean_abs[idx])})
        except Exception:
            # SHAP is optional for this analysis.
            pass

    metrics_df = pd.DataFrame(metrics_rows)
    gain_seed_df = pd.DataFrame(gain_seed_rows)
    perm_seed_df = pd.DataFrame(perm_seed_rows)
    shap_seed_df = pd.DataFrame(shap_seed_rows)

    gain_df = aggregate_feature_importance(gain_seed_df, "gain", "gain_std", "gain_mean", "gain_rank")
    weight_df = aggregate_feature_importance(gain_seed_df, "weight", "weight_std", "weight_mean", "weight_rank")
    gain_df = gain_df.merge(weight_df[["feature", "weight_mean", "weight_std"]], on="feature", how="left")
    perm_df = aggregate_feature_importance(perm_seed_df, "perm_mean", "perm_std", "perm_mean", "perm_rank")

    summary_df = gain_df.merge(perm_df[["feature", "perm_mean", "perm_std", "perm_rank"]], on="feature", how="outer").fillna(0.0)
    if not shap_seed_df.empty:
        shap_summary = aggregate_feature_importance(shap_seed_df, "shap_mean_abs", "shap_std", "shap_mean_abs", "shap_rank")
        summary_df = summary_df.merge(shap_summary[["feature", "shap_mean_abs", "shap_std", "shap_rank"]], on="feature", how="left")
        shap_path = output_dir / "distinction_shap_summary.csv"
        shap_summary.to_csv(shap_path, index=False, encoding="utf-8")
    summary_df = classify_feature_stability(summary_df)
    summary_df = summary_df.sort_values(["gain_rank", "perm_rank"], ascending=[True, True]).reset_index(drop=True)

    top_features = (
        summary_df.sort_values(["gain_rank", "perm_rank"], ascending=[True, True])["feature"].astype(str).head(int(args.top_n)).tolist()
    )

    dist_rows: list[dict] = []
    mask_dist = y == 1
    mask_pass = y == 0
    for feature in top_features:
        s_dist = pd.to_numeric(X.loc[mask_dist, feature], errors="coerce").fillna(0.0)
        s_pass = pd.to_numeric(X.loc[mask_pass, feature], errors="coerce").fillna(0.0)
        row = {
            "feature": feature,
            "distinction_mean": float(s_dist.mean()),
            "distinction_median": float(s_dist.median()),
            "distinction_std": float(s_dist.std(ddof=0)),
            "distinction_min": float(s_dist.min()),
            "distinction_max": float(s_dist.max()),
            "pass_mean": float(s_pass.mean()),
            "pass_median": float(s_pass.median()),
            "pass_std": float(s_pass.std(ddof=0)),
            "pass_min": float(s_pass.min()),
            "pass_max": float(s_pass.max()),
        }
        row["effect_size"] = compute_effect_size(
            mean_pos=row["distinction_mean"],
            std_pos=row["distinction_std"],
            mean_neg=row["pass_mean"],
            std_neg=row["pass_std"],
        )
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows).sort_values("effect_size", ascending=False).reset_index(drop=True)

    candidates = build_candidate_features(
        top_features=summary_df[summary_df["stability_label"] == "strong_and_useful"]["feature"].astype(str).head(25).tolist()
    )
    candidates["top_separating_features"] = dist_df["feature"].astype(str).head(15).tolist() if not dist_df.empty else []
    candidates["unstable_features"] = (
        summary_df[summary_df["stability_label"] == "noisy_or_unstable"]["feature"].astype(str).head(20).tolist()
    )

    metrics_path = output_dir / "distinction_probe_seed_metrics.csv"
    gain_path = output_dir / "distinction_feature_importance_gain.csv"
    perm_path = output_dir / "distinction_feature_importance_permutation.csv"
    summary_path = output_dir / "distinction_feature_importance_summary.csv"
    dist_path = output_dir / "distinction_feature_distribution_summary.csv"
    report_path = output_dir / "distinction_feature_discovery_report.md"
    candidates_path = output_dir / "distinction_feature_candidates.json"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    gain_df.to_csv(gain_path, index=False, encoding="utf-8")
    perm_df.to_csv(perm_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    dist_df.to_csv(dist_path, index=False, encoding="utf-8")
    write_candidates_json(candidates, candidates_path)
    write_report(
        out_path=report_path,
        feature_set=str(args.feature_set),
        metrics_df=metrics_df,
        gain_df=gain_df,
        perm_df=perm_df,
        summary_df=summary_df,
        dist_df=dist_df,
        candidates=candidates,
        shap_available=shap_available and shap_path is not None,
        shap_path=shap_path,
    )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved gain importance to: {gain_path}")
    print(f"Saved permutation importance to: {perm_path}")
    print(f"Saved importance summary to: {summary_path}")
    print(f"Saved distribution summary to: {dist_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved candidates to: {candidates_path}")
    if shap_available and shap_path is not None:
        print(f"Saved SHAP summary to: {shap_path}")
    else:
        print("SHAP summary unavailable in this environment; skipped.")


if __name__ == "__main__":
    main()

