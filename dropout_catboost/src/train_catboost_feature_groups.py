from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src import config
from src.evaluate import compute_metrics, plot_pr_curve, plot_roc_curve
from src.utils import save_json

TARGET_CANDIDATES = [
    "Target",
    "target",
    "STATUS",
    "Status",
    "status",
    "Outcome",
    "outcome",
    "Class",
    "class",
]
TARGET_MAPPING = {"dropout": 1, "enrolled": 0, "graduate": 0}

GROUP_PATTERNS = {
    "G1_demographic": [
        "Gender",
        "Age at enrollment",
        "Nationality",
        "Marital status",
        "Displaced",
        "Educational special needs",
    ],
    "G2_admission_program": [
        "Course",
        "Application mode",
        "Application order",
        "Previous qualification",
        "Admission grade",
        "Daytime/evening attendance",
    ],
    "G3_financial_admin": [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ],
    "G4_academic_semesters": [
        "1st sem",
        "2nd sem",
        "Curricular units",
    ],
}

EXPERIMENTS = [
    ("A_G1", ["G1_demographic"]),
    ("B_G1_G2", ["G1_demographic", "G2_admission_program"]),
    ("C_G1_G2_G3", ["G1_demographic", "G2_admission_program", "G3_financial_admin"]),
    ("D_Full", ["G1_demographic", "G2_admission_program", "G3_financial_admin", "G4_academic_semesters"]),
    ("Full", ["G1_demographic", "G2_admission_program", "G3_financial_admin", "G4_academic_semesters"]),
    ("w/o_G1", ["G2_admission_program", "G3_financial_admin", "G4_academic_semesters"]),
    ("w/o_G2", ["G1_demographic", "G3_financial_admin", "G4_academic_semesters"]),
    ("w/o_G3", ["G1_demographic", "G2_admission_program", "G4_academic_semesters"]),
    ("w/o_G4", ["G1_demographic", "G2_admission_program", "G3_financial_admin"]),
]


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _map_target(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(TARGET_MAPPING)


def _build_group_columns(columns: pd.Index) -> dict[str, list[str]]:
    group_columns: dict[str, list[str]] = {}
    lowered = [col.lower() for col in columns]
    for group_name, patterns in GROUP_PATTERNS.items():
        group_patterns = [pattern.lower() for pattern in patterns]
        matches = [
            columns[idx]
            for idx, column_name in enumerate(lowered)
            if any(pattern in column_name for pattern in group_patterns)
        ]
        group_columns[group_name] = matches
    return group_columns


def _select_features(
    group_columns: dict[str, list[str]],
    groups: list[str],
    experiment_name: str,
) -> list[str]:
    selected: list[str] = []
    for group in groups:
        columns = group_columns.get(group, [])
        if not columns:
            print(f"Warning: {group} has 0 columns; skipping.")
            continue
        for column in columns:
            if column not in selected:
                selected.append(column)
    if not selected:
        raise ValueError(
            f"Feature set is empty for experiment '{experiment_name}'. "
            "Check group definitions and input data."
        )
    return selected


def _run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    test_idx: pd.Index,
    feature_columns: list[str],
) -> tuple[dict, pd.Series, pd.Series]:
    X_selected = X[feature_columns].copy()

    cat_cols = X_selected.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X_selected[col] = X_selected[col].astype(str).fillna("NA")

    X_train = X_selected.loc[train_idx]
    X_test = X_selected.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update(
        {
            "accuracy": float(acc),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    return metrics, y_test, y_proba


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    y = _map_target(df[target_col])
    X = df.drop(columns=[target_col])

    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]

    group_columns = _build_group_columns(X.columns)

    train_idx, test_idx = train_test_split(
        X.index,
        test_size=config.TEST_SIZE,
        random_state=42,
        stratify=y,
    )

    results: list[dict] = []
    full_curve_data: tuple[pd.Series, pd.Series] | None = None
    for experiment_name, groups in EXPERIMENTS:
        feature_columns = _select_features(group_columns, groups, experiment_name)
        metrics, y_test, y_proba = _run_experiment(
            X,
            y,
            train_idx,
            test_idx,
            feature_columns,
        )
        results.append(
            {
                "experiment": experiment_name,
                "groups": groups,
                "num_features": len(feature_columns),
                "features": feature_columns,
                **metrics,
            }
        )
        if experiment_name == "Full":
            full_curve_data = (y_test, y_proba)

    results_df = pd.DataFrame(results)
    summary_columns = [
        "experiment",
        "num_features",
        "f1",
        "recall",
        "roc_auc",
        "pr_auc",
        "accuracy",
    ]
    if "accuracy" not in results_df.columns:
        print(
            "[WARN] accuracy column missing. Available columns:",
            results_df.columns.tolist(),
        )
    existing_cols = [c for c in summary_columns if c in results_df.columns]
    summary_df = results_df[existing_cols].copy()
    metric_cols = [c for c in ["f1", "recall", "roc_auc", "pr_auc", "accuracy"] if c in summary_df.columns]
    summary_df[metric_cols] = summary_df[metric_cols].astype(float).round(4)

    print("\nCatBoost Feature Group Performance Summary")
    print(summary_df.to_string(index=False))
    print("\nMarkdown table:\n")
    print(summary_df.to_markdown(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_DIR) / "catboost_feature_groups" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_output_dir = Path(config.OUTPUT_DIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(base_output_dir / "catboost_feature_group_summary.csv", index=False)
    summary_df.to_json(
        base_output_dir / "catboost_feature_group_summary.json",
        orient="records",
        indent=2,
    )

    save_json(results, output_dir / "feature_group_results.json")
    results_df.to_csv(output_dir / "feature_group_results.csv", index=False)

    if full_curve_data is not None:
        y_test, y_proba = full_curve_data
        plot_roc_curve(y_test, y_proba, output_dir / "full_roc_curve.png")
        plot_pr_curve(y_test, y_proba, output_dir / "full_pr_curve.png")


if __name__ == "__main__":
    main()
