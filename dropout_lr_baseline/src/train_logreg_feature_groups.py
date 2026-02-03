from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.evaluate import compute_metrics
from src.utils import save_json

TARGET_COLUMN = "Target"
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
    (
        "D_G1_G2_G3_G4",
        ["G1_demographic", "G2_admission_program", "G3_financial_admin", "G4_academic_semesters"],
    ),
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


def _validate_and_map_target(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unique_values = sorted(series.dropna().unique().tolist())
    print(f"Unique values in {column_name}: {unique_values}")
    print(
        "Unique target values (normalized):",
        sorted(normalized.dropna().unique().tolist()),
    )
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
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
) -> dict:
    X_selected = X[feature_columns]
    numeric_features = X_selected.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_selected.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train = X_selected.loc[train_idx]
    X_test = X_selected.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_proba)


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    target_column = _detect_target_column(df.columns)

    y = _validate_and_map_target(df[target_column], target_column)
    X = df.drop(columns=[target_column])
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
    for experiment_name, groups in EXPERIMENTS:
        feature_columns = _select_features(group_columns, groups, experiment_name)
        metrics = _run_experiment(X, y, train_idx, test_idx, feature_columns)
        results.append(
            {
                "experiment": experiment_name,
                "groups": groups,
                "num_features": len(feature_columns),
                "features": feature_columns,
                **metrics,
            }
        )

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
    existing_cols = [c for c in summary_columns if c in results_df.columns]
    summary_df = results_df[existing_cols].copy()
    metric_cols = ["f1", "recall", "roc_auc", "pr_auc", "accuracy"]
    metric_cols = [c for c in metric_cols if c in summary_df.columns]
    summary_df[metric_cols] = summary_df[metric_cols].astype(float).round(4)

    print("\nLogistic Regression Feature Group Performance Summary")
    print(summary_df.to_string(index=False))
    print("\nMarkdown table:\n")
    try:
        print(summary_df.to_markdown(index=False))
    except ImportError:
        print("[WARN] 'tabulate' not installed. Falling back to plain text table.")
        print(summary_df.to_string(index=False))

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "logreg_feature_group_summary.csv", index=False)
    summary_df.to_json(
        output_dir / "logreg_feature_group_summary.json",
        orient="records",
        indent=2,
    )
    save_json(results, output_dir / "feature_group_results.json")
    results_df.to_csv(output_dir / "feature_group_results.csv", index=False)


if __name__ == "__main__":
    main()
