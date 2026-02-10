from __future__ import annotations

import warnings

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import *
from utils import load_csv, save_json

warnings.filterwarnings("ignore", message="X does not have valid feature names*")

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
TARGET_MAPPING = {"dropout": 0, "enrolled": 1, "graduate": 2}

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


def _normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


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
    experiment_name: str,
) -> dict:
    X_selected = X[feature_columns]
    num_features = len(feature_columns)
    numeric_features = X_selected.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = X_selected.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_train = X_selected.loc[train_idx]
    X_test = X_selected.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_train_trans = preprocessor.fit_transform(X_train_sub)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )

    model.fit(
        X_train_trans,
        y_train_sub,
        eval_set=[(X_val_trans, y_val)],
        callbacks=[early_stopping(stopping_rounds=50)],
    )

    y_pred = model.predict(X_test_trans)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "num_features": int(num_features),
    }
    return metrics


def main() -> None:
    df = load_csv(DATA_PATH)
    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")

    raw_unique = sorted(df[target_column].dropna().unique().tolist())
    print(f"Raw unique values in Target: {raw_unique}")

    normalized = _normalize_target(df[target_column])
    normalized_unique = sorted(normalized.dropna().unique().tolist())
    print(f"Normalized unique values: {normalized_unique}")

    y = normalized.map(TARGET_MAPPING)
    X = df.drop(columns=[target_column])
    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())

    group_columns = _build_group_columns(X.columns)

    train_idx, test_idx = train_test_split(
        X.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: list[dict] = []
    for experiment_name, groups in EXPERIMENTS:
        feature_columns = _select_features(group_columns, groups, experiment_name)
        metrics = _run_experiment(
            X,
            y,
            train_idx,
            test_idx,
            feature_columns,
            experiment_name,
        )
        results.append(
            {
                "experiment": experiment_name,
                "groups": groups,
                "num_features": metrics["num_features"],
                "features": feature_columns,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "model_name": "LightGBM",
            }
        )

    results_df = pd.DataFrame(results)
    summary_df = results_df[
        ["experiment", "num_features", "accuracy", "macro_f1", "weighted_f1"]
    ].copy()
    summary_df[["accuracy", "macro_f1", "weighted_f1"]] = summary_df[
        ["accuracy", "macro_f1", "weighted_f1"]
    ].astype(float).round(4)

    print()
    print("LightGBM Feature Group Performance Summary")
    print(summary_df.to_string(index=False))
    print()
    print("Markdown table:")
    print(summary_df.to_markdown(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "lightgbm_feature_group_summary.csv", index=False)
    summary_df.to_json(
        OUTPUT_DIR / "lightgbm_feature_group_summary.json",
        orient="records",
        indent=2,
    )
    save_json(OUTPUT_DIR / "feature_group_results.json", results)
    results_df.to_csv(OUTPUT_DIR / "feature_group_results.csv", index=False)


if __name__ == "__main__":
    main()
