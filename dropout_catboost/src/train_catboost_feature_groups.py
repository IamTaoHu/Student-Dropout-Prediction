from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
RANDOM_STATE = 42
TEST_SIZE = 0.2


def save_json(obj: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path

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


def _map_target(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING))
    if unexpected:
        raise ValueError(f"Unexpected target labels in {column_name}: {unexpected}")
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
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        early_stopping_rounds=200,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test,
        y_pred,
        target_names=["dropout", "enrolled", "graduate"],
        digits=4,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    y = _map_target(df[target_col], target_col)
    X = df.drop(columns=[target_col])

    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]

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
        )
        macro_f1 = metrics["classification_report"]["macro avg"]["f1-score"]
        weighted_f1 = metrics["classification_report"]["weighted avg"]["f1-score"]
        results.append(
            {
                "experiment": experiment_name,
                "groups": groups,
                "num_features": len(feature_columns),
                "features": feature_columns,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    summary_df = results_df[
        ["experiment", "num_features", "accuracy", "macro_f1", "weighted_f1"]
    ].copy()
    summary_df[["accuracy", "macro_f1", "weighted_f1"]] = summary_df[
        ["accuracy", "macro_f1", "weighted_f1"]
    ].astype(float).round(4)

    print("\nCatBoost Feature Group Performance Summary")
    print(summary_df.to_string(index=False))
    print("\nMarkdown table:\n")
    print(summary_df.to_markdown(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT_DIR / "feature_groups" / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(outdir / "summary.csv", index=False)
    save_json(summary_df.to_dict(orient="records"), outdir / "summary.json")
    save_json(results, outdir / "details.json")


if __name__ == "__main__":
    main()
