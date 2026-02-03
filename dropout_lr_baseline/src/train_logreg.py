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
from src.evaluate import compute_metrics, plot_pr_curve, plot_roc_curve
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


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, sep=None, engine="python")

    print("Available columns:", df.columns.tolist())
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")

    y = _validate_and_map_target(df[target_column], target_column)
    X = df.drop(columns=[target_column])
    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("Metrics:")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("Classification Report:")
    print(metrics["classification_report"])

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "metrics.json")
    plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_proba, output_dir / "pr_curve.png")


if __name__ == "__main__":
    main()
