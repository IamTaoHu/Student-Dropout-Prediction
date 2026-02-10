from __future__ import annotations

import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.metrics import compute_metrics, print_metrics_table
from src.utils import ensure_dir, load_csv, save_json, save_model


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
TARGET_MAPPING_BINARY = {"dropout": 1, "enrolled": 0, "graduate": 0}


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _map_target(series: pd.Series, task: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    normalized = _normalize_target(series)
    mapping = TARGET_MAPPING if task == "multiclass" else TARGET_MAPPING_BINARY
    unexpected = sorted(set(normalized.dropna().unique()) - set(mapping.keys()))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
    return normalized.map(mapping)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(config.DATA_PATH))
    parser.add_argument("--output_dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--test_size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--random_state", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    df = load_csv(args.data_path)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")

    y = _map_target(df[target_column], args.task)
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

    if args.task == "multiclass":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
        )
        labels = [0, 1, 2]
    else:
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
        )
        labels = [0, 1]

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if args.task == "multiclass":
        y_proba = clf.predict_proba(X_test)
    else:
        y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba, labels=labels, task=args.task)
    print_metrics_table(metrics, task=args.task)

    output_dir = ensure_dir(args.output_dir)
    save_model(clf, output_dir / "model.joblib")
    save_json(metrics, output_dir / "metrics.json")


if __name__ == "__main__":
    main()
