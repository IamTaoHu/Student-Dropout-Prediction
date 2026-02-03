from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.evaluate import compute_metrics, plot_pr_curve, plot_roc_curve
from src.utils import load_csv, save_json

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


def _normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def main() -> None:
    df = load_csv(Path(config.DATA_PATH))
    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")

    raw_unique = sorted(df[target_column].dropna().unique().tolist())
    print(f"Raw unique values in {target_column}: {raw_unique}")

    normalized = _normalize_target(df[target_column])
    normalized_unique = sorted(normalized.dropna().unique().tolist())
    print("Normalized target values:", normalized_unique)

    y = normalized.map(TARGET_MAPPING)
    X = df.drop(columns=[target_column])
    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())

    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
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
    )

    model.fit(
        X_train_trans,
        y_train_sub,
        eval_set=[(X_val_trans, y_val)],
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)],
    )

    y_proba = model.predict_proba(X_test_trans)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["num_features"] = int(X_test_trans.shape[1])
    metrics["model_name"] = "LightGBM"

    best_iter = getattr(model, "best_iteration_", None)

    print("Metrics:")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Best iteration: {best_iter}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("Classification Report:")
    print(metrics["classification_report"])

    output_dir = Path(config.OUTPUT_DIR)
    save_json(output_dir / "metrics.json", metrics)
    plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_proba, output_dir / "pr_curve.png")


if __name__ == "__main__":
    main()
