from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
RANDOM_STATE = 42
TEST_SIZE = 0.2


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
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
LABELS = ["dropout", "enrolled", "graduate"]


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _map_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING.keys()))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
    return normalized.map(TARGET_MAPPING)


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")

    y = _map_target(df[target_column])
    X = df.drop(columns=[target_column])
    valid = y.notna()
    y = y[valid]
    X = X.loc[valid]
    print("Class distribution after mapping:")
    print(y.value_counts())

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        tree_method="hist",
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
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nXGBoost Metrics (3-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=LABELS,
    )
    plt.title("Confusion Matrix (3-class)")
    plt.show()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=LABELS,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
    }

    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "models" / "xgboost_model.joblib"
    joblib.dump(clf, model_path)

    metrics_path = OUTPUT_DIR / "metrics" / "metrics.json"
    save_json(metrics, metrics_path)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
