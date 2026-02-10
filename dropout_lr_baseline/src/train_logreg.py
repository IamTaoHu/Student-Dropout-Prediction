from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def _normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _map_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    normalized = _normalize_target(series)
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING.keys()))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
    return normalized.map(TARGET_MAPPING)


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    if len(df) == 0:
        raise ValueError(
            f"DATA_PATH is empty: {DATA_PATH}. "
            "This file contains no rows and cannot be used for training."
        )
    if "Target" not in df.columns:
        raise ValueError(
            "Training requires a 'Target' column, but none was found. "
            f"Columns: {df.columns.tolist()}"
        )
    if df["Target"].notna().sum() == 0:
        raise ValueError(
            "The 'Target' column contains no labels (all NaN). "
            "You are likely using a prediction-only dataset. "
            "Please provide a training dataset with labels."
        )
    print(f"Reading DATA_PATH: {DATA_PATH}")
    print(f"DataFrame shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Target head (raw):", df["Target"].head(20).tolist())
    print("Target non-null count:", int(df["Target"].notna().sum()))
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")
    raw_unique = sorted(df[target_column].dropna().unique().tolist())
    print(f"Raw unique values in {target_column}: {raw_unique[:50]}")
    normalized_preview = (
        df[target_column].astype(str).str.strip().str.lower().dropna().unique().tolist()
    )
    normalized_preview = sorted(normalized_preview)
    print(f"Normalized unique values: {normalized_preview[:50]}")
    print(f"Expected labels: {sorted(list(TARGET_MAPPING.keys()))}")

    y = _map_target(df[target_column])
    X = df.drop(columns=[target_column])
    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())
    if len(y) == 0:
        raise ValueError(
            "After applying TARGET_MAPPING, 0 rows remain. "
            "Your target labels do not match TARGET_MAPPING. "
            "Check the 'Raw unique values' / 'Normalized unique values' printed above "
            "and update TARGET_MAPPING accordingly."
        )

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
        solver="lbfgs",
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
    y_proba = clf.predict_proba(X_test)

    print("\nLogReg Metrics (3-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=LABELS,
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=LABELS,
    )
    plt.title("Confusion Matrix (3-class)")
    plt.show()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=LABELS,
            digits=4,
            output_dict=True,
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "logreg_model.joblib"
    joblib.dump(clf, model_path)
    save_json(metrics, OUTPUT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
