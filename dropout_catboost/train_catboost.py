from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --- Minimal config (local folder defaults) ---
PROJECT_ROOT = Path(__file__).resolve().parent
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
TARGET_MAPPING = {
    "dropout": 0,
    "enrolled": 1,
    "graduate": 2,
}


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _validate_and_map_target(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING))
    if unexpected:
        raise ValueError(f"Unexpected target labels in {column_name}: {unexpected}")
    return normalized.map(TARGET_MAPPING)


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    print(f"Detected target column: {target_col}")
    raw_unique = sorted(df[target_col].dropna().unique().tolist())
    print(f"Raw unique values in {target_col}: {raw_unique}")
    normalized = df[target_col].astype(str).str.strip().str.lower()
    normalized_unique = sorted(normalized.dropna().unique().tolist())
    print("Normalized unique values:", normalized_unique)
    y = _validate_and_map_target(df[target_col], target_col)
    X = df.drop(columns=[target_col])
    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())

    # Detect categorical columns (non-numeric)
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Make categoricals explicit strings for safety
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=5000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        verbose=200,
        early_stopping_rounds=200,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["dropout", "enrolled", "graduate"],
        digits=4,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report_dict,
    }

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, outdir / "metrics.json")

    model_path = outdir / "catboost_model.cbm"
    model.save_model(model_path)

    print("\nCatBoost Metrics (3-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["dropout", "enrolled", "graduate"],
            digits=4,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["dropout", "enrolled", "graduate"],
    )
    plt.title("Confusion Matrix (3-class)")
    plt.show()
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
