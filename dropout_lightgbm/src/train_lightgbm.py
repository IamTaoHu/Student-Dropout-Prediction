from __future__ import annotations

import pickle

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import *
from utils import load_csv, save_json

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
SHOW_CM_PLOT = True


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


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
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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
    )

    model.fit(
        X_train_trans,
        y_train_sub,
        eval_set=[(X_val_trans, y_val)],
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)],
    )

    y_pred = model.predict(X_test_trans)

    print("LightGBM Metrics (3-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["dropout", "enrolled", "graduate"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    if SHOW_CM_PLOT:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["dropout", "enrolled", "graduate"],
        )

        disp.plot(cmap="viridis", values_format="d")
        plt.title("Confusion Matrix (3-class)")
        plt.show()

    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["dropout", "enrolled", "graduate"],
            digits=4,
        ),
    }

    trained_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    model_path = OUTPUT_DIR / "lightgbm_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(trained_pipeline, handle)
    print(f"Saved model to: {model_path.resolve()}")

    save_json(OUTPUT_DIR / "metrics.json", metrics)


if __name__ == "__main__":
    main()
