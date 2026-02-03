from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from src import config
from src.evaluate import compute_metrics, plot_pr_curve, plot_roc_curve
from src.utils import save_json

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
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING))
    if unexpected:
        raise ValueError(f"Unexpected target labels in {column_name}: {unexpected}")
    return normalized.map(TARGET_MAPPING)


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, sep=None, engine="python")
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
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=5000,
        learning_rate=0.05,
        depth=6,
        random_seed=config.RANDOM_STATE,
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
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["accuracy"] = float(accuracy)
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    outdir = Path(config.OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, outdir / "metrics.json")
    plot_roc_curve(y_test, y_proba, outdir / "roc_curve.png")
    plot_pr_curve(y_test, y_proba, outdir / "pr_curve.png")

    model_path = outdir / "catboost_model.cbm"
    model.save_model(model_path)

    print("CatBoost Metrics:")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(f"[[{tn} {fp}]")
    print(f" [{fn} {tp}]]")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
