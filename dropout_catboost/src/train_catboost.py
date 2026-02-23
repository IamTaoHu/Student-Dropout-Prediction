from __future__ import annotations

import argparse
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

# --- config (local folder defaults) ---
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
    "final_result",
    "STATUS",
    "Status",
    "status",
    "Outcome",
    "outcome",
    "Class",
    "class",
]
DEFAULT_3CLASS_MAPPING = {"dropout": 0, "enrolled": 1, "graduate": 2}


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _build_label_mapping(normalized_labels: pd.Series) -> dict[str, int]:
    labels = sorted(set(normalized_labels.dropna().unique()))
    if set(labels) == set(DEFAULT_3CLASS_MAPPING.keys()):
        return DEFAULT_3CLASS_MAPPING
    return {lab: i for i, lab in enumerate(labels)}


def _map_target(series: pd.Series, column_name: str) -> tuple[pd.Series, dict[str, int]]:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = _build_label_mapping(normalized)

    unexpected = sorted(set(normalized.dropna().unique()) - set(mapping.keys()))
    if unexpected:
        raise ValueError(f"Unexpected target labels in {column_name}: {unexpected}")

    y = normalized.map(mapping)
    return y, mapping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(DATA_PATH),
        help="Path to input CSV (default: data/data.csv).",
    )

    # ---- training objective / class imbalance ----
    p.add_argument(
        "--loss",
        choices=["multiclass", "onevsall"],
        default="multiclass",
        help="Loss function: 'multiclass' or 'onevsall' (MultiClassOneVsAll).",
    )
    p.add_argument(
        "--auto_class_weights",
        choices=["None", "Balanced"],
        default="Balanced",
        help="Auto class weights in CatBoost: Balanced or None.",
    )
    p.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Custom class weights as comma-separated floats, e.g. 3.5,3.0,1.0,2.0 (order must match label mapping).",
    )

    # ---- core hyperparams ----
    p.add_argument("--iterations", type=int, default=5000, help="Training iterations.")
    p.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate.")
    p.add_argument("--depth", type=int, default=6, help="Tree depth.")
    p.add_argument("--l2_leaf_reg", type=float, default=3.0, help="L2 leaf regularization.")
    p.add_argument("--random_strength", type=float, default=1.0, help="Random strength (regularization).")
    p.add_argument("--bagging_temperature", type=float, default=1.0, help="Bagging temperature.")

    # ---- train control ----
    p.add_argument("--test_size", type=float, default=TEST_SIZE, help="Test split size.")
    p.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    p.add_argument("--early_stopping_rounds", type=int, default=200, help="Early stopping rounds.")
    p.add_argument("--verbose", type=int, default=200, help="Verbose frequency.")
    p.add_argument("--no_plot", action="store_true", help="Disable confusion matrix plot window.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.input).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    print(f"Detected target column: {target_col}")
    raw_unique = sorted(df[target_col].dropna().unique().tolist())
    print(f"Raw unique values in {target_col}: {raw_unique}")
    normalized = df[target_col].astype(str).str.strip().str.lower()
    normalized_unique = sorted(normalized.dropna().unique().tolist())
    print("Normalized unique values:", normalized_unique)
    y, label_mapping = _map_target(df[target_col], target_col)
    X = df.drop(columns=[target_col])

    inv = {v: k for k, v in label_mapping.items()}
    label_names = [inv[i] for i in range(len(inv))]

    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())
    print("Label mapping:", label_mapping)

    # Detect categorical columns (non-numeric)
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Make categoricals explicit strings for safety
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols]

    loss_fn = "MultiClassOneVsAll" if args.loss == "onevsall" else "MultiClass"
    auto_w = None if args.auto_class_weights == "None" else args.auto_class_weights
    eval_metric = "MultiClassOneVsAll" if args.loss == "onevsall" else "MultiClass"
    custom_weights = None
    if args.class_weights is not None:
        try:
            custom_weights = [float(x.strip()) for x in args.class_weights.split(",")]
            print(f"Using custom class_weights: {custom_weights}")
        except Exception as e:
            raise ValueError(f"Invalid --class_weights format: {args.class_weights}") from e

    model = CatBoostClassifier(
        loss_function=loss_fn,
        eval_metric="Accuracy",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        random_seed=args.seed,
        class_weights=custom_weights if custom_weights is not None else None,
        auto_class_weights=None if custom_weights is not None else auto_w,
        verbose=args.verbose,
        early_stopping_rounds=args.early_stopping_rounds,
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
        target_names=label_names,
        digits=4,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report_dict,
        "label_mapping": label_mapping,
        "label_names": label_names,
    }

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, outdir / "metrics.json")

    model_path = outdir / "catboost_model.cbm"
    model.save_model(model_path)

    print(f"\nCatBoost Metrics ({len(label_names)}-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            digits=4,
        )
    )

    if not args.no_plot:
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=label_names,
        )
        plt.title(f"Confusion Matrix ({len(label_names)}-class)")
        plt.show()
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
