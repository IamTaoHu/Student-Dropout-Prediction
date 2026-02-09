from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

# Keep mapping consistent with training
LABELS = ["dropout", "enrolled", "graduate"]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
MODEL_PATH = (PROJECT_ROOT / "outputs" / "catboost_model.cbm").resolve()

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CatBoost 3-class prediction (batch).")
    p.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Input CSV path (default: data/data.csv)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Model path (default: outputs/catboost_model.cbm)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str((PROJECT_ROOT / "outputs" / "predictions.csv").resolve()),
        help="Output CSV path (default: outputs/predictions.csv)",
    )
    p.add_argument(
        "--head",
        type=int,
        default=10,
        help="How many rows to print (default: 10)",
    )
    return p.parse_args()


def _detect_target_column(columns: pd.Index) -> str | None:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    return None


def main() -> None:
    args = parse_args()

    # Load data (can be full data or an input file you pass later)
    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    X = df.drop(columns=[target_col]) if target_col else df.copy()

    # Handle categoricals as strings (same logic as training)
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    # Load model
    model = CatBoostClassifier()
    model_path = Path(args.model).expanduser().resolve()
    model.load_model(model_path)

    # Predict probabilities (n_samples, 3)
    proba = model.predict_proba(X)

    # Build a nice output table
    out = pd.DataFrame(proba, columns=[f"proba_{k}" for k in LABELS])
    out["pred_class_id"] = out[[f"proba_{k}" for k in LABELS]].values.argmax(axis=1)
    out["pred_label"] = out["pred_class_id"].map(lambda i: LABELS[int(i)])

    # Print first 10
    print(out.head(args.head).to_string(index=False))

    # Save predictions
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nInput: {input_path}")
    print(f"Model: {model_path}")
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
