from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()

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
LABELS = ["dropout", "enrolled", "graduate"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost 3-class prediction (batch).")
    p.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Input CSV path (default: data/data.csv)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str((OUTPUT_DIR / "models" / "xgboost_model.joblib").resolve()),
        help="Model path (default: outputs/models/xgboost_model.joblib)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str((OUTPUT_DIR / "predictions" / "predictions.csv").resolve()),
        help="Output CSV path (default: outputs/predictions/predictions.csv)",
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

    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    X = df.drop(columns=[target_col]) if target_col else df.copy()

    model_path = Path(args.model).expanduser().resolve()
    clf = joblib.load(model_path)

    proba = clf.predict_proba(X)
    out = pd.DataFrame(
        proba, columns=["proba_dropout", "proba_enrolled", "proba_graduate"]
    )
    out["pred_class_id"] = out[
        ["proba_dropout", "proba_enrolled", "proba_graduate"]
    ].values.argmax(axis=1)
    out["pred_label"] = out["pred_class_id"].map(lambda i: LABELS[int(i)])

    print(out.head(args.head).to_string(index=False))

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
