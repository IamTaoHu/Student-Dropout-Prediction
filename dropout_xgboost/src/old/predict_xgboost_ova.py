from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
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
    "final_result",
    "FinalResult",
    "label",
    "Label",
    "y",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with calibrated OVA XGBoost model bundle.")
    p.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Input CSV path",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str((OUTPUT_DIR / "models" / "xgboost_ova_calibrated.joblib").resolve()),
        help="Model bundle path",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str((OUTPUT_DIR / "predictions" / "predictions_ova.csv").resolve()),
        help="Output CSV path",
    )
    p.add_argument("--topk", type=int, default=4, help="Number of probability columns to print")
    p.add_argument(
        "--drop_cols",
        type=str,
        default="",
        help="Comma-separated columns to drop before prediction",
    )
    p.add_argument("--no_print", action="store_true", default=False)
    return p.parse_args()


def _detect_target_column(columns: pd.Index) -> str | None:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    return None


def ova_predict_proba(models: list, X) -> np.ndarray:
    proba_list = []
    for i in range(len(models)):
        p_pos = models[i].predict_proba(X)[:, 1]
        proba_list.append(p_pos)

    proba = np.column_stack(proba_list)
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = proba / np.clip(row_sum, 1e-12, None)
    return proba


def _parse_drop_cols(raw: str) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    X = df.drop(columns=[target_col]) if target_col else df.copy()

    drop_cols = _parse_drop_cols(args.drop_cols)
    if drop_cols:
        existing = [c for c in drop_cols if c in X.columns]
        missing = [c for c in drop_cols if c not in X.columns]
        if existing:
            X = X.drop(columns=existing)
        if missing:
            print(f"Warning: drop_cols not found and ignored: {missing}")

    bundle_path = Path(args.model).expanduser().resolve()
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle: expected dict.")

    labels = bundle.get("labels")
    models = bundle.get("models")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("Invalid model bundle: missing valid 'labels' list.")
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("Invalid model bundle: missing non-empty 'models' list.")

    proba = ova_predict_proba(models, X)
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"Model output classes mismatch: proba has {proba.shape[1]} columns, labels has {len(labels)}."
        )

    pred_id = proba.argmax(axis=1)
    pred_label = [labels[int(i)] for i in pred_id]

    proba_cols = [f"proba_{label}" for label in labels]
    out = pd.DataFrame(proba, columns=proba_cols)
    out["pred_class_id"] = pred_id
    out["pred_label"] = pred_label

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    if not args.no_print:
        k = max(1, min(int(args.topk), len(proba_cols)))
        print_cols = proba_cols[:k] + ["pred_class_id", "pred_label"]
        print(out.loc[:, print_cols].head(20).to_string(index=False))

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
# ===== Quick Test Runs (user executes manually) =====
    # Example prediction command:
    # py .\src\predict_xgboost_ova.py --input .\data\kuzilek_clean_plus.csv --model .\outputs\models\xgboost_ova_calibrated.joblib