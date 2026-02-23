from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb


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
        "--metrics",
        type=str,
        default=str((OUTPUT_DIR / "metrics" / "metrics.json").resolve()),
        help="Metrics JSON path that contains labels (default: outputs/metrics/metrics.json)",
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


def _load_labels_from_metrics(metrics_path: Path) -> list[str] | None:
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    labels = data.get("labels")
    if (
        isinstance(labels, list)
        and len(labels) >= 2
        and all(isinstance(x, str) for x in labels)
    ):
        return labels
    return None


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    X = df.drop(columns=[target_col]) if target_col else df.copy()
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        print("Dropping non-numeric columns:", non_numeric)
        X = X.drop(columns=non_numeric)
    X = X.astype("float32")

    model_path = Path(args.model).expanduser().resolve()
    model = joblib.load(model_path)

    print("X shape:", X.shape)
    print("X dtypes summary:", X.dtypes.value_counts().to_dict())

    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X)
        proba = model.predict(dmat)
    else:
        proba = model.predict_proba(X)

    if getattr(proba, "ndim", None) != 2:
        raise ValueError(
            f"Expected 2D probability array (n_samples, n_classes), got shape={getattr(proba, 'shape', None)}"
        )
    num_class = proba.shape[1]
    labels_from_metrics = _load_labels_from_metrics(Path(args.metrics).expanduser().resolve())
    if labels_from_metrics is not None and len(labels_from_metrics) == num_class:
        labels = labels_from_metrics
        print(f"Loaded labels from metrics.json: {labels}")
    else:
        labels = [f"class_{i}" for i in range(num_class)]
        print("metrics.json labels not found/mismatched; using fallback labels:", labels)

    proba_cols = [f"proba_{lab}" for lab in labels]
    out = pd.DataFrame(proba, columns=proba_cols)
    out["pred_class_id"] = out[proba_cols].values.argmax(axis=1)
    out["pred_label"] = out["pred_class_id"].map(lambda i: labels[int(i)])

    print(out.head(args.head).to_string(index=False))

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
