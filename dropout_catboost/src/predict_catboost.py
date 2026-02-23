from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
DEFAULT_SAVE_DIR = (OUTPUT_DIR / "save1").resolve()

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CatBoost multiclass prediction (batch).")
    p.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Path to input CSV.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str((DEFAULT_SAVE_DIR / "catboost_model.cbm").resolve()),
        help="Path to trained CatBoost model (.cbm).",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=str((DEFAULT_SAVE_DIR / "metrics.json").resolve()),
        help="Path to metrics.json saved during training.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str((DEFAULT_SAVE_DIR / "predictions.csv").resolve()),
        help="Path to save predictions CSV.",
    )
    p.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of rows to print in terminal.",
    )
    return p.parse_args()


def _detect_target_column(columns: pd.Index) -> str | None:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    return None


def _labels_from_classification_report(rep: dict) -> list[str]:
    excluded = {"accuracy", "macro avg", "weighted avg"}
    labels: list[str] = []
    for key, value in rep.items():
        if key in excluded:
            continue
        if isinstance(value, dict) and all(m in value for m in ("precision", "recall", "f1-score")):
            labels.append(str(key))
    if len(labels) < 2:
        raise ValueError(
            "Could not infer at least 2 class labels from classification_report in metrics.json."
        )
    return labels


def _load_labels_from_metrics(metrics_path: Path) -> list[str]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse metrics JSON: {metrics_path}") from exc

    label_names = data.get("label_names")
    if isinstance(label_names, list) and len(label_names) >= 2 and all(isinstance(x, str) for x in label_names):
        return label_names

    label_mapping = data.get("label_mapping")
    if isinstance(label_mapping, dict) and len(label_mapping) >= 2:
        try:
            mapping = {str(k): int(v) for k, v in label_mapping.items()}
        except Exception as exc:
            raise ValueError("metrics.json label_mapping must map label -> integer class_id") from exc

        return [lab for lab, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    classification_report = data.get("classification_report")
    if isinstance(classification_report, dict):
        report_labels = _labels_from_classification_report(classification_report)
        if len(report_labels) >= 2:
            return report_labels

    existing_keys = sorted(data.keys())
    raise ValueError(
        f"Could not infer class labels from metrics.json ({metrics_path}). "
        f"Found keys: {existing_keys}. "
        "Fix: re-train and save label_mapping/label_names in metrics.json, "
        "or provide labels manually via a future --labels argument."
    )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    labels = _load_labels_from_metrics(metrics_path)

    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    X = df.drop(columns=[target_col]) if target_col else df.copy()

    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].fillna("NA").astype(str)

    model = CatBoostClassifier()
    model.load_model(model_path)

    proba = model.predict_proba(X)
    if getattr(proba, "ndim", None) != 2:
        raise ValueError(
            f"Expected predict_proba output with 2 dimensions, got shape={getattr(proba, 'shape', None)}"
        )

    num_class = int(proba.shape[1])
    if num_class != len(labels):
        raise ValueError(
            f"Model/metrics mismatch: model outputs {num_class} classes but metrics labels length is {len(labels)}."
        )

    proba_cols = [f"proba_{lab}" for lab in labels]
    out = pd.DataFrame(proba, columns=proba_cols)
    out["pred_class_id"] = out[proba_cols].values.argmax(axis=1)
    out["pred_label"] = out["pred_class_id"].map(lambda i: labels[int(i)])
    out["pred_confidence"] = out[proba_cols].max(axis=1)

    print(out.head(args.head).to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nInput: {input_path}")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
