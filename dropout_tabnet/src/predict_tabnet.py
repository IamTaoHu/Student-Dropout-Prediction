import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from .utils import (
    ensure_dir,
    load_csv,
    infer_target_column,
    _map_target_to_int,
    compute_metrics,
    load_artifacts,
    save_json,
    ID2LABEL,
)


def _resolve_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if path.name == "data.csv":
        alt = path.with_name("data.cvs")
        if alt.exists():
            return alt
    return path


def _prepare_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    filled = df.copy()
    if len(num_cols) > 0:
        for col in num_cols:
            filled[col] = filled[col].fillna(filled[col].median())
    if len(cat_cols) > 0:
        for col in cat_cols:
            filled[col] = filled[col].fillna("missing")

    encoded = pd.get_dummies(filled, columns=cat_cols, drop_first=False)
    aligned = encoded.reindex(columns=feature_names, fill_value=0)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with TabNet model")
    parser.add_argument("--data_path", default="data/data.csv")
    parser.add_argument("--model_dir", default="outputs/models")
    parser.add_argument("--model_zips", default="", help="Comma-separated model zip filenames in model_dir")
    parser.add_argument("--target_col", default="")
    parser.add_argument("--threshold", default="")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model_dir = Path(args.model_dir)
    if args.model_zips.strip():
        zip_names = [z.strip() for z in args.model_zips.split(",") if z.strip()]
        model_zip_paths = [model_dir / z for z in zip_names]
    else:
        model_zip_paths = [model_dir / "tabnet_model.zip"]

    for zp in model_zip_paths:
        if not zp.exists():
            raise FileNotFoundError(f"Model file not found: {zp}")

    scaler, meta = load_artifacts(str(model_dir))

    threshold = None
    if isinstance(args.threshold, str) and args.threshold.strip() != "":
        threshold = float(args.threshold)
    else:
        threshold = float(meta.get("threshold", 0.5))
    print("Warning: multi-class prediction ignores --threshold.")

    df = load_csv(str(data_path))

    target_col = args.target_col.strip()
    if not target_col:
        target_col = infer_target_column(df)

    y_true = None
    if target_col and target_col in df.columns:
        y_series = df[target_col]
        y_true = _map_target_to_int(y_series)
        X_raw = df.drop(columns=[target_col])
    else:
        X_raw = df

    feature_names = meta.get("feature_names", [])
    X_aligned = _prepare_features(X_raw, feature_names)
    X_scaled = scaler.transform(X_aligned).astype(np.float32)

    probas = []
    for zp in model_zip_paths:
        m = TabNetClassifier()
        m.load_model(str(zp))
        probas.append(m.predict_proba(X_scaled))

    y_proba = np.mean(np.stack(probas, axis=0), axis=0)
    if y_proba.ndim != 2 or y_proba.shape[1] != 3:
        raise ValueError(f"Expected y_proba with shape (n, 3), got {y_proba.shape}")
    y_pred = np.argmax(y_proba, axis=1).astype(int)
    pred_labels = [ID2LABEL[int(i)] for i in y_pred]

    base_dir = Path(".")
    preds_dir = base_dir / "outputs" / "predictions"
    metrics_dir = base_dir / "outputs" / "metrics"
    ensure_dir(str(preds_dir))
    ensure_dir(str(metrics_dir))

    preds_payload = {
        "y_pred": y_pred,
        "pred_label": pred_labels,
        "prob_0": y_proba[:, 0],
        "prob_1": y_proba[:, 1],
        "prob_2": y_proba[:, 2],
    }
    if y_true is not None:
        preds_payload["y_true"] = y_true
    preds_df = pd.DataFrame(preds_payload)
    preds_path = preds_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    print(f"Predicted rows: {len(preds_df)}")

    if y_true is not None:
        metrics = compute_metrics(y_true, y_proba)
        metrics_payload = {
            "model_name": "TabNet",
            "num_features": len(feature_names),
            **metrics,
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
        }
        metrics_path = metrics_dir / "metrics_predict.json"
        save_json(str(metrics_path), metrics_payload)

        tn = metrics.get("TN", None)
        fp = metrics.get("FP", None)
        fn = metrics.get("FN", None)
        tp = metrics.get("TP", None)
    def _fmt(v, width=10, prec=4):
        if v is None:
            return f"{'NA':>{width}}"
        if isinstance(v, float):
            return f"{v:>{width}.{prec}f}"
        return f"{str(v):>{width}}"

    print()
    print(
        f"{'Model':<10}"
        f"{'accuracy':>10}"
        f"{'f1':>10}"
        f"{'recall':>10}"
        f"{'roc_auc':>10}"
        f"{'pr_auc':>10}"
        f"{'TN':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'TP':>6}"
    )
    print("-" * 84)
    print(
        f"{'TabNet':<10}"
        f"{_fmt(metrics.get('accuracy'))}"
        f"{_fmt(metrics.get('f1'))}"
        f"{_fmt(metrics.get('recall'))}"
        f"{_fmt(metrics.get('roc_auc'))}"
        f"{_fmt(metrics.get('pr_auc'))}"
        f"{_fmt(metrics.get('TN'), 6, 0)}"
        f"{_fmt(metrics.get('FP'), 6, 0)}"
        f"{_fmt(metrics.get('FN'), 6, 0)}"
        f"{_fmt(metrics.get('TP'), 6, 0)}"
    )
    print()



if __name__ == "__main__":
    main()
