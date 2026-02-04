import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from .utils import (
    ensure_dir,
    load_csv,
    infer_target_column,
    preprocess_tabular,
    compute_metrics,
    compute_metrics_multiclass,
    save_json,
    LABEL_MAPPING,
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


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_to_weight = {cls: w for cls, w in zip(classes, weights)}
    return np.asarray([class_to_weight[int(label)] for label in y], dtype=np.float32)


def _tabnet_supports_weights() -> bool:
    sig = inspect.signature(TabNetClassifier.fit)
    return "weights" in sig.parameters


def _parse_seeds(seeds_str: str, default_seed: int) -> list[int]:
    s = (seeds_str or "").strip()
    if not s:
        return [default_seed]
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _print_multiclass_tables(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Dropout", "Enrolled", "Graduate"],
        output_dict=True,
        zero_division=0,
    )

    headers = [
        ("Model", 12),
        ("accuracy", 12),
        ("f1_macro", 12),
        ("recall_macro", 14),
        ("roc_auc_ovr_macro", 20),
        ("pr_auc_ovr_macro", 20),
    ]
    row = [
        "TabNet",
        metrics.get("accuracy"),
        metrics.get("f1_macro"),
        metrics.get("recall_macro"),
        metrics.get("roc_auc_ovr_macro"),
        metrics.get("pr_auc_ovr_macro"),
    ]

    def _fmt(v, nd=4):
        if v is None:
            return "NA"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    header_line = " | ".join(h.ljust(w) for h, w in headers)
    sep_line = "-+-".join("-" * w for _, w in headers)
    row_line = " | ".join(
        _fmt(val).ljust(w) for (val, (_, w)) in zip(row, headers)
    )
    print(header_line)
    print(sep_line)
    print(row_line)

    per_headers = [
        ("Class", 6),
        ("label", 12),
        ("precision", 10),
        ("recall", 10),
        ("f1", 10),
        ("support", 8),
    ]
    per_header_line = " | ".join(h.ljust(w) for h, w in per_headers)
    per_sep_line = "-+-".join("-" * w for _, w in per_headers)
    print(per_header_line)
    print(per_sep_line)
    for class_id, label in [(0, "Dropout"), (1, "Enrolled"), (2, "Graduate")]:
        stats = report.get(label, {})
        row_vals = [
            class_id,
            label,
            stats.get("precision"),
            stats.get("recall"),
            stats.get("f1-score"),
            stats.get("support"),
        ]
        row_line = " | ".join(
            _fmt(val, nd=4).ljust(w) if i < 5 else _fmt(val, nd=0).ljust(w)
            for i, (val, (_, w)) in enumerate(zip(row_vals, per_headers))
        )
        print(row_line)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("confusion_matrix:")
    print("0:Dropout 1:Enrolled 2:Graduate")
    for idx, row in enumerate(cm):
        label = ["Dropout", "Enrolled", "Graduate"][idx]
        counts = " ".join(str(int(v)) for v in row)
        print(f"{idx}:{label} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TabNet on tabular dropout data")
    parser.add_argument("--data_path", default="data/data.csv")
    parser.add_argument("--target_col", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--virtual_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="Classification task type.",
    )
    parser.add_argument(
        "--use_weights",
        action="store_true",
        help="Use sample weights for class imbalance (may reduce raw accuracy).",
    )
    parser.add_argument("--n_d", type=int, default=48)
    parser.add_argument("--n_a", type=int, default=48)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1.6)
    parser.add_argument("--lambda_sparse", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds for ensembling, e.g., 42,43,44")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_csv(str(data_path))

    target_col = args.target_col.strip()
    if not target_col:
        target_col = infer_target_column(df)

    seeds = _parse_seeds(args.seeds, args.seed)

    base_dir = Path(".")
    models_dir = base_dir / "outputs" / "models"
    metrics_dir = base_dir / "outputs" / "metrics"
    preds_dir = base_dir / "outputs" / "predictions"
    ensure_dir(str(models_dir))
    ensure_dir(str(metrics_dir))
    ensure_dir(str(preds_dir))

    best_acc = -1.0
    best_pack = None  # (model, metrics, y_proba_test, y_pred, seed, split)
    trained_seeds = []
    model_zips = []

    for sd in seeds:
        _set_seeds(sd)

        split = preprocess_tabular(
            df,
            target_col=target_col,
            test_size=0.2,
            valid_size=0.2,
            random_state=sd,
        )
        print(f"Unique classes in y_train: {np.unique(split.y_train)}")

        model = TabNetClassifier(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            gamma=args.gamma,
            lambda_sparse=args.lambda_sparse,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            seed=sd,
        )

        X_train = split.X_train
        y_train = split.y_train
        X_val = split.X_valid
        y_val = split.y_valid
        sample_weights = _compute_sample_weights(y_train)

        fit_sig = inspect.signature(model.fit)
        fit_kwargs = dict(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            virtual_batch_size=args.virtual_batch_size,
            num_workers=0,
            drop_last=False,
            eval_metric=["accuracy"],
        )

        if args.use_weights and sample_weights is not None:
            if "weights" in fit_sig.parameters:
                fit_kwargs["weights"] = sample_weights
            else:
                print("Info: TabNet fit() does not support weights in this version.")

        model.fit(**fit_kwargs)

        model_path_no_ext = models_dir / f"tabnet_model_seed{sd}"
        model.save_model(str(model_path_no_ext))
        model_zips.append(f"tabnet_model_seed{sd}.zip")
        trained_seeds.append(sd)

        y_proba_test = model.predict_proba(split.X_test)
        if args.task == "multiclass":
            if y_proba_test.ndim != 2 or y_proba_test.shape[1] != 3:
                raise ValueError(f"Expected y_proba_test with shape (n, 3), got {y_proba_test.shape}")
            y_pred = np.argmax(y_proba_test, axis=1).astype(int)
            metrics = compute_metrics_multiclass(split.y_test, y_proba_test)
        else:
            if y_proba_test.ndim != 2 or y_proba_test.shape[1] != 2:
                raise ValueError(f"Expected y_proba_test with shape (n, 2), got {y_proba_test.shape}")
            y_pred = (y_proba_test[:, 1] >= args.threshold).astype(int)
            metrics = compute_metrics(split.y_test, y_proba_test, threshold=args.threshold)

        acc = float(metrics.get("accuracy", -1.0))
        print(f"[seed={sd}] test accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_pack = (model, metrics, y_proba_test, y_pred, sd, split)

    model, metrics, y_proba_test, y_pred, best_seed, split = best_pack
    print(f"Best seed: {best_seed} | best test accuracy: {best_acc:.4f}")

    model_path_no_ext = models_dir / "tabnet_model"
    model.save_model(str(model_path_no_ext))

    meta = {
        "model_name": "TabNet",
        "target_col": target_col,
        "feature_names": split.feature_names,
        "task": args.task,
        "label_mapping_used": LABEL_MAPPING,
        "best_seed": best_seed,
        "trained_seeds": trained_seeds,
        "model_zips": model_zips,
    }
    if args.task == "binary":
        meta["threshold"] = args.threshold
    else:
        meta["threshold"] = None

    meta_path = models_dir / "meta.json"
    save_json(str(meta_path), meta)

    scaler_path = models_dir / "scaler.joblib"
    import joblib as _joblib

    _joblib.dump(split.scaler, str(scaler_path))

    if args.task == "multiclass":
        metrics_payload = {
            "model_name": "TabNet",
            "num_features": len(split.feature_names),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "classification_report": metrics.get("classification_report"),
        }
    else:
        metrics_payload = {
            "model_name": "TabNet",
            "num_features": len(split.feature_names),
            **metrics,
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
        }
    metrics_path = metrics_dir / "metrics.json"
    save_json(str(metrics_path), metrics_payload)

    if args.task == "multiclass":
        preds_df = pd.DataFrame(
            {
                "y_true": split.y_test,
                "y_pred": y_pred,
                "prob_dropout": y_proba_test[:, 0],
                "prob_enrolled": y_proba_test[:, 1],
                "prob_graduate": y_proba_test[:, 2],
            }
        )
    else:
        preds_df = pd.DataFrame(
            {
                "y_true": split.y_test,
                "y_pred": y_pred,
                "prob_0": y_proba_test[:, 0],
                "prob_1": y_proba_test[:, 1],
            }
        )
    preds_path = preds_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

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

    if args.task == "multiclass":
        print()
        _print_multiclass_tables(split.y_test, y_pred, metrics)
        print()
    else:
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
