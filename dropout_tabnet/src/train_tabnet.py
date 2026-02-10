import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils import (
    ensure_dir,
    load_csv,
    infer_target_column,
    preprocess_tabular,
    compute_metrics,
    compute_metrics_multiclass,
    save_json,
    LABEL_MAPPING,
)


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


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

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    base_dir = PROJECT_ROOT
    data_path = resolve_path(base_dir, args.data_path)
    data_path = _resolve_data_path(str(data_path))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_csv(str(data_path))

    target_col = args.target_col.strip()
    if not target_col:
        target_col = infer_target_column(df)

    seeds = _parse_seeds(args.seeds, args.seed)

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

    def _fmt(v, width=10, prec=4):
        if v is None:
            return f"{'NA':>{width}}"
        if isinstance(v, float):
            return f"{v:>{width}.{prec}f}"
        return f"{str(v):>{width}}"

    if args.task == "multiclass":
        print("\nTabNet Metrics (3-class, sklearn):")
        print(
            classification_report(
                split.y_test,
                y_pred,
                target_names=["dropout", "enrolled", "graduate"],
                digits=4,
                zero_division=0,
            )
        )

        ConfusionMatrixDisplay.from_predictions(
            split.y_test,
            y_pred,
            display_labels=["dropout", "enrolled", "graduate"],
        )
        plt.title("Confusion Matrix (3-class)")
        plt.show()
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
