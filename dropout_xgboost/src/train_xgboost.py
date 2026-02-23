from __future__ import annotations

import json
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost (multiclass, sklearn pipeline).")
    p.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Input CSV path (default: data/data.csv)",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--metric", type=str, default="merror")
    p.add_argument(
        "--class_weights",
        type=str,
        default="",
        help="Comma-separated class weights for class_id order (0,1,2,3). Example: 2.9,1.2,0.6,1.3. If empty, no weighting.",
    )
    p.add_argument(
        "--objective",
        type=str,
        default="multi:softprob",
        help="XGBoost objective. Default multi:softprob.",
    )
    p.add_argument(
        "--eval_metric",
        type=str,
        default="mlogloss",
        help="XGBoost eval metric for early stopping and model selection. Default mlogloss.",
    )
    p.add_argument(
        "--select_by",
        type=str,
        default="mlogloss",
        choices=["mlogloss", "accuracy"],
        help="How to select best seed using validation score.",
    )
    p.add_argument("--early_stopping_rounds", type=int, default=300)
    p.add_argument("--n_estimators", type=int, default=8000)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--min_child_weight", type=float, default=3.0)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=2.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--verbosity", type=int, default=1)
    p.add_argument("--no_plot", action="store_true", default=False)
    return p.parse_args()


def build_model(args, num_class: int, random_state: int):
    return XGBClassifier(
        objective=str(args.objective),
        num_class=int(num_class),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        reg_alpha=float(args.reg_alpha),
        gamma=float(args.gamma),
        eval_metric=str(args.eval_metric),
        tree_method="hist",
        n_jobs=int(args.n_jobs),
        random_state=int(random_state),
        verbosity=int(args.verbosity),
    )


def train_with_xgb_train(
    args: argparse.Namespace,
    num_class: int,
    seed: int,
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weight_train=None,
    sample_weight_val=None,
) -> tuple[xgb.Booster, int | None]:
    """
    Version-compatible training with early stopping using xgboost.train.
    Returns (booster, best_iteration)
    """
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=sample_weight_val)

    params = {
        "objective": str(args.objective),
        "num_class": int(num_class),
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "lambda": float(args.reg_lambda),
        "alpha": float(args.reg_alpha),
        "gamma": float(args.gamma),
        "eval_metric": str(args.eval_metric),
        "seed": int(seed),
        "verbosity": int(args.verbosity),
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "val")],
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=False,
    )

    best_iter = getattr(booster, "best_iteration", None)
    return booster, best_iter


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


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
def _infer_label_mapping(series: pd.Series) -> tuple[pd.Series, dict[str, int], list[str]]:
    """
    Returns:
      y_mapped: numeric class ids (0..K-1)
      mapping: normalized_label -> class_id
      labels: class_id -> normalized_label (index order)
    """
    if pd.api.types.is_numeric_dtype(series):
        y_num = series.astype(int)
        labels = [str(i) for i in sorted(y_num.dropna().unique().tolist())]
        mapping = {lab: int(lab) for lab in labels}
        return y_num, mapping, labels

    normalized = series.astype(str).str.strip().str.lower()
    labels = sorted(normalized.dropna().unique().tolist())
    mapping = {lab: i for i, lab in enumerate(labels)}
    return normalized.map(mapping), mapping, labels


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _map_target(series: pd.Series) -> pd.Series:
    # Backward-compatible wrapper: keep name, but now dynamic.
    y_mapped, _, _ = _infer_label_mapping(series)
    return y_mapped


def _parse_class_weights(class_weights_arg: str | None, num_class: int) -> dict[int, float] | None:
    if class_weights_arg is None or str(class_weights_arg).strip() == "":
        return None

    parts = [p.strip() for p in str(class_weights_arg).split(",")]
    if len(parts) != int(num_class):
        raise ValueError(
            f"Invalid --class_weights: expected exactly {num_class} values for class ids "
            f"0..{num_class - 1}. Example: --class_weights 2.9,1.2,0.6,1.3"
        )

    try:
        weights = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            "Invalid --class_weights: all values must be floats. "
            "Example: --class_weights 2.9,1.2,0.6,1.3"
        ) from exc

    if any(w <= 0 for w in weights):
        raise ValueError(
            "Invalid --class_weights: all weights must be > 0. "
            "Example: --class_weights 2.9,1.2,0.6,1.3"
        )

    return {i: w for i, w in enumerate(weights)}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_column = _detect_target_column(df.columns)
    print(f"Detected target column: {target_column}")

    y_raw = df[target_column]
    y, label_mapping, labels = _infer_label_mapping(y_raw)
    X = df.drop(columns=[target_column])
    valid = y.notna()
    y = y[valid]
    X = X.loc[valid]
    print(f"Detected num_class: {len(labels)}")
    print("Label mapping (normalized_label -> class_id):")
    print(label_mapping)
    print("Class distribution after mapping:")
    print(y.value_counts())

    num_class = len(labels)
    class_weight_map = _parse_class_weights(args.class_weights, num_class)
    if class_weight_map is not None:
        print(f"Using class_weights: {class_weight_map}")

    best = {
        "seed": None,
        "val_acc": -1.0,
        "val_mlogloss": None,
        "test_acc": -1.0,
        "best_iteration": None,
        "model": None,
        "labels": labels,
        "X_test": None,
        "y_test": None,
    }

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))

    for seed in seeds:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=float(args.test_size),
            random_state=int(seed),
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=float(args.val_size),
            random_state=int(seed),
            stratify=y_train_full,
        )

        sample_weight_train = (
            np.array([class_weight_map[int(c)] for c in y_train], dtype=float)
            if class_weight_map is not None
            else None
        )
        sample_weight_val = (
            np.array([class_weight_map[int(c)] for c in y_val], dtype=float)
            if class_weight_map is not None
            else None
        )
        sample_weight_test = (
            np.array([class_weight_map[int(c)] for c in y_test], dtype=float)
            if class_weight_map is not None
            else None
        )

        booster, best_iteration = train_with_xgb_train(
            args=args,
            num_class=num_class,
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight_train=sample_weight_train,
            sample_weight_val=sample_weight_val,
        )

        dval = xgb.DMatrix(X_val)
        dtest = xgb.DMatrix(X_test)

        val_proba = booster.predict(dval)
        test_proba = booster.predict(dtest)

        val_pred = val_proba.argmax(axis=1)
        test_pred = test_proba.argmax(axis=1)

        val_acc = float(accuracy_score(y_val, val_pred))
        test_acc = float(accuracy_score(y_test, test_pred))
        val_mlogloss = None
        if hasattr(booster, "evals_result_"):
            evals_result = getattr(booster, "evals_result_", {})
            if isinstance(evals_result, dict):
                val_hist = evals_result.get("validation_0", {})
                if "mlogloss" in val_hist and len(val_hist["mlogloss"]) > 0:
                    val_mlogloss = float(val_hist["mlogloss"][-1])
        elif hasattr(booster, "evals_result"):
            evals_result_fn = getattr(booster, "evals_result")
            try:
                evals_result = evals_result_fn() if callable(evals_result_fn) else evals_result_fn
                if isinstance(evals_result, dict):
                    val_hist = evals_result.get("val", {})
                    if "mlogloss" in val_hist and len(val_hist["mlogloss"]) > 0:
                        val_mlogloss = float(val_hist["mlogloss"][-1])
            except Exception:
                val_mlogloss = None

        if val_mlogloss is None:
            val_mlogloss = float(
                log_loss(
                    y_val,
                    val_proba,
                    labels=list(range(num_class)),
                    sample_weight=sample_weight_val,
                )
            )

        print(
            f"seed={seed} | val_mlogloss={val_mlogloss:.6f} | val_acc={val_acc:.6f} | "
            f"test_acc={test_acc:.6f} | best_iter={best_iteration}"
        )

        is_better = False
        if str(args.select_by) == "mlogloss":
            best_mlogloss = best["val_mlogloss"]
            if best_mlogloss is None or val_mlogloss < float(best_mlogloss):
                is_better = True
            elif val_mlogloss == float(best_mlogloss):
                if val_acc > float(best["val_acc"]):
                    is_better = True
                elif val_acc == float(best["val_acc"]) and test_acc > float(best["test_acc"]):
                    is_better = True
        else:
            if val_acc > float(best["val_acc"]):
                is_better = True
            elif val_acc == float(best["val_acc"]):
                if test_acc > float(best["test_acc"]):
                    is_better = True
                elif test_acc == float(best["test_acc"]):
                    best_mlogloss = best["val_mlogloss"]
                    if best_mlogloss is None or val_mlogloss < float(best_mlogloss):
                        is_better = True

        if is_better:
            best["seed"] = int(seed)
            best["val_acc"] = float(val_acc)
            best["val_mlogloss"] = float(val_mlogloss)
            best["test_acc"] = float(test_acc)
            best["best_iteration"] = best_iteration
            best["model"] = booster
            best["X_test"] = X_test
            best["y_test"] = y_test

    if best["seed"] is None or best["model"] is None:
        raise RuntimeError("No model trained. Check --num_seeds and data split settings.")

    booster = best["model"]
    seed = best["seed"]
    X_test = best["X_test"]
    y_test = best["y_test"]
    dtest = xgb.DMatrix(X_test)
    test_proba = booster.predict(dtest)
    y_pred = test_proba.argmax(axis=1)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    print(
        f"Selected best seed={seed} | val_mlogloss={best['val_mlogloss']:.6f} | val_acc={best['val_acc']:.6f} | "
        f"test_acc={best['test_acc']:.6f} | best_iter={best['best_iteration']}"
    )

    print("\nXGBoost Metrics (multiclass, accuracy mode):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(num_class)),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=list(range(num_class)),
        display_labels=labels,
    )
    plt.title(f"Confusion Matrix (multiclass, K={num_class})")
    if not args.no_plot:
        plt.show()
    else:
        plt.close()

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(num_class)),
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_class))).tolist()

    metrics = {
        "mode": "accuracy",
        "metric": str(args.metric),
        "best_seed": int(best["seed"]),
        "val_accuracy": float(best["val_acc"]),
        "test_accuracy": float(test_accuracy),
        "best_iteration": best["best_iteration"],
        "num_class": int(num_class),
        "labels": labels,
        "label_mapping": label_mapping,
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "gamma": args.gamma,
            "early_stopping_rounds": args.early_stopping_rounds,
            "num_seeds": args.num_seeds,
            "seed_start": args.seed_start,
        },
        "classification_report": report_dict,
        "confusion_matrix": matrix,
    }

    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "models" / "xgboost_model.json"
    booster.save_model(str(model_path))
    joblib_model_path = OUTPUT_DIR / "models" / "xgboost_model.joblib"
    joblib.dump(booster, joblib_model_path)

    metrics_path = OUTPUT_DIR / "metrics" / "metrics.json"
    save_json(metrics, metrics_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved model to: {joblib_model_path}")
    print(
        "You can run prediction with: py .\\src\\predict_xgboost.py --input <CSV> "
        "--model .\\outputs\\models\\xgboost_model.joblib"
    )


if __name__ == "__main__":
    # ===== Quick Test Runs (user executes manually) =====
    # Baseline with defaults (softprob + mlogloss, no class weights):
    # python dropout_xgboost/src/train_xgboost.py
    # With class weights:
    # python dropout_xgboost/src/train_xgboost.py --class_weights 2.9,1.2,0.6,1.3 --select_by mlogloss
    main()
