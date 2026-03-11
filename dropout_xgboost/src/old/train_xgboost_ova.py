from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split

try:
    from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6 style
except Exception:
    FrozenEstimator = None


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

CANONICAL_LABEL_ORDER = ["distinction", "fail", "pass", "withdrawn"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train One-vs-All XGBoost with probability calibration.")
    p.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Input CSV path (default: data/data.csv)",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument(
        "--select_by",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "logloss", "accuracy"],
        help="How to select best seed using validation score.",
    )
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=12000)
    p.add_argument("--learning_rate", type=float, default=0.02)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--min_child_weight", type=float, default=5.0)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=3.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--max_delta_step", type=float, default=1.0)
    p.add_argument("--early_stopping_rounds", type=int, default=300)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--verbosity", type=int, default=1)
    p.add_argument("--no_plot", action="store_true", default=False)
    p.add_argument(
        "--calibration",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="Calibration method for one-vs-all probabilities.",
    )
    p.add_argument(
        "--pos_weight_multipliers",
        type=str,
        default="1,1,1,1",
        help=(
            "Comma-separated 4 floats in label order "
            "(distinction,fail,pass,withdrawn). "
            "Each value multiplies auto neg/pos per class."
        ),
    )
    return p.parse_args()


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _infer_label_mapping(series: pd.Series) -> tuple[pd.Series, dict[str, int], list[str]]:
    normalized = series.astype(str).str.strip().str.lower()
    labels = sorted(normalized.dropna().unique().tolist())
    # Lock class IDs to canonical order when all expected classes are present.
    if set(labels) == set(CANONICAL_LABEL_ORDER):
        labels = CANONICAL_LABEL_ORDER.copy()
    mapping = {lab: i for i, lab in enumerate(labels)}
    y = normalized.map(mapping).astype("Int64")
    return y, mapping, labels


def load_data(input_path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, int]]:
    data_path = Path(input_path).expanduser().resolve()
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_column = _detect_target_column(df.columns)
    y_raw = df[target_column]
    y, label_mapping, labels = _infer_label_mapping(y_raw)

    valid = y.notna()
    X = df.drop(columns=[target_column]).loc[valid]
    y = y.loc[valid].astype(int)

    return X, y, labels, label_mapping


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=float(val_size),
        random_state=int(seed),
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_binary_labels(y: pd.Series, positive_class_id: int) -> np.ndarray:
    return (y.to_numpy() == int(positive_class_id)).astype(int)


def compute_pos_weight(y_bin: np.ndarray) -> float:
    y_arr = np.asarray(y_bin).astype(int)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg / pos)


def build_binary_model(args: argparse.Namespace, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        reg_alpha=float(args.reg_alpha),
        gamma=float(args.gamma),
        max_delta_step=float(args.max_delta_step),
        tree_method="hist",
        n_jobs=int(args.n_jobs),
        random_state=int(seed),
        verbosity=int(args.verbosity),
    )


def fit_one_ova_model(
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train_bin: np.ndarray,
    X_val: pd.DataFrame,
    y_val_bin: np.ndarray,
    seed: int,
    pos_weight_multiplier: float,
) -> xgb.XGBClassifier:
    model = build_binary_model(args=args, seed=seed)
    base_pos_weight = compute_pos_weight(y_train_bin)
    final_pos_weight = float(base_pos_weight * float(pos_weight_multiplier))

    y_train_arr = np.asarray(y_train_bin).astype(int)
    sample_weight_train = np.where(y_train_arr == 1, final_pos_weight, 1.0)

    # NOTE: XGBoost sklearn-wrapper API differs by version.
    # We try early stopping in a compatibility-first order:
    # 1) callbacks + eval_set (newer API)
    # 2) early_stopping_rounds + eval_set (older API)
    # 3) eval_set only
    # 4) minimal fit
    try:
        model.fit(
            X_train,
            y_train_bin,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val_bin)],
            callbacks=[EarlyStopping(rounds=int(args.early_stopping_rounds), save_best=True)],
            verbose=False,
        )
    except TypeError:
        try:
            model.fit(
                X_train,
                y_train_bin,
                sample_weight=sample_weight_train,
                eval_set=[(X_val, y_val_bin)],
                early_stopping_rounds=int(args.early_stopping_rounds),
                verbose=False,
            )
        except TypeError:
            try:
                model.fit(
                    X_train,
                    y_train_bin,
                    sample_weight=sample_weight_train,
                    eval_set=[(X_val, y_val_bin)],
                    verbose=False,
                )
            except TypeError:
                model.fit(
                    X_train,
                    y_train_bin,
                    sample_weight=sample_weight_train,
                    verbose=False,
                )
    return model


def calibrate_model(model, X_calib, y_calib_bin, method: str):
    """
    Calibrate a *pre-fitted* binary classifier.
    New scikit-learn versions removed cv="prefit"; use FrozenEstimator when available.
    """
    # New sklearn: wrap prefit estimator
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
        )
        calibrated.fit(X_calib, y_calib_bin)
        return calibrated

    # Old sklearn fallback: cv="prefit"
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_calib, y_calib_bin)
    return calibrated


def ova_predict_proba_raw(calibrated_models: list, X) -> np.ndarray:
    proba_list = []
    for i in range(len(calibrated_models)):
        p_pos = calibrated_models[i].predict_proba(X)[:, 1]
        proba_list.append(p_pos)

    return np.column_stack(proba_list)


def ova_predict_proba(calibrated_models: list, X, normalize: bool = False) -> np.ndarray:
    proba_raw = ova_predict_proba_raw(calibrated_models, X)
    if not normalize:
        return proba_raw
    return _normalize_ova_probabilities(proba_raw)


def ova_predict(calibrated_models: list, X) -> np.ndarray:
    proba = ova_predict_proba(calibrated_models, X, normalize=False)
    return proba.argmax(axis=1)


def _parse_pos_weight_multipliers(raw: str) -> dict[str, float]:
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--pos_weight_multipliers must contain exactly 4 values in "
            "(distinction,fail,pass,withdrawn) order."
        )

    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--pos_weight_multipliers values must be floats.") from exc

    if any(v <= 0 for v in values):
        raise ValueError("--pos_weight_multipliers values must be > 0.")

    return {name: values[idx] for idx, name in enumerate(CANONICAL_LABEL_ORDER)}


def _fit_calibrator(y_true_bin: pd.Series, raw_proba: np.ndarray, method: str) -> dict:
    y_values = np.asarray(y_true_bin).astype(int)
    p = np.clip(np.asarray(raw_proba, dtype=float), 1e-6, 1.0 - 1e-6)

    if len(np.unique(y_values)) < 2:
        return {"kind": "constant", "value": float(y_values.mean() if len(y_values) else 0.0)}

    if method == "sigmoid":
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(logits, y_values)
        return {"kind": "sigmoid", "model": model}

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(p, y_values)
    return {"kind": "isotonic", "model": model}


def _apply_calibrator(calibrator: dict, raw_proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(raw_proba, dtype=float), 1e-6, 1.0 - 1e-6)
    kind = calibrator.get("kind")

    if kind == "constant":
        return np.full(shape=p.shape, fill_value=float(calibrator["value"]), dtype=float)

    if kind == "sigmoid":
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return calibrator["model"].predict_proba(logits)[:, 1]

    if kind == "isotonic":
        return calibrator["model"].predict(p)

    return p


def _normalize_ova_probabilities(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    row_sum = p.sum(axis=1, keepdims=True)
    out = np.divide(p, row_sum, out=np.zeros_like(p), where=row_sum > 0)

    zero_rows = np.where(row_sum.ravel() <= 0)[0]
    if len(zero_rows) > 0:
        out[zero_rows] = 1.0 / p.shape[1]

    return out


def _fit_binary_booster(
    args: argparse.Namespace,
    seed: int,
    scale_pos_weight: float,
    X_train: pd.DataFrame,
    y_train_bin: pd.Series,
    X_val: pd.DataFrame,
    y_val_bin: pd.Series,
) -> tuple[xgb.Booster, int | None, np.ndarray, np.ndarray]:
    dtrain = xgb.DMatrix(X_train, label=y_train_bin)
    dval = xgb.DMatrix(X_val, label=y_val_bin)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "lambda": float(args.reg_lambda),
        "alpha": float(args.reg_alpha),
        "gamma": float(args.gamma),
        "max_delta_step": float(args.max_delta_step),
        "scale_pos_weight": float(scale_pos_weight),
        "seed": int(seed),
        "verbosity": int(args.verbosity),
        "nthread": int(args.n_jobs),
        "tree_method": "hist",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "val")],
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=False,
    )

    best_iteration = getattr(booster, "best_iteration", None)
    val_proba = booster.predict(dval)
    return booster, best_iteration, val_proba, params


def main() -> None:
    args = parse_args()

    X, y, labels, label_mapping = load_data(args.input)
    num_class = len(labels)
    if num_class < 2:
        raise RuntimeError(f"Need at least 2 classes for OVA training, got {num_class}.")
    if num_class != 4:
        print(f"Warning: expected 4 classes but found {num_class}. Proceeding with K={num_class}.")

    print("Detected num_class:", num_class)
    print("Label mapping (normalized_label -> class_id):")
    print(label_mapping)
    print("Class distribution after mapping:")
    print(y.value_counts())

    multiplier_parts = [p.strip() for p in str(args.pos_weight_multipliers).split(",")]
    try:
        multipliers = [float(x) for x in multiplier_parts]
    except ValueError as exc:
        raise ValueError("--pos_weight_multipliers must be comma-separated floats.") from exc
    if len(multipliers) != num_class:
        raise ValueError(
            f"--pos_weight_multipliers length mismatch: expected {num_class} values, got {len(multipliers)}."
        )
    if any(v <= 0 for v in multipliers):
        raise ValueError("--pos_weight_multipliers values must be > 0.")

    best = {
        "seed": None,
        "val_acc": -1.0,
        "val_macro_f1": -1.0,
        "val_bal_acc": -1.0,
        "val_logloss": None,
        "test_acc": -1.0,
        "labels": labels,
        "calibrated_models": None,
        "class_details": None,
        "X_test": None,
        "y_test": None,
    }

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))

    for seed in seeds:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X=X,
            y=y,
            seed=seed,
            test_size=float(args.test_size),
            val_size=float(args.val_size),
        )

        class_calibrators = []
        class_details = []

        for class_id, label_name in enumerate(labels):
            y_train_bin = make_binary_labels(y_train, class_id)
            y_val_bin = make_binary_labels(y_val, class_id)

            base_model = fit_one_ova_model(
                args=args,
                X_train=X_train,
                y_train_bin=y_train_bin,
                X_val=X_val,
                y_val_bin=y_val_bin,
                seed=int(seed),
                pos_weight_multiplier=float(multipliers[class_id]),
            )
            calibrated = calibrate_model(
                model=base_model,
                X_calib=X_val,
                y_calib_bin=y_val_bin,
                method=args.calibration,
            )

            class_calibrators.append(calibrated)
            class_details.append(
                {
                    "class_id": int(class_id),
                    "label": str(label_name),
                    "train_pos": int((y_train_bin == 1).sum()),
                    "train_neg": int((y_train_bin == 0).sum()),
                    "auto_neg_pos_ratio": float(compute_pos_weight(y_train_bin)),
                    "multiplier": float(multipliers[class_id]),
                    "calibration": str(args.calibration),
                    "best_iteration": getattr(base_model, "best_iteration", None),
                }
            )

        val_proba_ova_raw = ova_predict_proba(class_calibrators, X_val, normalize=False)
        val_pred = val_proba_ova_raw.argmax(axis=1)
        test_proba_ova_raw = ova_predict_proba(class_calibrators, X_test, normalize=False)
        test_pred = test_proba_ova_raw.argmax(axis=1)

        val_acc = float(accuracy_score(y_val, val_pred))
        val_report_dict = classification_report(
            y_val,
            val_pred,
            labels=list(range(num_class)),
            target_names=labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        val_macro_f1 = float(val_report_dict["macro avg"]["f1-score"])
        val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
        val_proba_ova_norm = _normalize_ova_probabilities(val_proba_ova_raw)
        val_logloss = float(log_loss(y_val, val_proba_ova_norm, labels=list(range(num_class))))
        test_acc = float(accuracy_score(y_test, test_pred))

        print(
            f"seed={seed} | val_logloss={val_logloss:.6f} | "
            f"val_acc={val_acc:.6f} | val_macro_f1={val_macro_f1:.6f} | "
            f"val_bal_acc={val_bal_acc:.6f} | test_acc={test_acc:.6f}"
        )

        is_better = False
        if args.select_by == "macro_f1":
            if val_macro_f1 > float(best["val_macro_f1"]):
                is_better = True
            elif val_macro_f1 == float(best["val_macro_f1"]):
                if val_bal_acc > float(best["val_bal_acc"]):
                    is_better = True
                elif val_bal_acc == float(best["val_bal_acc"]):
                    best_logloss = best.get("val_logloss")
                    if best_logloss is None or val_logloss < float(best_logloss):
                        is_better = True
        elif args.select_by == "logloss":
            best_logloss = best.get("val_logloss")
            if best_logloss is None or val_logloss < float(best_logloss):
                is_better = True
            elif val_logloss == float(best_logloss):
                if val_acc > float(best["val_acc"]):
                    is_better = True
        else:
            if val_acc > float(best["val_acc"]):
                is_better = True
            elif val_acc == float(best["val_acc"]):
                if val_macro_f1 > float(best["val_macro_f1"]):
                    is_better = True
                elif val_macro_f1 == float(best["val_macro_f1"]):
                    best_logloss = best.get("val_logloss")
                    if best_logloss is None or val_logloss < float(best_logloss):
                        is_better = True

        if is_better:
            best.update(
                {
                    "seed": int(seed),
                    "val_acc": float(val_acc),
                    "val_macro_f1": float(val_macro_f1),
                    "val_bal_acc": float(val_bal_acc),
                    "val_logloss": float(val_logloss),
                    "test_acc": float(test_acc),
                    "labels": labels,
                    "calibrated_models": class_calibrators,
                    "class_details": class_details,
                    "X_test": X_test,
                    "y_test": y_test,
                }
            )

    if best["seed"] is None:
        raise RuntimeError("No valid model trained. Check data split settings and class balance.")

    X_test = best["X_test"]
    y_test = best["y_test"]
    best_calibrated_models = best["calibrated_models"]
    test_proba = ova_predict_proba(best_calibrated_models, X_test)
    y_pred = ova_predict(best_calibrated_models, X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    print(
        f"Selected best seed={best['seed']} | val_logloss={best['val_logloss']:.6f} | "
        f"val_acc={best['val_acc']:.6f} | val_macro_f1={best['val_macro_f1']:.6f} | "
        f"val_bal_acc={best['val_bal_acc']:.6f} | test_acc={best['test_acc']:.6f}"
    )

    print("\nXGBoost OVA Metrics (multiclass):")
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
    plt.title(f"Confusion Matrix (XGBoost OVA, K={num_class})")
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

    training_params = {
        "input": str(args.input),
        "test_size": args.test_size,
        "val_size": args.val_size,
        "select_by": args.select_by,
        "num_seeds": args.num_seeds,
        "seed_start": args.seed_start,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "gamma": args.gamma,
        "max_delta_step": args.max_delta_step,
        "early_stopping_rounds": args.early_stopping_rounds,
        "n_jobs": args.n_jobs,
        "verbosity": args.verbosity,
        "calibration": args.calibration,
        "pos_weight_multipliers": [float(v) for v in multipliers],
    }

    metrics = {
        "mode": "ova_calibrated",
        "best_seed": int(best["seed"]),
        "val_accuracy": float(best["val_acc"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "val_balanced_accuracy": float(best["val_bal_acc"]),
        "val_logloss": float(best["val_logloss"]),
        "test_accuracy": float(test_accuracy),
        "labels": labels,
        "label_mapping": label_mapping,
        "num_class": int(num_class),
        "pos_weight_multipliers": [float(v) for v in multipliers],
        "params": training_params,
        "classification_report": report_dict,
        "confusion_matrix": matrix,
    }

    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "mode": "ova_calibrated",
        "labels": labels,
        "label_mapping": label_mapping,
        "models": best_calibrated_models,
        "params": training_params,
        "best_seed": int(best["seed"]),
    }

    model_path = OUTPUT_DIR / "models" / "xgboost_ova_calibrated.joblib"
    joblib.dump(model_artifact, model_path)

    metrics_path = OUTPUT_DIR / "metrics" / "metrics_ova.json"
    save_json(metrics, metrics_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("Train OVA: python .\\src\\train_xgboost_ova.py --input .\\data\\data.csv")
    print(
        "Predict OVA: python .\\src\\predict_xgboost_ova.py --input <CSV> "
        "--model .\\outputs\\models\\xgboost_ova_calibrated.joblib --metrics .\\outputs\\metrics\\metrics_ova.json"
    )


if __name__ == "__main__":
    main()
     # ===== Quick Test Runs (user executes manually) =====
    # Example training command (macro_f1 selection):
    # py .\src\train_xgboost_ova.py --select_by macro_f1 --learning_rate 0.02 --n_estimators 15000 --max_depth 7 --min_child_weight 5 --pos_weight_multipliers 1.6,1.2,1.0,1.2
