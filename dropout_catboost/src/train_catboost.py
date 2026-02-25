from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# --- config (local folder defaults) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
RANDOM_STATE = 42
TEST_SIZE = 0.2


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


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
DEFAULT_3CLASS_MAPPING = {"dropout": 0, "enrolled": 1, "graduate": 2}


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _build_label_mapping(normalized_labels: pd.Series) -> dict[str, int]:
    labels = sorted(set(normalized_labels.dropna().unique()))
    if set(labels) == set(DEFAULT_3CLASS_MAPPING.keys()):
        return DEFAULT_3CLASS_MAPPING
    return {lab: i for i, lab in enumerate(labels)}


def _map_target(series: pd.Series, column_name: str) -> tuple[pd.Series, dict[str, int]]:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = _build_label_mapping(normalized)

    unexpected = sorted(set(normalized.dropna().unique()) - set(mapping.keys()))
    if unexpected:
        raise ValueError(f"Unexpected target labels in {column_name}: {unexpected}")

    y = normalized.map(mapping)
    return y, mapping


def _sqrt_inv_class_weights(y: pd.Series) -> list[float]:
    """
    Compute sqrt inverse frequency weights, normalized so the most frequent class has weight 1.0.
    Returns weights in class-id order: [w0, w1, ..., wK-1].
    """
    vc = y.value_counts().sort_index()
    if vc.empty:
        return []
    max_count = float(vc.max())
    weights = []
    for cls_id, count in vc.items():
        w = (max_count / float(count)) ** 0.5
        weights.append(float(w))
    # safety: ensure at least 1 class
    return weights


def _blend_weights(weights: list[float], alpha: float) -> list[float]:
    """
    Blend weights with 1.0: w' = 1 + alpha*(w - 1)
    alpha=0 => all 1.0, alpha=1 => original weights.
    """
    a = float(alpha)
    if a <= 0:
        return [1.0 for _ in weights]
    if a >= 1:
        return [float(w) for w in weights]
    return [1.0 + a * (float(w) - 1.0) for w in weights]


def _class_priors(y: pd.Series, n_classes: int) -> np.ndarray:
    """
    Compute class prior probabilities aligned to class ids 0..K-1
    """
    counts = y.value_counts().sort_index()
    priors = np.ones(n_classes, dtype=np.float64) * 1e-12

    for cls_id, count in counts.items():
        if 0 <= int(cls_id) < n_classes:
            priors[int(cls_id)] = float(count)

    priors = priors / priors.sum()
    return priors


def _apply_logit_adjustment(proba: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    """
    log p'(k|x) = log p(k|x) - tau * log prior(k)
    Then softmax normalize.
    """
    if tau <= 0:
        return proba

    p = np.clip(proba.astype(np.float64), 1e-12, 1.0)
    pri = np.clip(priors.astype(np.float64), 1e-12, 1.0)

    logits = np.log(p) - tau * np.log(pri)[None, :]
    logits -= logits.max(axis=1, keepdims=True)

    exp_vals = np.exp(logits)
    adjusted = exp_vals / exp_vals.sum(axis=1, keepdims=True)

    return adjusted


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(DATA_PATH),
        help="Path to input CSV (default: data/data.csv).",
    )

    # ---- training objective / class imbalance ----
    p.add_argument(
        "--loss",
        choices=["multiclass", "onevsall"],
        default="multiclass",
        help="Loss function: 'multiclass' or 'onevsall' (MultiClassOneVsAll).",
    )
    p.add_argument(
        "--auto_class_weights",
        choices=["None", "Balanced"],
        default="Balanced",
        help="Auto class weights in CatBoost: Balanced or None.",
    )
    p.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Custom class weights as comma-separated floats, e.g. 3.5,3.0,1.0,2.0 (order must match label mapping).",
    )
    p.add_argument(
        "--class_weight_mode",
        choices=["auto", "balanced", "sqrtbalanced", "custom", "none"],
        default="auto",
        help=(
            "Class weighting strategy. "
            "'auto' = sqrt inverse frequency computed from y_train, "
            "'balanced' = CatBoost auto_class_weights=Balanced, "
            "'sqrtbalanced' = CatBoost auto_class_weights=SqrtBalanced, "
            "'custom' = use --class_weights, "
            "'none' = no weighting."
        ),
    )
    p.add_argument(
        "--class_weight_alpha",
        type=float,
        default=1.0,
        help=(
            "Blend strength for class weights (0..1). "
            "alpha=1 uses full weights, alpha=0 disables weighting. "
            "Applied to 'auto' and 'custom' modes; "
            "ignored for 'balanced'/'sqrtbalanced' which use CatBoost internal weights."
        ),
    )
    p.add_argument(
        "--eval_metric",
        choices=["TotalF1", "Accuracy"],
        default="TotalF1",
        help="Metric used for early stopping / best iteration selection.",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="If set, run a lightweight manual hyperparameter search on a small grid and train the best model.",
    )
    p.add_argument(
        "--tune_trials",
        type=int,
        default=18,
        help="Number of random combinations to try in --tune mode (kept small to avoid long runs).",
    )
    p.add_argument(
        "--tune_seed",
        type=int,
        default=123,
        help="Seed for random sampling in --tune mode.",
    )
    p.add_argument(
        "--tune_metric",
        choices=["macro_f1", "weighted_f1", "accuracy"],
        default="macro_f1",
        help="Which metric to maximize in --tune mode.",
    )
    p.add_argument(
        "--postprocess",
        choices=["none", "logit_adjust"],
        default="none",
        help="Optional postprocessing on predicted probabilities."
    )
    p.add_argument(
        "--logit_tau",
        type=float,
        default=0.5,
        help="Strength of logit prior adjustment (0 disables). Recommended: 0.3 - 0.7"
    )

    # ---- core hyperparams ----
    p.add_argument("--iterations", type=int, default=5000, help="Training iterations.")
    p.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate.")
    p.add_argument("--depth", type=int, default=6, help="Tree depth.")
    p.add_argument("--l2_leaf_reg", type=float, default=3.0, help="L2 leaf regularization.")
    p.add_argument("--random_strength", type=float, default=1.0, help="Random strength (regularization).")
    p.add_argument("--bagging_temperature", type=float, default=1.0, help="Bagging temperature.")

    # ---- train control ----
    p.add_argument("--test_size", type=float, default=TEST_SIZE, help="Test split size.")
    p.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    p.add_argument("--early_stopping_rounds", type=int, default=200, help="Early stopping rounds.")
    p.add_argument("--verbose", type=int, default=200, help="Verbose frequency.")
    p.add_argument("--no_plot", action="store_true", help="Disable confusion matrix plot window.")

    return p.parse_args()


def _score_for_tune(y_true, y_pred, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, average="weighted"))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Unknown tune metric: {metric}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.input).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    print(f"Detected target column: {target_col}")
    raw_unique = sorted(df[target_col].dropna().unique().tolist())
    print(f"Raw unique values in {target_col}: {raw_unique}")
    normalized = df[target_col].astype(str).str.strip().str.lower()
    normalized_unique = sorted(normalized.dropna().unique().tolist())
    print("Normalized unique values:", normalized_unique)
    y, label_mapping = _map_target(df[target_col], target_col)
    X = df.drop(columns=[target_col])

    inv = {v: k for k, v in label_mapping.items()}
    label_names = [inv[i] for i in range(len(inv))]

    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]
    print("Class distribution after mapping:")
    print(y.value_counts())
    print("Label mapping:", label_mapping)

    # Detect categorical columns (non-numeric)
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Make categoricals explicit strings for safety
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols]

    loss_fn = "MultiClassOneVsAll" if args.loss == "onevsall" else "MultiClass"
    # Note: CatBoost eval_metric should be a metric name like 'TotalF1' or 'Accuracy'
    eval_metric = args.eval_metric
    # --- class weights resolution ---
    custom_weights = None
    auto_w = None if args.auto_class_weights == "None" else args.auto_class_weights

    if args.class_weight_mode == "custom":
        if args.class_weights is None:
            raise ValueError("--class_weight_mode=custom requires --class_weights")
        try:
            custom_weights = [float(x.strip()) for x in args.class_weights.split(",")]
            custom_weights = _blend_weights(custom_weights, args.class_weight_alpha)
            print(f"Using custom class_weights (alpha={args.class_weight_alpha}): {custom_weights}")
        except Exception as e:
            raise ValueError(f"Invalid --class_weights format: {args.class_weights}") from e
        auto_w = None

    elif args.class_weight_mode == "auto":
        custom_weights = _sqrt_inv_class_weights(y_train)
        custom_weights = _blend_weights(custom_weights, args.class_weight_alpha)
        print(f"Using auto sqrt-inv class_weights (alpha={args.class_weight_alpha}): {custom_weights}")
        auto_w = None

    elif args.class_weight_mode == "balanced":
        # use CatBoost's internal balancing
        custom_weights = None
        auto_w = "Balanced"
        print("Using CatBoost auto_class_weights=Balanced")

    elif args.class_weight_mode == "sqrtbalanced":
        custom_weights = None
        auto_w = "SqrtBalanced"
        print("Using CatBoost auto_class_weights=SqrtBalanced")

    elif args.class_weight_mode == "none":
        custom_weights = None
        auto_w = None
        print("No class weighting (class_weight_mode=none)")

    n_classes = len(label_names)
    priors_train = _class_priors(y_train, n_classes)
    print(f"Using postprocess={args.postprocess} with logit_tau={args.logit_tau}")

    best_params_override = None

    if args.tune:
        import random

        rng = random.Random(args.tune_seed)

        # small, sensible search space (kept conservative to avoid crazy runs)
        grid = {
            "depth": [5, 6, 7, 8, 9],
            "learning_rate": [0.02, 0.03, 0.05, 0.07],
            "l2_leaf_reg": [3.0, 6.0, 9.0, 12.0],
            "random_strength": [0.5, 1.0, 2.0, 3.0],
            "bagging_temperature": [0.2, 0.6, 1.0, 1.5],
            "min_data_in_leaf": [10, 20, 40],
            "border_count": [64, 128, 254],
            "rsm": [0.8, 0.9, 1.0],
        }

        keys = list(grid.keys())

        def sample_params():
            return {k: rng.choice(grid[k]) for k in keys}

        best_score = -1e9
        best_params = None

        print(f"\n[TUNE] Running {args.tune_trials} trials; optimizing {args.tune_metric} ...")
        for t in range(1, args.tune_trials + 1):
            p = sample_params()

            tmp = CatBoostClassifier(
                loss_function=loss_fn,
                eval_metric=eval_metric,
                custom_metric=["Accuracy", "TotalF1"],
                iterations=min(args.iterations, 2500),
                learning_rate=p["learning_rate"],
                depth=p["depth"],
                l2_leaf_reg=p["l2_leaf_reg"],
                random_strength=p["random_strength"],
                bagging_temperature=p["bagging_temperature"],
                bootstrap_type="Bayesian",
                rsm=p["rsm"],
                border_count=p["border_count"],
                min_data_in_leaf=p["min_data_in_leaf"],
                random_seed=args.seed,
                class_weights=custom_weights if custom_weights is not None else None,
                auto_class_weights=None if custom_weights is not None else auto_w,
                verbose=False,
                early_stopping_rounds=args.early_stopping_rounds,
                allow_writing_files=False,
            )

            tmp.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_test, y_test),
                use_best_model=True,
            )

            proba = tmp.predict_proba(X_test)
            if args.postprocess == "logit_adjust":
                proba = _apply_logit_adjustment(proba, priors_train, args.logit_tau)
            pred = np.argmax(proba, axis=1)
            score = _score_for_tune(y_test, pred, args.tune_metric)

            print(f"[TUNE] trial {t:02d}/{args.tune_trials} score={score:.4f} params={p}")

            if score > best_score:
                best_score = score
                best_params = p

        best_params_override = best_params
        print(f"[TUNE] Best score={best_score:.4f} best_params={best_params_override}\n")

    model = CatBoostClassifier(
        loss_function=loss_fn,
        eval_metric=eval_metric,
        custom_metric=["Accuracy", "TotalF1"],
        iterations=args.iterations,
        learning_rate=(best_params_override["learning_rate"] if best_params_override else 0.04),
        depth=(best_params_override["depth"] if best_params_override else 9),
        l2_leaf_reg=(best_params_override["l2_leaf_reg"] if best_params_override else 6.0),
        random_strength=(best_params_override["random_strength"] if best_params_override else args.random_strength),
        bagging_temperature=(best_params_override["bagging_temperature"] if best_params_override else args.bagging_temperature),
        bootstrap_type="Bayesian",
        rsm=(best_params_override["rsm"] if best_params_override else 0.95),                 # feature subsampling
        border_count=(best_params_override["border_count"] if best_params_override else 254),         # better numeric binning
        min_data_in_leaf=(best_params_override["min_data_in_leaf"] if best_params_override else 10),      # regularization
        random_seed=args.seed,
        class_weights=custom_weights if custom_weights is not None else None,
        auto_class_weights=None if custom_weights is not None else auto_w,
        verbose=args.verbose,
        early_stopping_rounds=args.early_stopping_rounds,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    proba = model.predict_proba(X_test)
    if args.postprocess == "logit_adjust":
        proba = _apply_logit_adjustment(proba, priors_train, args.logit_tau)
    y_pred = np.argmax(proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc),
        "eval_metric_used": eval_metric,
        "class_weight_mode": args.class_weight_mode,
        "auto_class_weights": auto_w,
        "class_weight_alpha": float(args.class_weight_alpha),
        "postprocess": args.postprocess,
        "logit_tau": float(args.logit_tau),
        "train_priors": priors_train.tolist(),
        "predict_proba_shape": list(proba.shape),
        "best_params_override": best_params_override,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report_dict,
        "label_mapping": label_mapping,
        "label_names": label_names,
    }

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, outdir / "metrics.json")

    model_path = outdir / "catboost_model.cbm"
    model.save_model(model_path)

    print(f"\nCatBoost Metrics ({len(label_names)}-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            digits=4,
        )
    )
    print(f"Extra metrics: macro_f1={macro_f1:.4f} | weighted_f1={weighted_f1:.4f} | balanced_acc={bal_acc:.4f}")

    if not args.no_plot:
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=label_names,
        )
        plt.title(f"Confusion Matrix ({len(label_names)}-class)")
        plt.show()
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
