from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (PROJECT_ROOT / "data" / "kuzilek_clean_plus.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "hierarchical").resolve()
DEFAULT_SWEEP_OUTPUT_JSON = (OUTPUT_DIR / "stage2_sweep_results.json").resolve()
DEFAULT_OVA_SWEEP_OUTPUT_JSON = (OUTPUT_DIR / "stage2_ova_sweep_results.json").resolve()

VALID_LABELS = {"distinction", "fail", "pass", "withdrawn"}
STAGE2_MAPPING = {"distinction": 0, "fail": 1, "withdrawn": 2}
STAGE2_LABELS = ["distinction", "fail", "withdrawn"]


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 2 hierarchical model: distinction/fail/withdrawn.")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to input CSV.")
    p.add_argument("--target", type=str, default="final_result", help="Target column name.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split size from Stage 2 subset.")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation split size from remaining train+val set.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--mode",
        choices=["single", "sweep", "ova_sweep"],
        default="single",
        help="Run mode: single model training or class-weight sweep scaffolding.",
    )
    p.add_argument(
        "--loss_mode",
        choices=["multiclass", "ova"],
        default="ova",
        help="Loss mode mapping: multiclass->MultiClass, ova->MultiClassOneVsAll.",
    )
    p.add_argument(
        "--base_weight_mode",
        choices=["balanced", "none", "manual"],
        default="balanced",
        help="Base weight mode used to compose sweep class weights.",
    )
    p.add_argument(
        "--manual_class_weights",
        type=str,
        default="",
        help="Manual base class weights for Stage2 ids [0,1,2] as 'w_dist,w_fail,w_withdrawn'.",
    )
    p.add_argument(
        "--sweep_fail_mult",
        type=str,
        default="1.0,1.2,1.4,1.6,1.8,2.0",
        help="Comma-separated multipliers for fail class weight in sweep mode.",
    )
    p.add_argument(
        "--sweep_withdrawn_mult",
        type=str,
        default="1.0,0.9,0.8,0.7,0.6",
        help="Comma-separated multipliers for withdrawn class weight in sweep mode.",
    )
    p.add_argument(
        "--sweep_dist_mult",
        type=str,
        default="1.0,1.1,1.2,1.3",
        help="Comma-separated multipliers for distinction class weight in sweep mode.",
    )
    p.add_argument(
        "--primary_metric",
        choices=["macro_f1", "balanced_accuracy"],
        default="macro_f1",
        help="Primary ranking metric for sweep mode.",
    )
    p.add_argument(
        "--fail_withdrawn_tradeoff",
        type=float,
        default=0.0,
        help="If >0, prefer configs improving fail_recall at small withdrawn_recall cost.",
    )
    p.add_argument("--save_top_k", type=int, default=10, help="Number of top sweep configs to retain.")
    p.add_argument(
        "--max_configs",
        type=int,
        default=0,
        help="Max configs to run in sweep modes (0 = run all).",
    )
    p.add_argument(
        "--min_withdrawn_recall",
        type=float,
        default=0.60,
        help="Withdrawn recall guardrail used for ranking/penalization in ova_sweep mode.",
    )
    p.add_argument(
        "--sweep_output_json",
        type=str,
        default=str(DEFAULT_SWEEP_OUTPUT_JSON),
        help="Path to save sweep results JSON.",
    )
    p.add_argument(
        "--ova_sweep_output_json",
        type=str,
        default=str(DEFAULT_OVA_SWEEP_OUTPUT_JSON),
        help="Path to save OVA sweep results JSON.",
    )
    p.add_argument("--sweep_depth", type=str, default="8,9,10,11", help="Depth grid for ova_sweep.")
    p.add_argument("--sweep_lr", type=str, default="0.02,0.03,0.04", help="Learning-rate grid for ova_sweep.")
    p.add_argument("--sweep_l2", type=str, default="6,9,12", help="L2 regularization grid for ova_sweep.")
    p.add_argument("--sweep_bag_temp", type=str, default="0.3,0.6,0.9", help="Bagging temperature grid for ova_sweep.")
    p.add_argument("--sweep_rstrength", type=str, default="1,2,3", help="Random-strength grid for ova_sweep.")
    p.add_argument("--sweep_iters", type=str, default="3000,5000", help="Iterations grid for ova_sweep.")
    p.add_argument(
        "--class_weight_mode",
        choices=["balanced", "manual", "none"],
        default="balanced",
        help="Class weight mode for Stage 2.",
    )
    p.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Manual class weights as comma-separated floats in order distinction,fail,withdrawn.",
    )
    p.add_argument("--loss_function", type=str, default="MultiClass", help="CatBoost loss function.")
    p.add_argument(
        "--eval_metric",
        type=str,
        default="TotalF1",
        help="Primary CatBoost eval metric. Falls back to MultiClass if unsupported.",
    )
    p.add_argument("--iterations", type=int, default=5000, help="Training iterations.")
    p.add_argument("--learning_rate", type=float, default=0.03, help="Learning rate.")
    p.add_argument("--depth", type=int, default=8, help="Tree depth.")
    p.add_argument("--l2_leaf_reg", type=float, default=9.0, help="L2 leaf regularization.")
    p.add_argument("--random_strength", type=float, default=2.0, help="Random strength.")
    p.add_argument("--bagging_temperature", type=float, default=0.6, help="Bagging temperature.")
    p.add_argument("--early_stopping_rounds", type=int, default=200, help="Early stopping rounds.")
    p.add_argument("--verbose", type=int, default=200, help="Verbose frequency.")
    return p.parse_args()


def _normalize_target(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.dropna().unique()) - VALID_LABELS)
    if unknown:
        raise ValueError(
            f"Unexpected labels in {column_name}: {unknown}. "
            f"Expected only: {sorted(VALID_LABELS)}"
        )
    return normalized


def compute_balanced_weights(y_train_ids: pd.Series, n_classes: int = 3) -> list[float]:
    total = float(len(y_train_ids))
    counts = y_train_ids.value_counts().sort_index()
    weights: list[float] = []
    for class_id in range(n_classes):
        count = float(counts.get(class_id, 0.0))
        if count <= 0:
            raise ValueError(f"Class id {class_id} missing in training split; cannot compute balanced weight.")
        weight = total / (float(n_classes) * count)
        weights.append(float(weight))
    return weights


def _compute_balanced_weights(y_train: pd.Series, n_classes: int) -> list[float]:
    return compute_balanced_weights(y_train, n_classes=n_classes)


def _parse_manual_weights(raw: str | None) -> list[float]:
    if raw is None:
        raise ValueError("--class_weight_mode=manual requires --class_weights")
    values = [float(v.strip()) for v in raw.split(",")]
    if len(values) != 3:
        raise ValueError("--class_weights must contain exactly 3 values for distinction,fail,withdrawn.")
    return values


def _parse_float_list_arg(raw: str, arg_name: str) -> list[float]:
    parts = [p.strip() for p in str(raw).split(",")]
    values: list[float] = []
    for part in parts:
        if part == "":
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"{arg_name} has a non-numeric value: '{part}'") from exc
    if not values:
        raise ValueError(f"{arg_name} must contain at least one numeric value.")
    return values


def _parse_int_list_arg(raw: str, arg_name: str) -> list[int]:
    parts = [p.strip() for p in str(raw).split(",")]
    values: list[int] = []
    for part in parts:
        if part == "":
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{arg_name} has a non-integer value: '{part}'") from exc
    if not values:
        raise ValueError(f"{arg_name} must contain at least one integer value.")
    return values


def _print_kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    width = 82
    print("\n" + "=" * width)
    print(title)
    print("-" * width)
    for key, value in rows:
        print(f"{key:<38} {value:>42}")
    print("=" * width)


def _print_sweep_topk(rows: list[dict], top_k: int) -> None:
    if not rows:
        return
    top = rows[:top_k]
    print("\nStage2 Sweep Top Configs")
    print("-" * 126)
    print(
        f"{'rank':>4} {'score':>9} {'macro_f1':>9} {'bal_acc':>9} "
        f"{'rec_dist':>9} {'rec_fail':>9} {'rec_wd':>9} {'pen':>5} {'weights':>55}"
    )
    print("-" * 126)
    for i, r in enumerate(top, start=1):
        w = r["class_weights_used"]
        w_str = f"[{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]"
        print(
            f"{i:>4d} {float(r['score']):>9.4f} {float(r['val_macro_f1']):>9.4f} {float(r['val_balanced_accuracy']):>9.4f} "
            f"{float(r['val_recall_distinction']):>9.4f} {float(r['val_recall_fail']):>9.4f} {float(r['val_recall_withdrawn']):>9.4f} "
            f"{('yes' if bool(r['penalized']) else 'no'):>5} {w_str:>55}"
        )
    print("-" * 126)


def _print_ova_sweep_topk(rows: list[dict], top_k: int) -> None:
    if not rows:
        return
    top = rows[:top_k]
    print("\nStage2 OVA Sweep Top Configs")
    print("-" * 150)
    print(
        f"{'rank':>4} {'score':>9} {'macro_f1':>9} {'bal_acc':>9} {'rec_fail':>9} {'rec_wd':>9} "
        f"{'overlap':>9} {'depth':>6} {'lr':>7} {'l2':>6} {'bag':>6} {'rstr':>6} {'iters':>7}"
    )
    print("-" * 150)
    for i, r in enumerate(top, start=1):
        p = r["params"]
        print(
            f"{i:>4d} {float(r['score']):>9.4f} {float(r['val_macro_f1']):>9.4f} {float(r['val_balanced_accuracy']):>9.4f} "
            f"{float(r['val_recall_fail']):>9.4f} {float(r['val_recall_withdrawn']):>9.4f} {float(r['val_overlap_rate']):>9.4f} "
            f"{int(p['depth']):>6d} {float(p['learning_rate']):>7.3f} {float(p['l2_leaf_reg']):>6.2f} "
            f"{float(p['bagging_temperature']):>6.2f} {float(p['random_strength']):>6.2f} {int(p['iterations']):>7d}"
        )
    print("-" * 150)


def _fit_with_metric_fallback(
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_feature_indices: list[int],
) -> tuple[CatBoostClassifier, str]:
    preferred_metric = model_params["eval_metric"]
    model = CatBoostClassifier(**model_params)
    try:
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        return model, str(preferred_metric)
    except Exception as exc:
        if str(preferred_metric) != "TotalF1":
            raise
        fallback_params = dict(model_params)
        fallback_params["eval_metric"] = "MultiClass"
        print(f"Falling back eval_metric from TotalF1 to MultiClass due to: {exc}")
        model = CatBoostClassifier(**fallback_params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        return model, "MultiClass"


def eval_multiclass(y_true_ids, y_pred_ids, label_names) -> dict:
    if list(label_names) != STAGE2_LABELS:
        raise ValueError(f"label_names must be {STAGE2_LABELS}, got {label_names}")

    labels = [0, 1, 2]
    y_true = np.asarray(y_true_ids)
    y_pred = np.asarray(y_pred_ids)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    fw_misclass = int(cm[1, 2])  # true fail predicted withdrawn
    wf_misclass = int(cm[2, 1])  # true withdrawn predicted fail
    overlap_total = int(fw_misclass + wf_misclass)
    denom = int(len(y_true))
    overlap_rate = float(overlap_total / denom) if denom > 0 else 0.0

    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {name: float(report[name]["recall"]) for name in label_names},
        "per_class_precision": {name: float(report[name]["precision"]) for name in label_names},
        "fw_misclass": fw_misclass,
        "wf_misclass": wf_misclass,
        "overlap_total": overlap_total,
        "overlap_rate": overlap_rate,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def train_one_config(X_train, y_train, X_val, y_val, X_test, y_test, class_weights, cat_params, seed) -> dict:
    params = dict(cat_params)
    cat_feature_indices = params.pop("cat_feature_indices", [])
    loss_mode = str(params.get("loss_mode", "multiclass")).strip().lower()
    if loss_mode not in {"multiclass", "ova"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    loss_function = "MultiClassOneVsAll" if loss_mode == "ova" else "MultiClass"

    model_params = {
        "loss_function": loss_function,
        "eval_metric": params.get("eval_metric", "TotalF1"),
        "iterations": int(params.get("iterations", 5000)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "depth": int(params.get("depth", 8)),
        "l2_leaf_reg": float(params.get("l2_leaf_reg", 9.0)),
        "random_strength": float(params.get("random_strength", 2.0)),
        "bagging_temperature": float(params.get("bagging_temperature", 0.6)),
        "random_seed": int(seed),
        "early_stopping_rounds": int(params.get("early_stopping_rounds", 200)),
        "verbose": int(params.get("verbose", 200)),
        "allow_writing_files": False,
        "class_weights": class_weights,
    }

    model, effective_eval_metric = _fit_with_metric_fallback(
        model_params=model_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_feature_indices=cat_feature_indices,
    )

    y_val_pred = np.argmax(model.predict_proba(X_val), axis=1)
    y_test_pred = np.argmax(model.predict_proba(X_test), axis=1)
    val_metrics = eval_multiclass(y_val, y_val_pred, STAGE2_LABELS)
    test_metrics = eval_multiclass(y_test, y_test_pred, STAGE2_LABELS)
    params_used = {
        "loss_mode": loss_mode,
        "loss_function": loss_function,
        "depth": int(params.get("depth", 8)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "l2_leaf_reg": float(params.get("l2_leaf_reg", 9.0)),
        "bagging_temperature": float(params.get("bagging_temperature", 0.6)),
        "random_strength": float(params.get("random_strength", 2.0)),
        "iterations": int(params.get("iterations", 5000)),
        "eval_metric_requested": str(params.get("eval_metric", "TotalF1")),
        "eval_metric_used": str(effective_eval_metric),
    }

    return {
        "model": model,
        "effective_eval_metric": effective_eval_metric,
        "params_used": params_used,
        "class_weights_used": None if class_weights is None else [float(w) for w in class_weights],
        "val_metrics": val_metrics,
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "test_metrics": test_metrics,
        "y_test_pred": y_test_pred,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be in (0, 1).")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val_size must be in (0, 1).")
    if args.save_top_k <= 0:
        raise ValueError("--save_top_k must be a positive integer.")
    if args.max_configs < 0:
        raise ValueError("--max_configs must be >= 0.")
    if not (0.0 <= args.min_withdrawn_recall <= 1.0):
        raise ValueError("--min_withdrawn_recall must be in [0, 1].")

    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {df.columns.tolist()}")

    y_all = _normalize_target(df[args.target], args.target)
    stage2_mask = y_all != "pass"
    if int(stage2_mask.sum()) == 0:
        raise ValueError("No NotPass samples found after filtering label != 'pass'.")

    df_stage2 = df.loc[stage2_mask].copy()
    y_stage2_raw = y_all.loc[stage2_mask].copy()
    invalid_stage2 = sorted(set(y_stage2_raw.unique()) - set(STAGE2_MAPPING.keys()))
    if invalid_stage2:
        raise ValueError(f"Unexpected Stage 2 labels: {invalid_stage2}")

    y_stage2 = y_stage2_raw.map(STAGE2_MAPPING).astype(int)
    X_stage2 = df_stage2.drop(columns=[args.target]).copy()

    cat_cols = X_stage2.select_dtypes(exclude=["number"]).columns.tolist()
    for col in cat_cols:
        X_stage2[col] = X_stage2[col].astype(str).fillna("NA")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_stage2,
        y_stage2,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_stage2,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_val,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    sweep_fail_mult: list[float] | None = None
    sweep_withdrawn_mult: list[float] | None = None
    sweep_dist_mult: list[float] | None = None
    sweep_output_path = Path(args.sweep_output_json).resolve()
    ova_sweep_output_path = Path(args.ova_sweep_output_json).resolve()
    ova_sweep_depth: list[int] | None = None
    ova_sweep_lr: list[float] | None = None
    ova_sweep_l2: list[float] | None = None
    ova_sweep_bag_temp: list[float] | None = None
    ova_sweep_rstrength: list[float] | None = None
    ova_sweep_iters: list[int] | None = None
    manual_base_weights: list[float] | None = None
    if args.mode == "sweep":
        sweep_fail_mult = _parse_float_list_arg(args.sweep_fail_mult, "--sweep_fail_mult")
        sweep_withdrawn_mult = _parse_float_list_arg(args.sweep_withdrawn_mult, "--sweep_withdrawn_mult")
        sweep_dist_mult = _parse_float_list_arg(args.sweep_dist_mult, "--sweep_dist_mult")
        print(
            "[INFO] mode=sweep scaffolding enabled. "
            "This step parses/validates sweep grids and keeps single-run training behavior unchanged."
        )
    if args.mode == "ova_sweep":
        ova_sweep_depth = _parse_int_list_arg(args.sweep_depth, "--sweep_depth")
        ova_sweep_lr = _parse_float_list_arg(args.sweep_lr, "--sweep_lr")
        ova_sweep_l2 = _parse_float_list_arg(args.sweep_l2, "--sweep_l2")
        ova_sweep_bag_temp = _parse_float_list_arg(args.sweep_bag_temp, "--sweep_bag_temp")
        ova_sweep_rstrength = _parse_float_list_arg(args.sweep_rstrength, "--sweep_rstrength")
        ova_sweep_iters = _parse_int_list_arg(args.sweep_iters, "--sweep_iters")
        if args.base_weight_mode == "manual":
            manual_base_weights = _parse_manual_weights(args.manual_class_weights or None)
        print("[INFO] mode=ova_sweep scaffolding enabled. Grids parsed and validated.")

    if args.class_weight_mode == "balanced":
        class_weights = compute_balanced_weights(y_train, n_classes=3)
    elif args.class_weight_mode == "manual":
        class_weights = _parse_manual_weights(args.class_weights)
    else:
        class_weights = None

    cat_params = {
        "loss_function": args.loss_function,
        "loss_mode": args.loss_mode,
        "eval_metric": args.eval_metric,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "bagging_temperature": args.bagging_temperature,
        "early_stopping_rounds": args.early_stopping_rounds,
        "verbose": args.verbose,
        "cat_feature_indices": cat_feature_indices,
    }

    if args.mode == "sweep":
        if sweep_fail_mult is None or sweep_withdrawn_mult is None or sweep_dist_mult is None:
            raise ValueError("Sweep multipliers were not initialized.")

        if args.base_weight_mode == "balanced":
            base_weights = compute_balanced_weights(y_train, n_classes=3)
        else:
            base_weights = [1.0, 1.0, 1.0]

        total_configs = len(sweep_dist_mult) * len(sweep_fail_mult) * len(sweep_withdrawn_mult)
        effective_total = total_configs if args.max_configs == 0 else min(total_configs, int(args.max_configs))
        print(f"[SWEEP] planned configs={total_configs}, running={effective_total}")
        progress_every = 10
        results: list[dict] = []
        idx = 0
        stop_early = False

        for dist_mult in sweep_dist_mult:
            for fail_mult in sweep_fail_mult:
                for withdrawn_mult in sweep_withdrawn_mult:
                    if args.max_configs > 0 and idx >= int(args.max_configs):
                        stop_early = True
                        break
                    idx += 1
                    class_weights_used = [
                        float(base_weights[0] * dist_mult),
                        float(base_weights[1] * fail_mult),
                        float(base_weights[2] * withdrawn_mult),
                    ]

                    run = train_one_config(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        class_weights=class_weights_used,
                        cat_params=cat_params,
                        seed=args.seed,
                    )
                    val_m = run["val_metrics"]
                    rec = val_m["per_class_recall"]
                    val_macro_f1 = float(val_m["macro_f1"])
                    val_bal_acc = float(val_m["balanced_accuracy"])
                    val_fail_recall = float(rec["fail"])
                    val_wd_recall = float(rec["withdrawn"])
                    val_dist_recall = float(rec["distinction"])

                    score = val_macro_f1 if args.primary_metric == "macro_f1" else val_bal_acc
                    if float(args.fail_withdrawn_tradeoff) > 0:
                        score += float(args.fail_withdrawn_tradeoff) * (val_fail_recall - val_wd_recall)

                    penalized = val_wd_recall < 0.60
                    results.append(
                        {
                            "config_index": idx,
                            "dist_mult": float(dist_mult),
                            "fail_mult": float(fail_mult),
                            "withdrawn_mult": float(withdrawn_mult),
                            "class_weights_used": class_weights_used,
                            "score": float(score),
                            "primary_metric": args.primary_metric,
                            "penalized": bool(penalized),
                            "val_macro_f1": val_macro_f1,
                            "val_balanced_accuracy": val_bal_acc,
                            "val_recall_distinction": val_dist_recall,
                            "val_recall_fail": val_fail_recall,
                            "val_recall_withdrawn": val_wd_recall,
                            "val_confusion_matrix": val_m["confusion_matrix"],
                            "val_metrics": val_m,
                            "effective_eval_metric": run["effective_eval_metric"],
                            "best_iteration": int(run["model"].get_best_iteration()),
                        }
                    )

                    if idx % progress_every == 0 or idx == effective_total:
                        print(f"[SWEEP] {idx}/{effective_total} configs done")
                if stop_early:
                    break
            if stop_early:
                break

        ranked = sorted(
            results,
            key=lambda r: (
                0 if not bool(r["penalized"]) else 1,
                -float(r["score"]),
                -float(r["val_recall_fail"]),
                -float(r["val_recall_withdrawn"]),
            ),
        )

        sweep_payload = {
            "mode": "sweep",
            "base_weight_mode": args.base_weight_mode,
            "base_weights": base_weights,
            "primary_metric": args.primary_metric,
            "fail_withdrawn_tradeoff": float(args.fail_withdrawn_tradeoff),
            "withdrawn_recall_penalty_threshold": 0.60,
            "total_configs_planned": int(total_configs),
            "total_configs_executed": int(len(results)),
            "top_k": int(args.save_top_k),
            "results": ranked,
        }
        save_json(sweep_payload, sweep_output_path)
        _print_sweep_topk(ranked, args.save_top_k)
        print(f"Saved sweep results: {sweep_output_path}")

        best = ranked[0]
        best_weights = [float(w) for w in best["class_weights_used"]]
        best_iter = int(best.get("best_iteration", -1))
        final_iterations = best_iter if best_iter > 0 else int(args.iterations)

        X_trainval = pd.concat([X_train, X_val], axis=0)
        y_trainval = pd.concat([y_train, y_val], axis=0)

        final_model_params = {
            "loss_function": args.loss_function,
            "eval_metric": args.eval_metric,
            "iterations": int(final_iterations),
            "learning_rate": float(args.learning_rate),
            "depth": int(args.depth),
            "l2_leaf_reg": float(args.l2_leaf_reg),
            "random_strength": float(args.random_strength),
            "bagging_temperature": float(args.bagging_temperature),
            "random_seed": int(args.seed),
            "early_stopping_rounds": None,
            "verbose": int(args.verbose),
            "allow_writing_files": False,
            "class_weights": best_weights,
        }
        final_model = CatBoostClassifier(**final_model_params)
        final_model.fit(
            X_trainval,
            y_trainval,
            cat_features=cat_feature_indices,
        )

        y_test_pred = np.argmax(final_model.predict_proba(X_test), axis=1)
        final_test_metrics = eval_multiclass(y_test, y_test_pred, STAGE2_LABELS)

        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "stage2_model.cbm"
        metrics_path = output_dir / "stage2_metrics.json"
        labels_path = output_dir / "stage2_labels.json"

        final_model.save_model(model_path)
        save_json(
            {
                "mapping": STAGE2_MAPPING,
                "inverse_mapping": {str(v): k for k, v in STAGE2_MAPPING.items()},
                "class_order": STAGE2_LABELS,
            },
            labels_path,
        )

        final_metrics_payload = {
            "mode": "sweep",
            "dataset": {
                "input_path": str(input_path),
                "original_shape": [int(df.shape[0]), int(df.shape[1])],
                "stage2_shape": [int(df_stage2.shape[0]), int(df_stage2.shape[1])],
                "target_column": args.target,
                "filter": "label != 'pass'",
            },
            "splits": {
                "trainval_size": int(len(X_trainval)),
                "test_size": int(len(X_test)),
                "test_fraction_stage2": float(args.test_size),
                "seed": int(args.seed),
            },
            "sweep_selection": {
                "primary_metric": args.primary_metric,
                "fail_withdrawn_tradeoff": float(args.fail_withdrawn_tradeoff),
                "withdrawn_recall_penalty_threshold": 0.60,
                "selected_rank": 1,
                "selected_config": best,
                "selected_multipliers": {
                    "dist_mult": float(best["dist_mult"]),
                    "fail_mult": float(best["fail_mult"]),
                    "withdrawn_mult": float(best["withdrawn_mult"]),
                },
                "final_class_weights": best_weights,
                "val_best_score": float(best["score"]),
                "val_metrics_at_selection": best["val_metrics"],
            },
            "final_training": {
                "strategy": "train_plus_val_no_early_stopping",
                "iterations_used": int(final_iterations),
                "selected_best_iteration_from_sweep": int(best_iter),
                "early_stopping_used": False,
                "eval_set_used": "none",
                "test_set_used_for_training": False,
            },
            "test_metrics": final_test_metrics,
            "labels": {
                "mapping": STAGE2_MAPPING,
                "class_names": STAGE2_LABELS,
            },
            "artifacts": {
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "labels_path": str(labels_path),
                "sweep_output_json": str(sweep_output_path),
            },
        }
        save_json(final_metrics_payload, metrics_path)

        rec = final_test_metrics["per_class_recall"]
        print("\nStage2 Sweep Final Summary")
        print(
            "Best multipliers: "
            f"dist={best['dist_mult']:.3f}, fail={best['fail_mult']:.3f}, withdrawn={best['withdrawn_mult']:.3f}"
        )
        print(
            "Best class weights: "
            f"[{best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}]"
        )
        print(
            "Test metrics: "
            f"macro_f1={float(final_test_metrics['macro_f1']):.4f}, "
            f"fail_recall={float(rec['fail']):.4f}, "
            f"withdrawn_recall={float(rec['withdrawn']):.4f}"
        )
        print(f"Saved model: {model_path}")
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved labels: {labels_path}")
        return

    if args.mode == "ova_sweep":
        if (
            ova_sweep_depth is None
            or ova_sweep_lr is None
            or ova_sweep_l2 is None
            or ova_sweep_bag_temp is None
            or ova_sweep_rstrength is None
            or ova_sweep_iters is None
        ):
            raise ValueError("OVA sweep grids were not initialized.")

        if args.base_weight_mode == "balanced":
            fixed_class_weights = compute_balanced_weights(y_train, n_classes=3)
        elif args.base_weight_mode == "none":
            fixed_class_weights = [1.0, 1.0, 1.0]
        else:
            if manual_base_weights is None:
                manual_base_weights = _parse_manual_weights(args.manual_class_weights or None)
            fixed_class_weights = [float(w) for w in manual_base_weights]

        total_configs = (
            len(ova_sweep_depth)
            * len(ova_sweep_lr)
            * len(ova_sweep_l2)
            * len(ova_sweep_bag_temp)
            * len(ova_sweep_rstrength)
            * len(ova_sweep_iters)
        )
        effective_total = total_configs if args.max_configs == 0 else min(total_configs, int(args.max_configs))
        print(f"[OVA_SWEEP] planned configs={total_configs}, running={effective_total}")
        progress_every = 10
        results: list[dict] = []
        config_id = 0
        selected_loss_mode = args.loss_mode if args.loss_mode in {"multiclass", "ova"} else "ova"
        stop_early = False

        for depth in ova_sweep_depth:
            for lr in ova_sweep_lr:
                for l2 in ova_sweep_l2:
                    for bag in ova_sweep_bag_temp:
                        for rstr in ova_sweep_rstrength:
                            for iters in ova_sweep_iters:
                                if args.max_configs > 0 and config_id >= int(args.max_configs):
                                    stop_early = True
                                    break
                                config_id += 1
                                run_params = dict(cat_params)
                                run_params.update(
                                    {
                                        "loss_mode": selected_loss_mode,
                                        "depth": int(depth),
                                        "learning_rate": float(lr),
                                        "l2_leaf_reg": float(l2),
                                        "bagging_temperature": float(bag),
                                        "random_strength": float(rstr),
                                        "iterations": int(iters),
                                    }
                                )
                                run = train_one_config(
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_val=X_val,
                                    y_val=y_val,
                                    X_test=X_test,
                                    y_test=y_test,
                                    class_weights=fixed_class_weights,
                                    cat_params=run_params,
                                    seed=args.seed,
                                )

                                val_m = run["val_metrics"]
                                rec = val_m["per_class_recall"]
                                val_macro_f1 = float(val_m["macro_f1"])
                                val_bal_acc = float(val_m["balanced_accuracy"])
                                val_rec_fail = float(rec["fail"])
                                val_rec_wd = float(rec["withdrawn"])
                                val_overlap_rate = float(val_m["overlap_rate"])

                                base_score = val_macro_f1 if args.primary_metric == "macro_f1" else val_bal_acc
                                tradeoff_score = float(args.fail_withdrawn_tradeoff) * (val_rec_fail - val_rec_wd)
                                score = float(base_score + tradeoff_score)
                                guardrail_penalized = val_rec_wd < float(args.min_withdrawn_recall)
                                if guardrail_penalized:
                                    score -= 0.05

                                results.append(
                                    {
                                        "config_id": int(config_id),
                                        "params": {
                                            "depth": int(depth),
                                            "learning_rate": float(lr),
                                            "l2_leaf_reg": float(l2),
                                            "bagging_temperature": float(bag),
                                            "random_strength": float(rstr),
                                            "iterations": int(iters),
                                            "loss_mode": selected_loss_mode,
                                        },
                                        "class_weights": [float(w) for w in fixed_class_weights],
                                        "score": float(score),
                                        "base_score": float(base_score),
                                        "tradeoff_score": float(tradeoff_score),
                                        "guardrail_penalized": bool(guardrail_penalized),
                                        "val_macro_f1": val_macro_f1,
                                        "val_balanced_accuracy": val_bal_acc,
                                        "val_recall_fail": val_rec_fail,
                                        "val_recall_withdrawn": val_rec_wd,
                                        "val_recall_distinction": float(rec["distinction"]),
                                        "val_overlap_total": int(val_m["overlap_total"]),
                                        "val_overlap_rate": val_overlap_rate,
                                        "val_confusion_matrix": val_m["confusion_matrix"],
                                        "val_metrics": val_m,
                                    }
                                )

                                if config_id % progress_every == 0 or config_id == effective_total:
                                    print(f"[OVA_SWEEP] {config_id}/{effective_total} configs done")
                            if stop_early:
                                break
                        if stop_early:
                            break
                    if stop_early:
                        break
                if stop_early:
                    break
            if stop_early:
                break

        ranked = sorted(
            results,
            key=lambda r: (
                -float(r["score"]),
                -float(r["val_macro_f1"]),
                -float(r["val_recall_fail"]),
                float(r["val_overlap_rate"]),
                -float(r["val_recall_withdrawn"]),
            ),
        )
        payload = {
            "mode": "ova_sweep",
            "loss_mode": selected_loss_mode,
            "base_weight_mode": args.base_weight_mode,
            "class_weights_fixed": [float(w) for w in fixed_class_weights],
            "primary_metric": args.primary_metric,
            "fail_withdrawn_tradeoff": float(args.fail_withdrawn_tradeoff),
            "min_withdrawn_recall": float(args.min_withdrawn_recall),
            "guardrail_penalty": 0.05,
            "total_configs_planned": int(total_configs),
            "total_configs_executed": int(len(results)),
            "top_k": int(args.save_top_k),
            "results": ranked,
        }
        save_json(payload, ova_sweep_output_path)
        _print_ova_sweep_topk(ranked, args.save_top_k)
        print(f"Saved OVA sweep results: {ova_sweep_output_path}")

        best = ranked[0]
        best_params = dict(best["params"])
        best_weights = [float(w) for w in best["class_weights"]]

        X_trainval = pd.concat([X_train, X_val], axis=0)
        y_trainval = pd.concat([y_train, y_val], axis=0)

        best_loss_mode = str(best_params.get("loss_mode", "ova")).strip().lower()
        best_loss_function = "MultiClassOneVsAll" if best_loss_mode == "ova" else "MultiClass"

        final_model = CatBoostClassifier(
            loss_function=best_loss_function,
            eval_metric=args.eval_metric,
            iterations=int(best_params["iterations"]),
            learning_rate=float(best_params["learning_rate"]),
            depth=int(best_params["depth"]),
            l2_leaf_reg=float(best_params["l2_leaf_reg"]),
            random_strength=float(best_params["random_strength"]),
            bagging_temperature=float(best_params["bagging_temperature"]),
            random_seed=int(args.seed),
            early_stopping_rounds=None,
            verbose=int(args.verbose),
            allow_writing_files=False,
            class_weights=best_weights,
        )
        final_model.fit(
            X_trainval,
            y_trainval,
            cat_features=cat_feature_indices,
        )

        y_test_pred = np.argmax(final_model.predict_proba(X_test), axis=1)
        final_test_metrics = eval_multiclass(y_test, y_test_pred, STAGE2_LABELS)

        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "stage2_model.cbm"
        metrics_path = output_dir / "stage2_metrics.json"
        labels_path = output_dir / "stage2_labels.json"

        final_model.save_model(model_path)
        save_json(
            {
                "mapping": STAGE2_MAPPING,
                "inverse_mapping": {str(v): k for k, v in STAGE2_MAPPING.items()},
                "class_order": STAGE2_LABELS,
            },
            labels_path,
        )

        final_metrics_payload = {
            "mode": "ova_sweep",
            "dataset": {
                "input_path": str(input_path),
                "original_shape": [int(df.shape[0]), int(df.shape[1])],
                "stage2_shape": [int(df_stage2.shape[0]), int(df_stage2.shape[1])],
                "target_column": args.target,
                "filter": "label != 'pass'",
            },
            "splits": {
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "trainval_size": int(len(X_trainval)),
                "test_size": int(len(X_test)),
                "test_fraction_stage2": float(args.test_size),
                "val_fraction_remaining": float(args.val_size),
                "seed": int(args.seed),
            },
            "selection": {
                "primary_metric": args.primary_metric,
                "fail_withdrawn_tradeoff": float(args.fail_withdrawn_tradeoff),
                "min_withdrawn_recall": float(args.min_withdrawn_recall),
                "selected_rank": 1,
                "val_best_score": float(best["score"]),
                "val_metrics_at_selection": best["val_metrics"],
                "selected_hyperparams": {
                    "depth": int(best_params["depth"]),
                    "learning_rate": float(best_params["learning_rate"]),
                    "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
                    "bagging_temperature": float(best_params["bagging_temperature"]),
                    "random_strength": float(best_params["random_strength"]),
                    "iterations": int(best_params["iterations"]),
                },
                "selected_loss_mode": best_loss_mode,
                "selected_loss_function": best_loss_function,
                "class_weights_used": best_weights,
            },
            "final_training": {
                "strategy": "train_plus_val_fixed_iterations_no_early_stopping",
                "early_stopping_used": False,
                "eval_set_used": "none",
                "test_set_used_for_training": False,
            },
            "test_metrics": {
                "macro_f1": float(final_test_metrics["macro_f1"]),
                "balanced_accuracy": float(final_test_metrics["balanced_accuracy"]),
                "per_class_recall": final_test_metrics["per_class_recall"],
                "overlap_rate": float(final_test_metrics["overlap_rate"]),
                "overlap_total": int(final_test_metrics["overlap_total"]),
                "confusion_matrix": final_test_metrics["confusion_matrix"],
                "classification_report": final_test_metrics["classification_report"],
                "accuracy": float(final_test_metrics["accuracy"]),
                "weighted_f1": float(final_test_metrics["weighted_f1"]),
                "fw_misclass": int(final_test_metrics["fw_misclass"]),
                "wf_misclass": int(final_test_metrics["wf_misclass"]),
            },
            "labels": {
                "mapping": STAGE2_MAPPING,
                "class_names": STAGE2_LABELS,
            },
            "artifacts": {
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "labels_path": str(labels_path),
                "ova_sweep_output_json": str(ova_sweep_output_path),
            },
        }
        save_json(final_metrics_payload, metrics_path)

        rec = final_test_metrics["per_class_recall"]
        print("\nStage2 OVA Sweep Final Summary")
        print(f"Best config params: {best_params}")
        print(
            "Test metrics: "
            f"macro_f1={float(final_test_metrics['macro_f1']):.4f}, "
            f"fail_recall={float(rec['fail']):.4f}, "
            f"withdrawn_recall={float(rec['withdrawn']):.4f}, "
            f"overlap_rate={float(final_test_metrics['overlap_rate']):.4f}"
        )
        print(f"Saved model: {model_path}")
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved labels: {labels_path}")
        return

    run = train_one_config(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_weights=class_weights,
        cat_params=cat_params,
        seed=args.seed,
    )
    model = run["model"]
    effective_eval_metric = run["effective_eval_metric"]
    val_metrics = run["val_metrics"]
    test_metrics = run["test_metrics"]
    y_pred = run["y_test_pred"]

    acc = float(test_metrics["accuracy"])
    bal_acc = float(test_metrics["balanced_accuracy"])
    macro_f1 = float(test_metrics["macro_f1"])
    weighted_f1 = float(test_metrics["weighted_f1"])
    cm = np.asarray(test_metrics["confusion_matrix"], dtype=int)
    report_dict = test_metrics["classification_report"]

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "stage2_model.cbm"
    metrics_path = output_dir / "stage2_metrics.json"
    labels_path = output_dir / "stage2_labels.json"

    model.save_model(model_path)
    save_json(
        {
            "mapping": STAGE2_MAPPING,
            "inverse_mapping": {str(v): k for k, v in STAGE2_MAPPING.items()},
            "class_order": STAGE2_LABELS,
        },
        labels_path,
    )

    metrics = {
        "dataset": {
            "input_path": str(input_path),
            "original_shape": [int(df.shape[0]), int(df.shape[1])],
            "stage2_shape": [int(df_stage2.shape[0]), int(df_stage2.shape[1])],
            "target_column": args.target,
            "filter": "label != 'pass'",
        },
        "splits": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "test_fraction_stage2": float(args.test_size),
            "val_fraction_remaining": float(args.val_size),
            "seed": int(args.seed),
        },
        "class_weighting": {
            "mode": args.class_weight_mode,
            "class_weights": class_weights,
            "train_counts": y_train.value_counts().sort_index().to_dict(),
        },
        "model": {
            "loss_function": args.loss_function,
            "eval_metric_requested": args.eval_metric,
            "eval_metric_used": effective_eval_metric,
            "iterations": int(args.iterations),
            "learning_rate": float(args.learning_rate),
            "depth": int(args.depth),
            "l2_leaf_reg": float(args.l2_leaf_reg),
            "random_strength": float(args.random_strength),
            "bagging_temperature": float(args.bagging_temperature),
            "early_stopping_rounds": int(args.early_stopping_rounds),
        },
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        },
        "val_metrics": val_metrics,
        "labels": {
            "mapping": STAGE2_MAPPING,
            "class_names": STAGE2_LABELS,
        },
        "run_mode": {
            "mode": args.mode,
            "loss_mode": args.loss_mode,
            "base_weight_mode": args.base_weight_mode,
            "manual_class_weights": manual_base_weights,
            "primary_metric": args.primary_metric,
            "fail_withdrawn_tradeoff": float(args.fail_withdrawn_tradeoff),
            "min_withdrawn_recall": float(args.min_withdrawn_recall),
            "save_top_k": int(args.save_top_k),
            "sweep_output_json": str(sweep_output_path),
            "ova_sweep_output_json": str(ova_sweep_output_path),
            "sweep_fail_mult": sweep_fail_mult,
            "sweep_withdrawn_mult": sweep_withdrawn_mult,
            "sweep_dist_mult": sweep_dist_mult,
            "sweep_depth": ova_sweep_depth,
            "sweep_lr": ova_sweep_lr,
            "sweep_l2": ova_sweep_l2,
            "sweep_bag_temp": ova_sweep_bag_temp,
            "sweep_rstrength": ova_sweep_rstrength,
            "sweep_iters": ova_sweep_iters,
        },
    }
    save_json(metrics, metrics_path)

    _print_kv_table(
        "Stage 2 Dataset",
        [
            ("Input CSV", str(input_path)),
            ("Original shape (rows, cols)", f"{df.shape[0]}, {df.shape[1]}"),
            ("Stage2 shape (rows, cols)", f"{df_stage2.shape[0]}, {df_stage2.shape[1]}"),
            ("Filter", "final_result != pass"),
        ],
    )
    _print_kv_table(
        "Stage 2 Splits",
        [
            ("Train size", str(len(X_train))),
            ("Validation size", str(len(X_val))),
            ("Test size", str(len(X_test))),
            ("Seed", str(args.seed)),
        ],
    )
    _print_kv_table(
        "Stage 2 Metrics (Test)",
        [
            ("Eval metric used", effective_eval_metric),
            ("Class weight mode", args.class_weight_mode),
            ("Accuracy", f"{acc:.4f}"),
            ("Balanced accuracy", f"{bal_acc:.4f}"),
            ("Macro F1", f"{macro_f1:.4f}"),
            ("Weighted F1", f"{weighted_f1:.4f}"),
        ],
    )
    _print_kv_table(
        "Confusion Matrix (rows=true, cols=pred)",
        [
            ("[distinction, distinction]", str(int(cm[0, 0]))),
            ("[distinction, fail]", str(int(cm[0, 1]))),
            ("[distinction, withdrawn]", str(int(cm[0, 2]))),
            ("[fail, distinction]", str(int(cm[1, 0]))),
            ("[fail, fail]", str(int(cm[1, 1]))),
            ("[fail, withdrawn]", str(int(cm[1, 2]))),
            ("[withdrawn, distinction]", str(int(cm[2, 0]))),
            ("[withdrawn, fail]", str(int(cm[2, 1]))),
            ("[withdrawn, withdrawn]", str(int(cm[2, 2]))),
        ],
    )

    print("Classification report (3-class):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=STAGE2_LABELS,
            digits=4,
            zero_division=0,
        )
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved labels: {labels_path}")


if __name__ == "__main__":
    main()
