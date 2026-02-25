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
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (PROJECT_ROOT / "data" / "kuzilek_clean_plus.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "hierarchical").resolve()
DEFAULT_STAGE2_MODEL_PATH = (OUTPUT_DIR / "stage2_model.cbm").resolve()
DEFAULT_STAGE2_LABELS_PATH = (OUTPUT_DIR / "stage2_labels.json").resolve()
DEFAULT_STAGE1_JOINT_SWEEP_PATH = (OUTPUT_DIR / "stage1_joint_sweep_results.json").resolve()

VALID_LABELS = {"distinction", "fail", "pass", "withdrawn"}
NEGATIVE_LABELS = {"distinction", "fail", "withdrawn"}
LABEL_ORDER_4 = ["distinction", "fail", "pass", "withdrawn"]
STAGE1_MAPPING_4 = {"distinction": 0, "fail": 1, "pass": 2, "withdrawn": 3}


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 hierarchical model: pass vs notpass.")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to input CSV.")
    p.add_argument("--target", type=str, default="final_result", help="Target column name.")
    p.add_argument(
        "--stage1_mode",
        choices=["binary", "ova4"],
        default="ova4",
        help="Stage1 training mode: binary pass-vs-notpass or 4-class OVA.",
    )
    p.add_argument(
        "--stage1_loss",
        choices=["logloss", "ova"],
        default="ova",
        help="Stage1 loss preference (ova4 mode forces MultiClassOneVsAll).",
    )
    p.add_argument("--test_size", type=float, default=0.2, help="Test split size from full dataset.")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation split size from remaining train+val set.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--class_weight_mode",
        choices=["balanced", "manual", "none"],
        default="balanced",
        help="Class-weight mode for stage1 ova4 training.",
    )
    p.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Manual class weights as comma-separated floats in order distinction,fail,pass,withdrawn.",
    )
    p.add_argument(
        "--threshold_metric",
        choices=["f1", "balanced_accuracy"],
        default="f1",
        help="Metric used for threshold tuning on validation set.",
    )
    p.add_argument(
        "--stage2_model_path",
        type=str,
        default=str(DEFAULT_STAGE2_MODEL_PATH),
        help="Path to pretrained Stage2 model (.cbm) used for pipeline-level threshold optimization.",
    )
    p.add_argument(
        "--stage2_labels_path",
        type=str,
        default=str(DEFAULT_STAGE2_LABELS_PATH),
        help="Path to Stage2 labels JSON mapping for class-id to label decoding.",
    )
    p.add_argument(
        "--optimize_mode",
        choices=["binary_f1", "pipeline_macro_f1"],
        default="pipeline_macro_f1",
        help="Threshold optimization objective mode.",
    )
    p.add_argument(
        "--pass_recall_min",
        type=float,
        default=0.83,
        help="Minimum Pass recall constraint (for pipeline-level threshold optimization on validation).",
    )
    p.add_argument("--optimize_joint", type=int, default=1, help="Enable joint routing sweep for ova4 mode (1=true).")
    p.add_argument("--grid_t_pass", type=str, default="0.45,0.50,0.55,0.60,0.65", help="Grid for t_pass.")
    p.add_argument("--grid_t_notpass", type=str, default="0.25,0.30,0.35,0.40,0.45", help="Grid for t_notpass.")
    p.add_argument("--grid_t_margin", type=str, default="0.00,0.03,0.05,0.08,0.10", help="Grid for t_margin.")
    p.add_argument("--dist_recall_min", type=float, default=0.12, help="Soft guardrail for distinction recall on VAL.")
    p.add_argument(
        "--route_policy",
        choices=["pass_threshold", "margin_gate"],
        default="margin_gate",
        help="Routing policy for Stage1 OVA probabilities.",
    )
    p.add_argument("--t_pass", type=float, default=0.55, help="Pass threshold for routing.")
    p.add_argument("--t_notpass", type=float, default=0.40, help="NotPass confidence threshold for margin_gate.")
    p.add_argument("--t_margin", type=float, default=0.05, help="Required pass margin over max notpass probability.")
    p.add_argument(
        "--route_default",
        choices=["pass", "notpass"],
        default="pass",
        help="Fallback route when margin_gate thresholds are not met.",
    )
    p.add_argument("--threshold_min", type=float, default=0.05, help="Minimum threshold in sweep.")
    p.add_argument("--threshold_max", type=float, default=0.95, help="Maximum threshold in sweep.")
    p.add_argument("--threshold_step", type=float, default=0.01, help="Threshold step size in sweep.")
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


def _to_binary_target(labels_4class: pd.Series) -> pd.Series:
    y_bin = (labels_4class == "pass").astype(int)
    return y_bin


def _to_stage1_4class_ids(labels_4class: pd.Series) -> pd.Series:
    return labels_4class.map(STAGE1_MAPPING_4).astype(int)


def _compute_balanced_weights(y_train_ids: pd.Series, n_classes: int) -> list[float]:
    total = float(len(y_train_ids))
    counts = y_train_ids.value_counts().sort_index()
    weights: list[float] = []
    for class_id in range(n_classes):
        count = float(counts.get(class_id, 0.0))
        if count <= 0:
            raise ValueError(f"Class id {class_id} missing in training split; cannot compute balanced weight.")
        weights.append(float(total / (float(n_classes) * count)))
    return weights


def _parse_manual_weights(raw: str | None, expected_len: int) -> list[float]:
    if raw is None:
        raise ValueError("--class_weight_mode=manual requires --class_weights")
    values = [float(v.strip()) for v in raw.split(",")]
    if len(values) != expected_len:
        raise ValueError(f"--class_weights must contain exactly {expected_len} values.")
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


def route_pass_mask(
    proba4: np.ndarray,
    policy: str,
    t_pass: float,
    t_notpass: float,
    t_margin: float,
    route_default: str,
) -> np.ndarray:
    p = np.asarray(proba4, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError(f"route_pass_mask expects proba shape (N,4), got {p.shape}")

    pass_idx = STAGE1_MAPPING_4["pass"]  # id=2
    p_pass = p[:, pass_idx]
    notpass_idx = [STAGE1_MAPPING_4["distinction"], STAGE1_MAPPING_4["fail"], STAGE1_MAPPING_4["withdrawn"]]
    notpass_max = p[:, notpass_idx].max(axis=1)

    if policy == "pass_threshold":
        return p_pass >= float(t_pass)

    if policy == "margin_gate":
        cond_pass = (p_pass >= float(t_pass)) & ((p_pass - notpass_max) >= float(t_margin))
        cond_notpass = notpass_max >= float(t_notpass)
        if route_default == "pass":
            fallback = np.ones(p.shape[0], dtype=bool)
        elif route_default == "notpass":
            fallback = np.zeros(p.shape[0], dtype=bool)
        else:
            raise ValueError(f"Unsupported route_default: {route_default}")
        out = np.where(cond_pass, True, np.where(cond_notpass, False, fallback))
        return out.astype(bool)

    raise ValueError(f"Unsupported route_policy: {policy}")


def _score_threshold(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported threshold metric: {metric}")


def _find_best_threshold(
    y_val: np.ndarray,
    proba_pass: np.ndarray,
    metric: str,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, float]:
    thresholds = np.arange(float(threshold_min), float(threshold_max) + 1e-12, float(threshold_step))
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        pred = (proba_pass >= threshold).astype(int)
        score = _score_threshold(y_val, pred, metric)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def _load_stage2_id_to_label(labels_path: Path) -> dict[int, str]:
    data = json.loads(labels_path.read_text(encoding="utf-8"))

    raw_mapping = data.get("mapping")
    if isinstance(raw_mapping, dict):
        keys_are_intlike = True
        for k in raw_mapping.keys():
            try:
                int(str(k))
            except Exception:
                keys_are_intlike = False
                break

        if keys_are_intlike:
            parsed = {int(str(k)): str(v).strip().lower() for k, v in raw_mapping.items()}
        else:
            parsed = {int(v): str(k).strip().lower() for k, v in raw_mapping.items()}
        return parsed

    inv_mapping = data.get("inverse_mapping")
    if isinstance(inv_mapping, dict):
        return {int(str(k)): str(v).strip().lower() for k, v in inv_mapping.items()}

    raise ValueError(f"Could not parse stage2 labels mapping from: {labels_path}")


def _map_stage2_pred_ids_to_labels(pred_ids: np.ndarray, id_to_label: dict[int, str]) -> np.ndarray:
    mapped: list[str] = []
    for pred_id in pred_ids.tolist():
        pid = int(pred_id)
        if pid not in id_to_label:
            raise ValueError(f"Missing stage2 label mapping for class id: {pid}")
        mapped.append(str(id_to_label[pid]).strip().lower())
    return np.array(mapped, dtype=object)


def simulate_pipeline_predictions(
    X_val: pd.DataFrame,
    stage1_model: CatBoostClassifier,
    stage2_model: CatBoostClassifier,
    stage2_id2label: dict[int, str],
    threshold: float,
) -> np.ndarray:
    stage1_classes = [int(c) for c in stage1_model.classes_]
    if 1 not in stage1_classes:
        raise RuntimeError(f"Stage1 classes do not include pass class id=1. classes_={stage1_model.classes_}")
    pass_idx = stage1_classes.index(1)

    stage1_proba = stage1_model.predict_proba(X_val)
    pass_proba = stage1_proba[:, pass_idx]
    stage1_is_pass = pass_proba >= float(threshold)

    n = len(X_val)
    final_pred = np.empty(n, dtype=object)
    final_pred[stage1_is_pass] = "pass"

    notpass_idx = np.where(~stage1_is_pass)[0]
    if len(notpass_idx) == 0:
        return final_pred

    X_val_notpass = X_val.iloc[notpass_idx]
    stage2_proba = stage2_model.predict_proba(X_val_notpass)
    stage2_pred_ids = np.argmax(stage2_proba, axis=1).astype(int)
    mapped_labels = _map_stage2_pred_ids_to_labels(stage2_pred_ids, stage2_id2label)
    final_pred[notpass_idx] = mapped_labels
    return final_pred


def _align_proba_by_class_ids(proba: np.ndarray, classes: list[int], n_classes: int) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape={p.shape}")
    if len(classes) != p.shape[1]:
        raise ValueError(f"classes length {len(classes)} does not match proba columns {p.shape[1]}")
    aligned = np.zeros((p.shape[0], n_classes), dtype=np.float64)
    for col_idx, cls in enumerate(classes):
        c = int(cls)
        if c < 0 or c >= n_classes:
            raise ValueError(f"Unexpected class id {c}; expected 0..{n_classes-1}")
        aligned[:, c] = p[:, col_idx]
    return aligned


def simulate_joint_pipeline(
    X: pd.DataFrame,
    stage1_model: CatBoostClassifier,
    stage2_model: CatBoostClassifier,
    stage2_id2label: dict[int, str],
    routing_params: dict,
) -> np.ndarray:
    stage1_classes = [int(c) for c in stage1_model.classes_]
    stage1_proba_raw = stage1_model.predict_proba(X)
    stage1_proba4 = _align_proba_by_class_ids(stage1_proba_raw, stage1_classes, n_classes=4)
    pass_mask = route_pass_mask(
        proba4=stage1_proba4,
        policy=str(routing_params.get("route_policy", "margin_gate")),
        t_pass=float(routing_params.get("t_pass", 0.55)),
        t_notpass=float(routing_params.get("t_notpass", 0.40)),
        t_margin=float(routing_params.get("t_margin", 0.05)),
        route_default=str(routing_params.get("route_default", "pass")),
    )

    final_pred = np.empty(len(X), dtype=object)
    final_pred[pass_mask] = "pass"

    notpass_idx = np.where(~pass_mask)[0]
    if len(notpass_idx) == 0:
        return final_pred

    X_notpass = X.iloc[notpass_idx]
    stage2_proba_raw = stage2_model.predict_proba(X_notpass)
    stage2_classes = [int(c) for c in stage2_model.classes_]
    stage2_proba = _align_proba_by_class_ids(stage2_proba_raw, stage2_classes, n_classes=3)
    stage2_pred_ids = np.argmax(stage2_proba, axis=1).astype(int)
    final_pred[notpass_idx] = _map_stage2_pred_ids_to_labels(stage2_pred_ids, stage2_id2label)
    return final_pred


def eval_4class(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)

    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER_4)}
    try:
        y_true_ids = np.array([label_to_id[str(v)] for v in y_true_arr], dtype=int)
        y_pred_ids = np.array([label_to_id[str(v)] for v in y_pred_arr], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unexpected label for 4-class evaluation: {exc}") from exc

    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            labels=LABEL_ORDER_4,
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            labels=LABEL_ORDER_4,
            zero_division=0,
        )
    )
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    balanced_accuracy = float(balanced_accuracy_score(y_true_ids, y_pred_ids))
    pass_recall = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            labels=["pass"],
            average=None,
            zero_division=0,
        )[0]
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=LABEL_ORDER_4)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=LABEL_ORDER_4,
        target_names=LABEL_ORDER_4,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    per_class_recall = {label: float(report.get(label, {}).get("recall", 0.0)) for label in LABEL_ORDER_4}
    per_class_precision = {label: float(report.get(label, {}).get("precision", 0.0)) for label in LABEL_ORDER_4}

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "balanced_acc": balanced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "pass_recall": pass_recall,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _print_kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    width = 78
    print("\n" + "=" * width)
    print(title)
    print("-" * width)
    for k, v in rows:
        print(f"{k:<34} {v:>43}")
    print("=" * width)


def _print_threshold_sweep_summary(rows: list[dict], best_threshold: float, top_k: int = 10) -> None:
    if not rows:
        return

    ranked = sorted(
        rows,
        key=lambda r: (float(r["macro_f1"]), float(r["balanced_accuracy"]), float(r["accuracy"])),
        reverse=True,
    )
    top = ranked[:top_k]

    print("\nThreshold Sweep Summary (top 10)")
    print("-" * 72)
    print(f"{' ':1} {'threshold':>9} {'macro_f1':>10} {'pass_recall':>12} {'bal_acc':>10} {'acc':>10}")
    print("-" * 72)
    for r in top:
        marker = "*" if abs(float(r["threshold"]) - float(best_threshold)) <= 1e-12 else " "
        print(
            f"{marker} "
            f"{float(r['threshold']):>9.2f} "
            f"{float(r['macro_f1']):>10.4f} "
            f"{float(r['pass_recall']):>12.4f} "
            f"{float(r['balanced_accuracy']):>10.4f} "
            f"{float(r['accuracy']):>10.4f}"
        )
    print("-" * 72)


def _print_joint_top10(rows: list[dict], top_k: int = 10) -> None:
    if not rows:
        return
    top = rows[:top_k]
    print("\nJoint Routing Sweep Top 10")
    print("-" * 132)
    print(
        f"{'rank':>4} {'score':>8} {'macro_f1':>8} {'bal_acc':>8} {'rec_dist':>8} {'rec_fail':>8} "
        f"{'rec_pass':>8} {'rec_wd':>8} {'t_pass':>8} {'t_notpass':>10} {'t_margin':>9}"
    )
    print("-" * 132)
    for i, r in enumerate(top, start=1):
        print(
            f"{i:>4d} {float(r['score']):>8.4f} {float(r['macro_f1']):>8.4f} {float(r['balanced_accuracy']):>8.4f} "
            f"{float(r['rec_dist']):>8.4f} {float(r['rec_fail']):>8.4f} {float(r['rec_pass']):>8.4f} {float(r['rec_wd']):>8.4f} "
            f"{float(r['t_pass']):>8.2f} {float(r['t_notpass']):>10.2f} {float(r['t_margin']):>9.2f}"
        )
    print("-" * 132)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be in (0, 1).")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val_size must be in (0, 1).")
    if not (0.0 <= args.threshold_min <= 1.0 and 0.0 <= args.threshold_max <= 1.0):
        raise ValueError("--threshold_min and --threshold_max must be in [0, 1].")
    if not (args.threshold_min < args.threshold_max):
        raise ValueError("--threshold_min must be smaller than --threshold_max.")
    if not (args.threshold_step > 0):
        raise ValueError("--threshold_step must be > 0.")
    if not (0.0 <= args.pass_recall_min <= 1.0):
        raise ValueError("--pass_recall_min must be in [0, 1].")

    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {df.columns.tolist()}")

    y_4class = _normalize_target(df[args.target], args.target)
    y_bin = _to_binary_target(y_4class)
    y4_id = _to_stage1_4class_ids(y_4class)
    X = df.drop(columns=[args.target]).copy()

    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("NA")

    if args.stage1_mode == "ova4":
        X_train_val, X_test, y_train_val, y_test, y4_train_val, y4_test = train_test_split(
            X,
            y4_id,
            y_4class,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y4_id,
        )

        X_train, X_val, y_train, y_val, y4_train, y4_val = train_test_split(
            X_train_val,
            y_train_val,
            y4_train_val,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=y_train_val,
        )

        feature_names = X_train.columns.tolist()
        cat_feature_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

        if args.class_weight_mode == "balanced":
            class_weights = _compute_balanced_weights(y_train, n_classes=4)
        elif args.class_weight_mode == "manual":
            class_weights = _parse_manual_weights(args.class_weights, expected_len=4)
        else:
            class_weights = None

        requested_eval_metric = "TotalF1"
        model = CatBoostClassifier(
            loss_function="MultiClassOneVsAll",
            eval_metric=requested_eval_metric,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            random_strength=args.random_strength,
            bagging_temperature=args.bagging_temperature,
            random_seed=args.seed,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=args.verbose,
            allow_writing_files=False,
            class_weights=class_weights,
        )
        try:
            model.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            eval_metric_used = requested_eval_metric
        except Exception as exc:
            print(f"Falling back eval_metric from TotalF1 to MultiClass due to: {exc}")
            model = CatBoostClassifier(
                loss_function="MultiClassOneVsAll",
                eval_metric="MultiClass",
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                random_strength=args.random_strength,
                bagging_temperature=args.bagging_temperature,
                random_seed=args.seed,
                early_stopping_rounds=args.early_stopping_rounds,
                verbose=args.verbose,
                allow_writing_files=False,
                class_weights=class_weights,
            )
            model.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            eval_metric_used = "MultiClass"

        val_pred_ids = np.argmax(model.predict_proba(X_val), axis=1)
        test_pred_ids = np.argmax(model.predict_proba(X_test), axis=1)
        id_to_label = {v: k for k, v in STAGE1_MAPPING_4.items()}
        val_pred_labels = np.array([id_to_label[int(i)] for i in val_pred_ids], dtype=object)
        test_pred_labels = np.array([id_to_label[int(i)] for i in test_pred_ids], dtype=object)

        val_metrics_4 = eval_4class(y4_val.to_numpy(), val_pred_labels)
        test_metrics_4 = eval_4class(y4_test.to_numpy(), test_pred_labels)
        joint_val_metrics_4: dict | None = None
        joint_sweep_results: list[dict] = []
        val_routing_breakdown: dict | None = None
        val_sucked_into_pass: dict | None = None
        stage2_loaded_for_joint = False
        stage2_model: CatBoostClassifier | None = None
        stage2_id2label: dict[int, str] | None = None
        stage2_model_path = Path(args.stage2_model_path).resolve()
        stage2_labels_path = Path(args.stage2_labels_path).resolve()
        joint_sweep_path = DEFAULT_STAGE1_JOINT_SWEEP_PATH
        best_routing: dict | None = None
        try:
            if not stage2_model_path.exists():
                raise FileNotFoundError(f"Stage2 model not found: {stage2_model_path}")
            if not stage2_labels_path.exists():
                raise FileNotFoundError(f"Stage2 labels not found: {stage2_labels_path}")
            stage2_model = CatBoostClassifier()
            stage2_model.load_model(stage2_model_path)
            stage2_id2label = _load_stage2_id_to_label(stage2_labels_path)

            if int(args.optimize_joint) == 1:
                grid_t_pass = _parse_float_list_arg(args.grid_t_pass, "--grid_t_pass")
                grid_t_notpass = _parse_float_list_arg(args.grid_t_notpass, "--grid_t_notpass")
                grid_t_margin = _parse_float_list_arg(args.grid_t_margin, "--grid_t_margin")

                for t_pass in grid_t_pass:
                    for t_notpass in grid_t_notpass:
                        for t_margin in grid_t_margin:
                            routing_params = {
                                "route_policy": args.route_policy,
                                "t_pass": float(t_pass),
                                "t_notpass": float(t_notpass),
                                "t_margin": float(t_margin),
                                "route_default": args.route_default,
                            }
                            y_val_joint = simulate_joint_pipeline(
                                X=X_val,
                                stage1_model=model,
                                stage2_model=stage2_model,
                                stage2_id2label=stage2_id2label,
                                routing_params=routing_params,
                            )
                            m = eval_4class(y4_val.to_numpy(), y_val_joint)
                            rec = m["per_class_recall"]
                            rec_pass = float(rec.get("pass", 0.0))
                            rec_dist = float(rec.get("distinction", 0.0))
                            rec_fail = float(rec.get("fail", 0.0))
                            rec_wd = float(rec.get("withdrawn", 0.0))

                            score = float(m["macro_f1"])
                            if rec_dist < float(args.dist_recall_min):
                                score -= 0.03
                            eligible = rec_pass >= float(args.pass_recall_min)

                            joint_sweep_results.append(
                                {
                                    "route_policy": args.route_policy,
                                    "route_default": args.route_default,
                                    "t_pass": float(t_pass),
                                    "t_notpass": float(t_notpass),
                                    "t_margin": float(t_margin),
                                    "score": score,
                                    "macro_f1": float(m["macro_f1"]),
                                    "balanced_accuracy": float(m["balanced_accuracy"]),
                                    "weighted_f1": float(m["weighted_f1"]),
                                    "accuracy": float(m["accuracy"]),
                                    "rec_dist": rec_dist,
                                    "rec_fail": rec_fail,
                                    "rec_pass": rec_pass,
                                    "rec_wd": rec_wd,
                                    "eligible": bool(eligible),
                                    "dist_guardrail_penalized": bool(rec_dist < float(args.dist_recall_min)),
                                    "metrics": m,
                                }
                            )

                constrained = [r for r in joint_sweep_results if bool(r["eligible"])]
                candidates = constrained if constrained else joint_sweep_results
                if not constrained:
                    print("[WARN] No joint config met pass_recall_min; using best macro_f1 with tie-breakers.")

                ranked = sorted(
                    candidates,
                    key=lambda r: (
                        -float(r["score"]),
                        -float(r["rec_dist"]),
                        -float(r["rec_fail"]),
                        -float(r["balanced_accuracy"]),
                    ),
                )
                best_routing = ranked[0]
                joint_val_metrics_4 = best_routing["metrics"]
                _print_joint_top10(ranked, top_k=10)
                save_json(
                    {
                        "route_policy": args.route_policy,
                        "route_default": args.route_default,
                        "constraints": {
                            "pass_recall_min": float(args.pass_recall_min),
                            "dist_recall_min": float(args.dist_recall_min),
                            "dist_penalty": 0.03,
                        },
                        "total_configs": int(len(joint_sweep_results)),
                        "eligible_configs": int(len(constrained)),
                        "results": joint_sweep_results,
                        "ranked_top10": ranked[:10],
                        "best": best_routing,
                    },
                    joint_sweep_path,
                )
            else:
                routing_params = {
                    "route_policy": args.route_policy,
                    "t_pass": float(args.t_pass),
                    "t_notpass": float(args.t_notpass),
                    "t_margin": float(args.t_margin),
                    "route_default": args.route_default,
                }
                y_val_joint = simulate_joint_pipeline(
                    X=X_val,
                    stage1_model=model,
                    stage2_model=stage2_model,
                    stage2_id2label=stage2_id2label,
                    routing_params=routing_params,
                )
                joint_val_metrics_4 = eval_4class(y4_val.to_numpy(), y_val_joint)
                best_routing = {
                    "route_policy": args.route_policy,
                    "route_default": args.route_default,
                    "t_pass": float(args.t_pass),
                    "t_notpass": float(args.t_notpass),
                    "t_margin": float(args.t_margin),
                    "macro_f1": float(joint_val_metrics_4["macro_f1"]),
                    "balanced_accuracy": float(joint_val_metrics_4["balanced_accuracy"]),
                    "rec_dist": float(joint_val_metrics_4["per_class_recall"]["distinction"]),
                    "rec_fail": float(joint_val_metrics_4["per_class_recall"]["fail"]),
                    "rec_pass": float(joint_val_metrics_4["per_class_recall"]["pass"]),
                    "rec_wd": float(joint_val_metrics_4["per_class_recall"]["withdrawn"]),
                }
            stage2_loaded_for_joint = True
        except Exception as exc:
            print(f"[WARN] Could not compute joint VAL pipeline metrics in ova4 mode: {exc}")

        # Routing diagnostics on VAL with selected best routing config.
        if best_routing is not None:
            val_proba4 = _align_proba_by_class_ids(
                model.predict_proba(X_val),
                [int(c) for c in model.classes_],
                n_classes=4,
            )
            val_pass_mask = route_pass_mask(
                proba4=val_proba4,
                policy=str(best_routing["route_policy"]),
                t_pass=float(best_routing["t_pass"]),
                t_notpass=float(best_routing["t_notpass"]),
                t_margin=float(best_routing["t_margin"]),
                route_default=str(best_routing["route_default"]),
            )
            total_n = int(len(X_val))
            pass_n = int(np.sum(val_pass_mask))
            notpass_n = int(total_n - pass_n)
            pass_rate = float((pass_n / total_n) if total_n > 0 else 0.0)
            notpass_rate = float((notpass_n / total_n) if total_n > 0 else 0.0)

            by_true_class: dict[str, dict[str, float | int]] = {}
            y_val_true = y4_val.to_numpy(dtype=object)
            for cls in LABEL_ORDER_4:
                cls_mask = y_val_true == cls
                cls_total = int(np.sum(cls_mask))
                cls_routed_pass = int(np.sum(cls_mask & val_pass_mask))
                cls_rate = float((cls_routed_pass / cls_total) if cls_total > 0 else 0.0)
                by_true_class[cls] = {
                    "count_true": cls_total,
                    "count_routed_pass": cls_routed_pass,
                    "routed_pass_rate": cls_rate,
                }

            val_routing_breakdown = {
                "total": total_n,
                "routed_pass_count": pass_n,
                "routed_notpass_count": notpass_n,
                "routed_pass_rate": pass_rate,
                "routed_notpass_rate": notpass_rate,
                "by_true_class": by_true_class,
            }
            val_sucked_into_pass = {
                cls: by_true_class[cls] for cls in ["distinction", "fail", "withdrawn"]
            }

            print("\nVAL Routing Breakdown (best joint config)")
            print(
                f"Routed PASS: {pass_n}/{total_n} ({pass_rate*100:.2f}%) | "
                f"Routed NOTPASS: {notpass_n}/{total_n} ({notpass_rate*100:.2f}%)"
            )
            print("-" * 88)
            print(f"{'true_class':<14} {'count_true':>12} {'count_routed_pass':>20} {'routed_pass_rate':>18}")
            print("-" * 88)
            for cls in LABEL_ORDER_4:
                row = by_true_class[cls]
                print(
                    f"{cls:<14} {int(row['count_true']):>12} {int(row['count_routed_pass']):>20} "
                    f"{float(row['routed_pass_rate']):>17.4f}"
                )
            print("-" * 88)

            print("\nSucked into PASS (VAL)")
            print("-" * 88)
            print(f"{'true_class':<14} {'count_true':>12} {'predicted_pass':>20} {'rate':>18}")
            print("-" * 88)
            for cls in ["distinction", "fail", "withdrawn"]:
                row = by_true_class[cls]
                print(
                    f"{cls:<14} {int(row['count_true']):>12} {int(row['count_routed_pass']):>20} "
                    f"{float(row['routed_pass_rate']):>17.4f}"
                )
            print("-" * 88)

        # Retrain Stage1 on TRAIN+VAL after routing params are selected on VAL.
        X_trainval = pd.concat([X_train, X_val], axis=0)
        y_trainval = pd.concat([y_train, y_val], axis=0)
        retrained_model = CatBoostClassifier(
            loss_function="MultiClassOneVsAll",
            eval_metric=eval_metric_used,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            random_strength=args.random_strength,
            bagging_temperature=args.bagging_temperature,
            random_seed=args.seed,
            early_stopping_rounds=None,
            verbose=args.verbose,
            allow_writing_files=False,
            class_weights=class_weights,
        )
        retrained_model.fit(
            X_trainval,
            y_trainval,
            cat_features=cat_feature_indices,
        )
        model = retrained_model

        # Stage1-alone test metrics after retraining (for completeness).
        retrained_test_pred_ids = np.argmax(model.predict_proba(X_test), axis=1)
        retrained_test_pred_labels = np.array([id_to_label[int(i)] for i in retrained_test_pred_ids], dtype=object)
        test_metrics_4 = eval_4class(y4_test.to_numpy(), retrained_test_pred_labels)

        # Final joint pipeline test metrics (TEST only, fixed routing params).
        test_pipeline_metrics: dict | None = None
        if stage2_model is not None and stage2_id2label is not None and best_routing is not None:
            test_routing = {
                "route_policy": best_routing["route_policy"],
                "t_pass": float(best_routing["t_pass"]),
                "t_notpass": float(best_routing["t_notpass"]),
                "t_margin": float(best_routing["t_margin"]),
                "route_default": best_routing["route_default"],
            }
            y_test_joint = simulate_joint_pipeline(
                X=X_test,
                stage1_model=model,
                stage2_model=stage2_model,
                stage2_id2label=stage2_id2label,
                routing_params=test_routing,
            )
            test_pipeline_metrics = eval_4class(y4_test.to_numpy(), y_test_joint)

        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "stage1_model.cbm"
        metrics_path = output_dir / "stage1_metrics.json"
        labels_path = output_dir / "stage1_labels.json"

        model.save_model(model_path)
        save_json(
            {
                "label_order_4": LABEL_ORDER_4,
                "mapping": STAGE1_MAPPING_4,
                "inverse_mapping": {str(v): k for k, v in STAGE1_MAPPING_4.items()},
            },
            labels_path,
        )

        metrics = {
            "mode": "ova4",
            "dataset": {
                "input_path": str(input_path),
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "target_column": args.target,
            },
            "splits": {
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "test_fraction_full": float(args.test_size),
                "val_fraction_remaining": float(args.val_size),
                "seed": int(args.seed),
            },
            "labels": {
                "label_order_4": LABEL_ORDER_4,
                "mapping": STAGE1_MAPPING_4,
            },
            "model": {
                "stage1_mode": args.stage1_mode,
                "stage1_loss_requested": args.stage1_loss,
                "loss_function": "MultiClassOneVsAll",
                "eval_metric_requested": requested_eval_metric,
                "eval_metric_used": eval_metric_used,
                "iterations": int(args.iterations),
                "learning_rate": float(args.learning_rate),
                "depth": int(args.depth),
                "l2_leaf_reg": float(args.l2_leaf_reg),
                "random_strength": float(args.random_strength),
                "bagging_temperature": float(args.bagging_temperature),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "class_weight_mode": args.class_weight_mode,
                "class_weights": class_weights,
            },
            "routing": {
                "route_policy": (best_routing["route_policy"] if best_routing is not None else args.route_policy),
                "t_pass": float(best_routing["t_pass"] if best_routing is not None else args.t_pass),
                "t_notpass": float(best_routing["t_notpass"] if best_routing is not None else args.t_notpass),
                "t_margin": float(best_routing["t_margin"] if best_routing is not None else args.t_margin),
                "route_default": (best_routing["route_default"] if best_routing is not None else args.route_default),
                "optimize_joint": int(args.optimize_joint),
                "grid_t_pass": args.grid_t_pass,
                "grid_t_notpass": args.grid_t_notpass,
                "grid_t_margin": args.grid_t_margin,
                "pass_recall_min": float(args.pass_recall_min),
                "dist_recall_min": float(args.dist_recall_min),
                "joint_sweep_path": str(joint_sweep_path),
            },
            "val_metrics_4class": val_metrics_4,
            "test_metrics_4class": test_metrics_4,
            "final_pipeline_val_metrics_4class": joint_val_metrics_4,
            "test_pipeline_metrics": test_pipeline_metrics,
            "val_routing_breakdown": val_routing_breakdown,
            "val_sucked_into_pass": val_sucked_into_pass,
            "stage2_loaded_for_joint_eval": stage2_loaded_for_joint,
            "label_info": {
                "train_original_distribution": y4_train.value_counts().sort_index().to_dict(),
                "val_original_distribution": y4_val.value_counts().sort_index().to_dict(),
                "test_original_distribution": y4_test.value_counts().sort_index().to_dict(),
            },
        }
        save_json(metrics, metrics_path)
        if best_routing is not None:
            threshold_path = output_dir / "stage1_threshold.json"
            save_json(
                {
                    "route_policy": best_routing["route_policy"],
                    "t_pass": float(best_routing["t_pass"]),
                    "t_notpass": float(best_routing["t_notpass"]),
                    "t_margin": float(best_routing["t_margin"]),
                    "route_default": best_routing["route_default"],
                    "val_macro_f1": float(best_routing.get("macro_f1", 0.0)),
                    "val_balanced_accuracy": float(best_routing.get("balanced_accuracy", 0.0)),
                    "val_recalls": {
                        "distinction": float(best_routing.get("rec_dist", 0.0)),
                        "fail": float(best_routing.get("rec_fail", 0.0)),
                        "pass": float(best_routing.get("rec_pass", 0.0)),
                        "withdrawn": float(best_routing.get("rec_wd", 0.0)),
                    },
                    "constraints_used": {
                        "pass_recall_min": float(args.pass_recall_min),
                        "dist_recall_min": float(args.dist_recall_min),
                        "dist_penalty": 0.03,
                    },
                    "optimize_joint": int(args.optimize_joint),
                },
                threshold_path,
            )

        _print_kv_table(
            "Stage 1 OVA4 Training",
            [
                ("Mode", args.stage1_mode),
                ("Loss function", "MultiClassOneVsAll"),
                ("Eval metric used", eval_metric_used),
                ("Label mapping", str(STAGE1_MAPPING_4)),
            ],
        )
        _print_kv_table(
            "Stage 1 OVA4 Splits",
            [
                ("Train size", str(len(X_train))),
                ("Validation size", str(len(X_val))),
                ("Test size", str(len(X_test))),
                ("Seed", str(args.seed)),
            ],
        )
        _print_kv_table(
            "Stage 1 OVA4 Validation Metrics",
            [
                ("Macro F1", f"{float(val_metrics_4['macro_f1']):.4f}"),
                ("Balanced accuracy", f"{float(val_metrics_4['balanced_accuracy']):.4f}"),
                ("Accuracy", f"{float(val_metrics_4['accuracy']):.4f}"),
                ("Weighted F1", f"{float(val_metrics_4['weighted_f1']):.4f}"),
            ],
        )
        if joint_val_metrics_4 is not None:
            _print_kv_table(
                "Joint Pipeline Validation Metrics",
                [
                    ("Macro F1", f"{float(joint_val_metrics_4['macro_f1']):.4f}"),
                    ("Balanced accuracy", f"{float(joint_val_metrics_4['balanced_accuracy']):.4f}"),
                    ("Accuracy", f"{float(joint_val_metrics_4['accuracy']):.4f}"),
                    ("Weighted F1", f"{float(joint_val_metrics_4['weighted_f1']):.4f}"),
                ],
            )
        if test_pipeline_metrics is not None:
            _print_kv_table(
                "Joint Pipeline Test Metrics",
                [
                    ("Macro F1", f"{float(test_pipeline_metrics['macro_f1']):.4f}"),
                    ("Balanced accuracy", f"{float(test_pipeline_metrics['balanced_accuracy']):.4f}"),
                    ("Accuracy", f"{float(test_pipeline_metrics['accuracy']):.4f}"),
                    ("Weighted F1", f"{float(test_pipeline_metrics['weighted_f1']):.4f}"),
                ],
            )
        print(f"Saved model: {model_path}")
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved labels: {labels_path}")
        if int(args.optimize_joint) == 1:
            print(f"Saved joint sweep: {joint_sweep_path}")
            print(f"Saved routing params: {output_dir / 'stage1_threshold.json'}")
        return

    X_train_val, X_test, y_train_val, y_test, y4_train_val, y4_test = train_test_split(
        X,
        y_bin,
        y_4class,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_bin,
    )

    X_train, X_val, y_train, y_val, y4_train, y4_val = train_test_split(
        X_train_val,
        y_train_val,
        y4_train_val,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_val,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="F1",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        random_seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    effective_optimize_mode = args.optimize_mode
    stage2_model = None
    stage2_id_to_label: dict[int, str] | None = None
    stage2_model_path = Path(args.stage2_model_path).resolve()
    stage2_labels_path = Path(args.stage2_labels_path).resolve()

    if effective_optimize_mode == "pipeline_macro_f1":
        try:
            if not stage2_model_path.exists():
                raise FileNotFoundError(f"Stage2 model not found: {stage2_model_path}")
            if not stage2_labels_path.exists():
                raise FileNotFoundError(f"Stage2 labels not found: {stage2_labels_path}")

            stage2_model = CatBoostClassifier()
            stage2_model.load_model(stage2_model_path)
            stage2_id_to_label = _load_stage2_id_to_label(stage2_labels_path)

            # Validation-only readiness check for next step (pipeline threshold sweep).
            _ = stage2_model.predict(X_val.iloc[:1]) if len(X_val) > 0 else None
            if len(X_val) > 0:
                sample_pred = stage2_model.predict(X_val.iloc[:1])
                sample_arr = np.asarray(sample_pred).reshape(-1)
                _ = _map_stage2_pred_ids_to_labels(sample_arr.astype(int), stage2_id_to_label)
        except Exception as exc:
            print(
                "[WARN] optimize_mode=pipeline_macro_f1 requested but Stage2 artifacts "
                f"could not be loaded ({exc}). Falling back to optimize_mode=binary_f1."
            )
            effective_optimize_mode = "binary_f1"
            stage2_model = None
            stage2_id_to_label = None

    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError(f"Model classes do not include positive class 1. classes_={classes}")
    pass_idx = classes.index(1)

    val_proba = model.predict_proba(X_val)[:, pass_idx]
    thresholds = np.arange(float(args.threshold_min), float(args.threshold_max) + 1e-9, float(args.threshold_step))
    if thresholds.size == 0:
        raise ValueError("Threshold grid is empty. Check threshold_min/threshold_max/threshold_step.")

    sweep_rows: list[dict] = []
    val_macro_f1_at_best: float | None = None
    val_pass_recall_at_best: float | None = None
    val_accuracy_at_best: float | None = None
    val_balanced_accuracy_at_best: float | None = None
    val_pipeline_diagnostics: dict | None = None

    if effective_optimize_mode == "pipeline_macro_f1" and stage2_model is not None and stage2_id_to_label is not None:
        y_val_true_4 = y4_val.to_numpy()
        for threshold in thresholds:
            y_pred_4 = simulate_pipeline_predictions(
                X_val=X_val,
                stage1_model=model,
                stage2_model=stage2_model,
                stage2_id2label=stage2_id_to_label,
                threshold=float(threshold),
            )
            m4 = eval_4class(y_val_true_4, y_pred_4)
            sweep_rows.append(
                {
                    "threshold": float(threshold),
                    "macro_f1": float(m4["macro_f1"]),
                    "pass_recall": float(m4["pass_recall"]),
                    "balanced_accuracy": float(m4["balanced_accuracy"]),
                    "accuracy": float(m4["accuracy"]),
                    "confusion_matrix": m4["confusion_matrix"],
                    "classification_report": m4["classification_report"],
                    "eligible": float(m4["pass_recall"]) >= float(args.pass_recall_min),
                }
            )

        constrained = [r for r in sweep_rows if bool(r["eligible"])]
        candidate_rows = constrained
        if not constrained:
            print("No threshold met pass_recall_min; using best macro_f1 without constraint.")
            candidate_rows = sweep_rows

        best_row = max(
            candidate_rows,
            key=lambda r: (float(r["macro_f1"]), float(r["balanced_accuracy"]), float(r["accuracy"])),
        )
        best_threshold = float(best_row["threshold"])
        best_val_score = float(best_row["macro_f1"])
        val_macro_f1_at_best = float(best_row["macro_f1"])
        val_pass_recall_at_best = float(best_row["pass_recall"])
        val_accuracy_at_best = float(best_row["accuracy"])
        val_balanced_accuracy_at_best = float(best_row["balanced_accuracy"])

        _print_threshold_sweep_summary(sweep_rows, best_threshold, top_k=10)
    else:
        best_threshold, best_val_score = _find_best_threshold(
            y_val.to_numpy(),
            val_proba,
            args.threshold_metric,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
        )
        val_pred_bin = (val_proba >= best_threshold).astype(int)
        val_pass_recall_at_best = float(recall_score(y_val, val_pred_bin, zero_division=0))
        val_accuracy_at_best = float(accuracy_score(y_val, val_pred_bin))
        val_balanced_accuracy_at_best = float(balanced_accuracy_score(y_val, val_pred_bin))

    val_pass_recall = float(val_pass_recall_at_best) if val_pass_recall_at_best is not None else 0.0

    if effective_optimize_mode == "pipeline_macro_f1" and stage2_model is not None and stage2_id_to_label is not None:
        y_val_pred_4 = simulate_pipeline_predictions(
            X_val=X_val,
            stage1_model=model,
            stage2_model=stage2_model,
            stage2_id2label=stage2_id_to_label,
            threshold=float(best_threshold),
        )
        val_metrics_4 = eval_4class(y4_val.to_numpy(), y_val_pred_4)

        cm4 = np.array(val_metrics_4["confusion_matrix"], dtype=int)
        rep4 = val_metrics_4["classification_report"]
        distinction_recall = float(rep4.get("distinction", {}).get("recall", 0.0))
        fail_recall = float(rep4.get("fail", {}).get("recall", 0.0))
        pass_recall_diag = float(rep4.get("pass", {}).get("recall", 0.0))
        withdrawn_recall = float(rep4.get("withdrawn", {}).get("recall", 0.0))

        y_val_true_arr = y4_val.to_numpy(dtype=object)
        y_val_pred_arr = np.asarray(y_val_pred_4, dtype=object)
        sucked_into_pass: list[dict] = []
        for cls in ["distinction", "fail", "withdrawn"]:
            cls_mask = y_val_true_arr == cls
            cls_total = int(np.sum(cls_mask))
            cls_to_pass = int(np.sum(cls_mask & (y_val_pred_arr == "pass")))
            cls_pct = float((100.0 * cls_to_pass / cls_total) if cls_total > 0 else 0.0)
            sucked_into_pass.append(
                {
                    "true_class": cls,
                    "count_true_class": cls_total,
                    "count_predicted_as_pass": cls_to_pass,
                    "percentage": cls_pct,
                }
            )

        print("\nVAL 4-class confusion matrix (order: distinction, fail, pass, withdrawn)")
        print(cm4)
        print(
            "Per-class recall: "
            f"distinction_recall={distinction_recall:.4f} | "
            f"fail_recall={fail_recall:.4f} | "
            f"pass_recall={pass_recall_diag:.4f} | "
            f"withdrawn_recall={withdrawn_recall:.4f}"
        )
        print("\nSucked into Pass")
        print("-" * 72)
        print(f"{'true_class':<14} {'count_true_class':>18} {'count_pred_as_pass':>20} {'percentage':>12}")
        print("-" * 72)
        for row in sucked_into_pass:
            print(
                f"{row['true_class']:<14} "
                f"{int(row['count_true_class']):>18} "
                f"{int(row['count_predicted_as_pass']):>20} "
                f"{float(row['percentage']):>11.2f}%"
            )
        print("-" * 72)

        val_pipeline_diagnostics = {
            "label_order": LABEL_ORDER_4,
            "best_threshold": float(best_threshold),
            "macro_f1": float(val_metrics_4["macro_f1"]),
            "weighted_f1": float(val_metrics_4["weighted_f1"]),
            "accuracy": float(val_metrics_4["accuracy"]),
            "balanced_accuracy": float(val_metrics_4["balanced_accuracy"]),
            "pass_recall": float(val_metrics_4["pass_recall"]),
            "confusion_matrix": val_metrics_4["confusion_matrix"],
            "classification_report": rep4,
            "per_class_recall": {
                "distinction_recall": distinction_recall,
                "fail_recall": fail_recall,
                "pass_recall": pass_recall_diag,
                "withdrawn_recall": withdrawn_recall,
            },
            "sucked_into_pass": sucked_into_pass,
        }

    test_proba = model.predict_proba(X_test)[:, pass_idx]
    y_pred = (test_proba >= best_threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1_bin = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["notpass", "pass"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stage1_model.cbm"
    metrics_path = output_dir / "stage1_metrics.json"
    threshold_path = output_dir / "stage1_threshold.json"

    model.save_model(model_path)

    metrics = {
        "dataset": {
            "input_path": str(input_path),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "target_column": args.target,
        },
        "splits": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "test_fraction_full": float(args.test_size),
            "val_fraction_remaining": float(args.val_size),
            "seed": int(args.seed),
        },
        "model": {
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "iterations": int(args.iterations),
            "learning_rate": float(args.learning_rate),
            "depth": int(args.depth),
            "l2_leaf_reg": float(args.l2_leaf_reg),
            "random_strength": float(args.random_strength),
            "bagging_temperature": float(args.bagging_temperature),
            "early_stopping_rounds": int(args.early_stopping_rounds),
        },
        "thresholding": {
            "metric": args.threshold_metric,
            "optimize_mode_requested": args.optimize_mode,
            "optimize_mode_effective": effective_optimize_mode,
            "best_threshold": float(best_threshold),
            "best_validation_score": float(best_val_score),
            "pass_recall_min": float(args.pass_recall_min),
            "val_macro_f1_at_best": val_macro_f1_at_best,
            "val_pass_recall_at_best": val_pass_recall_at_best,
            "val_accuracy_at_best": val_accuracy_at_best,
            "val_balanced_accuracy_at_best": val_balanced_accuracy_at_best,
            "validation_pass_recall_at_best_threshold": val_pass_recall,
            "search_start": float(args.threshold_min),
            "search_end": float(args.threshold_max),
            "search_step": float(args.threshold_step),
            "stage2_model_path": str(stage2_model_path),
            "stage2_labels_path": str(stage2_labels_path),
            "stage2_loaded": bool(stage2_model is not None and stage2_id_to_label is not None),
        },
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1": f1_bin,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        },
        "val_pipeline_diagnostics": val_pipeline_diagnostics,
        "label_info": {
            "original_4class_labels": sorted(VALID_LABELS),
            "negative_group_labels": sorted(NEGATIVE_LABELS),
            "binary_mapping": {"notpass": 0, "pass": 1},
            "train_original_distribution": y4_train.value_counts().sort_index().to_dict(),
            "val_original_distribution": y4_val.value_counts().sort_index().to_dict(),
            "test_original_distribution": y4_test.value_counts().sort_index().to_dict(),
        },
    }

    threshold_artifact = {
        "best_threshold": float(best_threshold),
        "optimize_mode": effective_optimize_mode,
        "pass_recall_min": float(args.pass_recall_min),
        "val_macro_f1_at_best": val_macro_f1_at_best,
        "val_pass_recall_at_best": val_pass_recall_at_best,
        "val_accuracy_at_best": val_accuracy_at_best,
        "val_balanced_accuracy_at_best": val_balanced_accuracy_at_best,
        "metric": args.threshold_metric,
        "best_validation_score": float(best_val_score),
    }

    save_json(metrics, metrics_path)
    save_json(threshold_artifact, threshold_path)

    _print_kv_table(
        "Stage 1 Dataset",
        [
            ("Input CSV", str(input_path)),
            ("Shape (rows, cols)", f"{df.shape[0]}, {df.shape[1]}"),
            ("Target", args.target),
            ("Binary mapping", "pass=1 | notpass=0"),
        ],
    )

    _print_kv_table(
        "Stage 1 Splits",
        [
            ("Train size", str(len(X_train))),
            ("Validation size", str(len(X_val))),
            ("Test size", str(len(X_test))),
            ("Seed", str(args.seed)),
        ],
    )

    _print_kv_table(
        "Stage 1 Metrics (Test)",
        [
            ("Threshold metric (val)", args.threshold_metric),
            ("Optimize mode", f"{effective_optimize_mode} (requested: {args.optimize_mode})"),
            ("Best threshold", f"{best_threshold:.2f}"),
            ("Validation best score", f"{best_val_score:.4f}"),
            ("VAL Pass recall @ best thr", f"{val_pass_recall:.4f}"),
            ("Accuracy", f"{acc:.4f}"),
            ("Balanced accuracy", f"{bal_acc:.4f}"),
            ("F1 (binary)", f"{f1_bin:.4f}"),
        ],
    )

    _print_kv_table(
        "Confusion Matrix (rows=true, cols=pred)",
        [
            ("[notpass, notpass]", str(int(cm[0, 0]))),
            ("[notpass, pass]", str(int(cm[0, 1]))),
            ("[pass, notpass]", str(int(cm[1, 0]))),
            ("[pass, pass]", str(int(cm[1, 1]))),
        ],
    )

    print("Classification report (binary):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["notpass", "pass"],
            digits=4,
            zero_division=0,
        )
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved threshold: {threshold_path}")


if __name__ == "__main__":
    main()
