from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (PROJECT_ROOT / "data" / "kuzilek_clean_plus.csv").resolve()
HIER_DIR = (PROJECT_ROOT / "outputs" / "hierarchical").resolve()

DEFAULT_STAGE1_MODEL = (HIER_DIR / "stage1_model.cbm").resolve()
DEFAULT_STAGE1_THRESHOLD = (HIER_DIR / "stage1_threshold.json").resolve()
DEFAULT_STAGE2_MODEL = (HIER_DIR / "stage2_model.cbm").resolve()
DEFAULT_STAGE2_LABELS = (HIER_DIR / "stage2_labels.json").resolve()
DEFAULT_OUTPUT = (HIER_DIR / "predictions_4class.csv").resolve()
DEFAULT_METRICS = (HIER_DIR / "hierarchical_metrics_4class.json").resolve()

VALID_4CLASS = {"distinction", "fail", "pass", "withdrawn"}
LABEL_ORDER_4 = ["distinction", "fail", "pass", "withdrawn"]
ORDER_4CLASS = ["distinction", "fail", "pass", "withdrawn"]


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical prediction: Stage1 (pass/notpass) + Stage2 (3-class NotPass).")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to input CSV.")
    p.add_argument("--target", type=str, default="final_result", help="Target column name if present.")
    p.add_argument("--stage1_model", type=str, default=str(DEFAULT_STAGE1_MODEL), help="Path to stage1 model (.cbm).")
    p.add_argument("--stage1_threshold", type=str, default=str(DEFAULT_STAGE1_THRESHOLD), help="Path to stage1 threshold JSON.")
    p.add_argument("--stage2_model", type=str, default=str(DEFAULT_STAGE2_MODEL), help="Path to stage2 model (.cbm).")
    p.add_argument("--stage2_labels", type=str, default=str(DEFAULT_STAGE2_LABELS), help="Path to stage2 labels JSON.")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV.")
    p.add_argument("--metrics_output", type=str, default=str(DEFAULT_METRICS), help="Path to save 4-class metrics JSON.")
    p.add_argument("--head", type=int, default=20, help="Number of rows to print in terminal preview.")
    return p.parse_args()


def _normalize_labels(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.dropna().unique()) - VALID_4CLASS)
    if unknown:
        raise ValueError(f"Unexpected labels in {column_name}: {unknown}. Expected only {sorted(VALID_4CLASS)}")
    return normalized


def _load_threshold(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Stage1 threshold file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))

    if "best_threshold" in obj:
        threshold = float(obj["best_threshold"])
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Invalid best_threshold {threshold} in {path}")
        return {"mode": "binary", "best_threshold": threshold}

    if "route_policy" in obj and "t_pass" in obj:
        return {
            "mode": "joint",
            "route_policy": obj.get("route_policy", "margin_gate"),
            "t_pass": float(obj.get("t_pass", 0.55)),
            "t_notpass": float(obj.get("t_notpass", 0.40)),
            "t_margin": float(obj.get("t_margin", 0.05)),
            "route_default": obj.get("route_default", "pass"),
        }

    raise ValueError(
        f"Invalid threshold schema in {path}. Expected keys for either "
        f"binary mode: ['best_threshold'] "
        f"or joint mode: ['route_policy', 't_pass']."
    )


def _load_stage2_mapping(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Stage2 labels file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"Missing 'mapping' dict in {path}")

    parsed = {int(v): str(k) for k, v in mapping.items()}
    expected = {0: "distinction", 1: "fail", 2: "withdrawn"}
    if parsed != expected:
        raise ValueError(f"Unexpected stage2 mapping in {path}: {parsed}. Expected {expected}")
    return parsed


def _route_pass_mask(proba4: np.ndarray, cfg: dict) -> np.ndarray:
    p = np.asarray(proba4, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError(f"_route_pass_mask expects shape (n,4), got {p.shape}")

    p_dist = p[:, 0]
    p_fail = p[:, 1]
    p_pass = p[:, 2]
    p_wd = p[:, 3]
    notpass_max = np.maximum.reduce([p_dist, p_fail, p_wd])

    policy = cfg.get("route_policy", "margin_gate")
    t_pass = float(cfg.get("t_pass", 0.55))
    t_notpass = float(cfg.get("t_notpass", 0.40))
    t_margin = float(cfg.get("t_margin", 0.05))
    route_default = cfg.get("route_default", "pass")

    if policy == "pass_threshold":
        return p_pass >= t_pass

    pass_mask = (p_pass >= t_pass) & ((p_pass - notpass_max) >= t_margin)
    notpass_mask = notpass_max >= t_notpass
    undecided = ~(pass_mask | notpass_mask)
    if route_default == "pass":
        pass_mask = pass_mask | undecided
    return pass_mask


def _print_preview_table(df: pd.DataFrame, head: int) -> None:
    if head <= 0:
        return
    preview = df.head(head).copy()
    float_cols = [
        "pass_proba",
        "stage1_p_dist",
        "stage1_p_fail",
        "stage1_p_pass",
        "stage1_p_withdrawn",
        "proba_distinction",
        "proba_fail",
        "proba_withdrawn",
    ]
    for col in float_cols:
        if col in preview.columns:
            preview[col] = preview[col].map(lambda x: f"{float(x):.4f}")

    print("\nHierarchical Prediction Preview")
    print("-" * 118)
    print(preview.to_string(index=False))
    print("-" * 118)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    stage1_model_path = Path(args.stage1_model).resolve()
    stage1_threshold_path = Path(args.stage1_threshold).resolve()
    stage2_model_path = Path(args.stage2_model).resolve()
    stage2_labels_path = Path(args.stage2_labels).resolve()
    output_path = Path(args.output).resolve()
    metrics_output_path = Path(args.metrics_output).resolve()

    for p in [input_path, stage1_model_path, stage1_threshold_path, stage2_model_path, stage2_labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    threshold_cfg = _load_threshold(stage1_threshold_path)
    stage2_id_to_label = _load_stage2_mapping(stage2_labels_path)

    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    has_target = args.target in df.columns
    true_labels = _normalize_labels(df[args.target], args.target) if has_target else None
    X = df.drop(columns=[args.target]).copy() if has_target else df.copy()

    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    stage1_model = CatBoostClassifier()
    stage1_model.load_model(stage1_model_path)
    stage2_model = CatBoostClassifier()
    stage2_model.load_model(stage2_model_path)

    stage1_classes = [int(c) for c in stage1_model.classes_]
    stage1_proba = stage1_model.predict_proba(X)
    stage1_p_dist = None
    stage1_p_fail = None
    stage1_p_pass = None
    stage1_p_withdrawn = None
    if threshold_cfg["mode"] == "binary":
        if 1 not in stage1_classes:
            raise RuntimeError(f"Stage1 binary model classes do not include class id 1 (pass). classes_={stage1_classes}")
        pass_idx = stage1_classes.index(1)
        pass_proba = stage1_proba[:, pass_idx]
        pass_mask = pass_proba >= float(threshold_cfg["best_threshold"])
        stage1_decision = np.where(pass_mask, "pass", "notpass")
    else:
        proba4 = np.zeros((stage1_proba.shape[0], 4), dtype=np.float64)
        for col_idx, cls_id in enumerate(stage1_classes):
            if 0 <= int(cls_id) < 4:
                proba4[:, int(cls_id)] = stage1_proba[:, col_idx]
        stage1_p_dist = proba4[:, LABEL_ORDER_4.index("distinction")]
        stage1_p_fail = proba4[:, LABEL_ORDER_4.index("fail")]
        stage1_p_pass = proba4[:, LABEL_ORDER_4.index("pass")]
        stage1_p_withdrawn = proba4[:, LABEL_ORDER_4.index("withdrawn")]
        pass_proba = stage1_p_pass
        pass_mask = _route_pass_mask(proba4, threshold_cfg)
        stage1_decision = np.where(pass_mask, "pass", "notpass")

    n_rows = len(X)
    stage2_dist = np.zeros(n_rows, dtype=np.float64)
    stage2_fail = np.zeros(n_rows, dtype=np.float64)
    stage2_withdrawn = np.zeros(n_rows, dtype=np.float64)
    final_pred = np.full(n_rows, "pass", dtype=object)

    notpass_idx = np.where(stage1_decision == "notpass")[0]
    if len(notpass_idx) > 0:
        X_notpass = X.iloc[notpass_idx]
        stage2_proba = stage2_model.predict_proba(X_notpass)
        stage2_classes = [int(c) for c in stage2_model.classes_]
        stage2_col_idx = {cid: i for i, cid in enumerate(stage2_classes)}

        required = [0, 1, 2]
        if any(cid not in stage2_col_idx for cid in required):
            raise RuntimeError(f"Stage2 model classes mismatch. classes_={stage2_model.classes_}")

        p_dist = stage2_proba[:, stage2_col_idx[0]]
        p_fail = stage2_proba[:, stage2_col_idx[1]]
        p_with = stage2_proba[:, stage2_col_idx[2]]

        stage2_dist[notpass_idx] = p_dist
        stage2_fail[notpass_idx] = p_fail
        stage2_withdrawn[notpass_idx] = p_with

        stage2_pred_ids = np.argmax(stage2_proba[:, [stage2_col_idx[0], stage2_col_idx[1], stage2_col_idx[2]]], axis=1)
        mapped_ids = np.array([required[i] for i in stage2_pred_ids], dtype=int)
        final_pred_notpass = np.array([stage2_id_to_label[int(i)] for i in mapped_ids], dtype=object)
        final_pred[notpass_idx] = final_pred_notpass

    out_data = {
        "pass_proba": pass_proba,
        "stage1_decision": stage1_decision,
        "proba_distinction": stage2_dist,
        "proba_fail": stage2_fail,
        "proba_withdrawn": stage2_withdrawn,
        "final_pred_label": final_pred,
    }
    if threshold_cfg["mode"] == "joint":
        out_data["stage1_p_dist"] = stage1_p_dist
        out_data["stage1_p_fail"] = stage1_p_fail
        out_data["stage1_p_pass"] = stage1_p_pass
        out_data["stage1_p_withdrawn"] = stage1_p_withdrawn
    out = pd.DataFrame(out_data)

    if has_target and true_labels is not None:
        out["true_label"] = true_labels.values
        out["correct"] = (out["final_pred_label"] == out["true_label"]).astype(int)

    _print_preview_table(out, args.head)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Stage1 model: {stage1_model_path}")
    if threshold_cfg["mode"] == "binary":
        print(
            f"Stage1 threshold: {stage1_threshold_path} "
            f"(mode=binary, best_threshold={float(threshold_cfg['best_threshold']):.4f})"
        )
    else:
        print(
            f"Stage1 threshold: {stage1_threshold_path} "
            f"(mode=joint, route_policy={threshold_cfg.get('route_policy')}, "
            f"t_pass={float(threshold_cfg.get('t_pass', 0.55)):.2f}, "
            f"t_notpass={float(threshold_cfg.get('t_notpass', 0.40)):.2f}, "
            f"t_margin={float(threshold_cfg.get('t_margin', 0.05)):.2f}, "
            f"route_default={threshold_cfg.get('route_default', 'pass')})"
        )
    print(f"Stage2 model: {stage2_model_path}")
    print(f"Stage2 labels: {stage2_labels_path}")
    print(f"Saved predictions: {output_path}")

    if not has_target:
        print("Target column not found; skipped 4-class metrics.")
        return

    y_true = true_labels.to_numpy()
    y_pred = out["final_pred_label"].to_numpy()

    print("\nFinal classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["distinction", "fail", "pass", "withdrawn"],
            digits=4
        )
    )

    labels = ["distinction", "fail", "pass", "withdrawn"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title("Confusion Matrix (multiclass, K=4)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("outputs/hierarchical/confusion_matrix.png", dpi=300)
    plt.show()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=ORDER_4CLASS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=ORDER_4CLASS, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=ORDER_4CLASS)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=ORDER_4CLASS,
        target_names=ORDER_4CLASS,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "primary_metric": "macro_f1",
        "label_order": ORDER_4CLASS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "artifacts": {
            "stage1_model": str(stage1_model_path),
            "stage1_threshold": str(stage1_threshold_path),
            "stage2_model": str(stage2_model_path),
            "stage2_labels": str(stage2_labels_path),
            "predictions_csv": str(output_path),
        },
    }
    save_json(metrics, metrics_output_path)

    summary_rows = [
        ("Accuracy", f"{acc:.4f}"),
        ("Balanced Accuracy", f"{bal_acc:.4f}"),
        ("Macro F1 (primary)", f"{macro_f1:.4f}"),
        ("Weighted F1", f"{weighted_f1:.4f}"),
    ]

    print("\n4-Class Hierarchical Metrics")
    print("-" * 52)
    for key, value in summary_rows:
        print(f"{key:<28} {value:>22}")
    print("-" * 52)
    print(f"Saved 4-class metrics: {metrics_output_path}")


if __name__ == "__main__":
    main()
