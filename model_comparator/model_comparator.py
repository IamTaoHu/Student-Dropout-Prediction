from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

COLUMNS = ["Model", "Accuracy", "F1-macro", "Recall-macro"]
NULL_STR = "null"


def _to_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return None  # non-scalar -> treat as missing


def _find_metric(metrics: Dict[str, Any], keys: List[str]) -> Any:
    # direct keys
    for k in keys:
        if k in metrics:
            return _to_scalar(metrics.get(k))

    # nested containers
    for container_key in ["metrics", "test", "eval", "evaluation", "results"]:
        container = metrics.get(container_key)
        if isinstance(container, dict):
            for k in keys:
                if k in container:
                    return _to_scalar(container.get(k))
    return None


def _extract_from_sklearn_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support sklearn classification_report as dict:
      {
        "accuracy": 0.75,
        "macro avg": {"precision":..., "recall":..., "f1-score":...},
        ...
      }
    """
    out = {"Accuracy": None, "F1-macro": None, "Recall-macro": None}

    rep = metrics.get("classification_report")
    if not isinstance(rep, dict):
        return out

    # accuracy may be present directly
    if "accuracy" in rep:
        out["Accuracy"] = _to_scalar(rep.get("accuracy"))

    macro = rep.get("macro avg")
    if isinstance(macro, dict):
        out["Recall-macro"] = _to_scalar(macro.get("recall"))
        out["F1-macro"] = _to_scalar(macro.get("f1-score"))

    return out


def _extract_confusion_terms(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support:
      A) explicit keys: TN/FP/FN/TP or tn/fp/fn/tp
      B) confusion_matrix: [[TN, FP], [FN, TP]]
      C) confusion_matrix: {"tn":..., "fp":..., "fn":..., "tp":...} (your CatBoost format)
    """
    out = {"TN": None, "FP": None, "FN": None, "TP": None}

    # A) direct keys (top-level)
    out["TN"] = _find_metric(metrics, ["TN", "tn"])
    out["FP"] = _find_metric(metrics, ["FP", "fp"])
    out["FN"] = _find_metric(metrics, ["FN", "fn"])
    out["TP"] = _find_metric(metrics, ["TP", "tp"])

    if all(out[k] is not None for k in out):
        return out

    cm = metrics.get("confusion_matrix")

    # C) dict format: {"tn":..., "fp":..., "fn":..., "tp":...}
    if isinstance(cm, dict):
        out["TN"] = _to_scalar(cm.get("tn") if "tn" in cm else cm.get("TN"))
        out["FP"] = _to_scalar(cm.get("fp") if "fp" in cm else cm.get("FP"))
        out["FN"] = _to_scalar(cm.get("fn") if "fn" in cm else cm.get("FN"))
        out["TP"] = _to_scalar(cm.get("tp") if "tp" in cm else cm.get("TP"))
        return out

    # B) list format: [[TN, FP], [FN, TP]]
    if isinstance(cm, list) and len(cm) == 2 and all(isinstance(r, list) and len(r) == 2 for r in cm):
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        out["TN"] = _to_scalar(tn)
        out["FP"] = _to_scalar(fp)
        out["FN"] = _to_scalar(fn)
        out["TP"] = _to_scalar(tp)

    return out

def _format_value(v: Any) -> str:
    if v is None:
        return NULL_STR
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)

def fmt(x: Any) -> str:
    if x is None:
        return "-"
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return "-"
        return f"{float(x):.4f}"
    except Exception:
        return "-"


def _escape_md(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", " ")


def read_metrics_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"__error__": "invalid_json"}


def find_metrics_path(model_dir: Path) -> Optional[Path]:
    candidates = [
        model_dir / "metrics.json",
        model_dir / "outputs" / "metrics.json",
        model_dir / "outputs" / "metrics" / "metrics.json",
        model_dir / "metrics" / "metrics.json",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def build_rows(base_dir: Path, debug: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not base_dir.exists():
        if debug:
            print(f"[DEBUG] base_dir not found: {base_dir}", file=sys.stderr)
        return rows

    subdirs = [p for p in base_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"]
    subdirs.sort(key=lambda p: p.name.lower())

    if debug:
        print(f"[DEBUG] base_dir = {base_dir}", file=sys.stderr)
        print(f"[DEBUG] found {len(subdirs)} subfolders: {[p.name for p in subdirs]}", file=sys.stderr)

    for model_dir in subdirs:
        name = model_dir.name
        metrics_path = find_metrics_path(model_dir)
        metrics = read_metrics_json(metrics_path) if metrics_path else None

        row: Dict[str, Any] = {"Model": name}

        if metrics is None or metrics.get("__error__") == "invalid_json":
            rows.append(row)
            if debug:
                print(f"[DEBUG] {name}: metrics.json missing/invalid at {metrics_path or 'NOT_FOUND'}", file=sys.stderr)
            continue

        row["F1-macro"] = _find_metric(metrics, ["f1_macro", "f1-macro", "f1_macro_avg", "f1", "F1-macro", "f1_macro_mean", "macro_f1", "f1_macro_score"])
        row["Recall-macro"] = _find_metric(metrics, ["recall_macro", "recall-macro", "recall", "Recall-macro", "macro_recall", "recall_macro_mean", "recall_macro_score"])
        row["Accuracy"] = _find_metric(metrics, ["accuracy", "acc", "Accuracy", "ACC"])
        fallback = _extract_from_sklearn_report(metrics)
        if row.get("Accuracy") is None and fallback["Accuracy"] is not None:
            row["Accuracy"] = fallback["Accuracy"]
        if row.get("F1-macro") is None and fallback["F1-macro"] is not None:
            row["F1-macro"] = fallback["F1-macro"]
        if row.get("Recall-macro") is None and fallback["Recall-macro"] is not None:
            row["Recall-macro"] = fallback["Recall-macro"]

        rows.append(row)

    return rows


def to_markdown_table(rows: List[Dict[str, Any]]) -> str:
    # 1) prepare string matrix
    table: List[List[str]] = []
    table.append(COLUMNS[:])  # header row

    for r in rows:
        row_cells: List[str] = []
        for col in COLUMNS:
            if col in ("Accuracy", "F1-macro", "Recall-macro"):
                cell = fmt(r.get(col))
            else:
                cell = _format_value(r.get(col))
            row_cells.append(_escape_md(cell))
        table.append(row_cells)

    # 2) compute column widths (max length among header+cells)
    widths = [0] * len(COLUMNS)
    for j in range(len(COLUMNS)):
        widths[j] = max(len(table[i][j]) for i in range(len(table)))

    # 3) build markdown lines with padding
    def pad(cell: str, w: int) -> str:
        return cell + (" " * (w - len(cell)))

    header = "| " + " | ".join(pad(table[0][j], widths[j]) for j in range(len(COLUMNS))) + " |"

    # separator: use dashes with same width to make it look aligned
    sep = "| " + " | ".join("-" * widths[j] if widths[j] >= 3 else "---" for j in range(len(COLUMNS))) + " |"

    lines = [header, sep]

    for i in range(1, len(table)):
        line = "| " + " | ".join(pad(table[i][j], widths[j]) for j in range(len(COLUMNS))) + " |"
        lines.append(line)

    return "\n".join(lines)

def main() -> None:
    debug = os.getenv("DEBUG", "0") == "1"
    # Output columns: Model, Accuracy, F1-macro, Recall-macro

    # โฟลเดอร์ model_comparator/ เป็นฐานตามโครงสร้างที่คุณตั้งไว้
    base_dir = Path(__file__).resolve().parent

    rows = build_rows(base_dir, debug=debug)

    # บังคับให้มี output เสมอ ถึงแม้ rows จะว่าง
    print(to_markdown_table(rows), flush=True)


if __name__ == "__main__":
    main()
