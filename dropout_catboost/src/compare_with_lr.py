from __future__ import annotations

import json
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    workspace_dir = base_dir.parent
    lr_metrics_path = workspace_dir / "dropout_lr_baseline" / "outputs" / "metrics.json"
    cb_metrics_path = base_dir / "outputs" / "catboost_metrics.json"
    out_md_path = base_dir / "outputs" / "lr_vs_catboost_table.md"

    print("LR metrics path:", lr_metrics_path)
    print("CatBoost metrics path:", cb_metrics_path)

    lr_metrics = _load_metrics(lr_metrics_path)
    cb_metrics = _load_metrics(cb_metrics_path)

    keys = ["f1", "recall", "roc_auc", "pr_auc"]

    header = f"{'metric':<10} {'lr':>12} {'catboost':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for k in keys:
        lr_val = lr_metrics.get(k)
        cb_val = cb_metrics.get(k)
        lr_str = f"{lr_val:.4f}" if isinstance(lr_val, (int, float)) else "n/a"
        cb_str = f"{cb_val:.4f}" if isinstance(cb_val, (int, float)) else "n/a"
        lines.append(f"{k:<10} {lr_str:>12} {cb_str:>12}")

    print("\n".join(lines))

    md_lines = [
        "| metric | lr | catboost |",
        "|---|---:|---:|",
    ]
    for k in keys:
        lr_val = lr_metrics.get(k)
        cb_val = cb_metrics.get(k)
        lr_str = f"{lr_val:.4f}" if isinstance(lr_val, (int, float)) else "n/a"
        cb_str = f"{cb_val:.4f}" if isinstance(cb_val, (int, float)) else "n/a"
        md_lines.append(f"| {k} | {lr_str} | {cb_str} |")

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
