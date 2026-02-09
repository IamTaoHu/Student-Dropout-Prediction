from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.explain.explain_shap import compute_global_shap_bar, compute_local_shap_waterfall

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
]


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    X = df.drop(columns=[target_col])

    outdir = Path(config.OUTPUT_DIR) / "explain"

    global_path = compute_global_shap_bar(X, outdir)
    local_path = compute_local_shap_waterfall(X.iloc[[0]], outdir)

    print(f"Saved global SHAP bar: {global_path}")
    print(f"Saved local SHAP waterfall: {local_path}")


if __name__ == "__main__":
    main()
