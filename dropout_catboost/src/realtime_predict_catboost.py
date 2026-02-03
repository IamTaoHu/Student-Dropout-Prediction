from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from src import config

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

    sample = X.iloc[[0]].copy()

    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    sample[cat_cols] = sample[cat_cols].astype(str).fillna("NA")

    model_path = Path(config.OUTPUT_DIR) / "catboost_model.cbm"
    model = CatBoostClassifier()
    model.load_model(model_path)

    proba = model.predict_proba(sample)[:, 1][0]
    print(f"Dropout risk: {proba:.4f}")


if __name__ == "__main__":
    main()
