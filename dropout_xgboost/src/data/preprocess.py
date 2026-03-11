from __future__ import annotations

import pandas as pd

from hierarchical_xgb.feature_engineering import normalize_kuzilek_base_columns


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return cleaned


def normalize_kuzilek_features(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_kuzilek_base_columns(clean_dataframe_columns(df))


__all__ = ["clean_dataframe_columns", "normalize_kuzilek_features"]
