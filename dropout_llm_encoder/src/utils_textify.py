from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


_TARGET_CANDIDATES = {"target", "label", "y", "status", "outcome", "class", "result"}


def _norm_col_name(name: str) -> str:
    return str(name).replace("\ufeff", "").strip().lower()


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a target column using common names (case-insensitive).
    Returns the first match in dataframe column order, or None if not found.
    """
    for col in df.columns:
        if _norm_col_name(col) in _TARGET_CANDIDATES:
            return col
    return None


def _value_to_text(value) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def row_to_text(row: pd.Series, target_col: Optional[str]) -> str:
    """
    Convert a row into a single string: "col=value; col=value; ..."
    Skips the target column when provided.
    """
    parts = []
    for col, value in row.items():
        if target_col is not None and col == target_col:
            continue
        parts.append(f"{col}={_value_to_text(value)}")
    return "; ".join(parts)


def df_to_texts(df: pd.DataFrame, target_col: Optional[str]) -> list[str]:
    """
    Convert a dataframe into a list of text strings, one per row.
    """
    return [row_to_text(row, target_col) for _, row in df.iterrows()]
