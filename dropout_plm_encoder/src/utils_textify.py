from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_TARGET_CANDIDATES = {"target", "label", "y", "status", "outcome", "class", "result"}

def _norm_col_name(name: str) -> str:
    return str(name).replace("\ufeff", "").strip().lower()

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _norm_col_name(col) in _TARGET_CANDIDATES:
            return col
    return None

def _clean_col_display(name: str) -> str:
    # remove weird tabs/multiple spaces in headers (your CSV has '\t' in one column name)
    return " ".join(str(name).replace("\t", " ").split())

def fit_textify_spec(
    df: pd.DataFrame,
    target_col: Optional[str],
    n_bins: int = 10,
    max_unique_as_categorical: int = 25,
    force_continuous_cols: Optional[Tuple[str, ...]] = (
        "admission grade",
        "previous qualification (grade)",
        "curricular units 1st sem (grade)",
        "curricular units 2nd sem (grade)",
        "unemployment rate",
        "inflation rate",
        "gdp",
        "age at enrollment",
    ),
) -> Dict[str, Any]:
    """
    Build a per-column spec describing how to convert values to tokens.
    Strategy (best for PLM on tabular-as-text):
      - Treat low-cardinality numerics as categorical tokens.
      - Treat continuous numerics as quantile bins (qcut) -> bin_0..bin_{k-1}
      - Keep strings/categoricals as cleaned categorical tokens.
    """
    spec: Dict[str, Any] = {"n_bins": int(n_bins), "columns": {}}

    force_set = set(force_continuous_cols or ())
    # work on a copy of columns, skipping target
    for col in df.columns:
        if target_col is not None and col == target_col:
            continue

        col_norm = _norm_col_name(col)
        series = df[col]

        # treat object as categorical
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            spec["columns"][col] = {"kind": "cat"}
            continue

        # numeric path
        s = pd.to_numeric(series, errors="coerce")
        nunique = int(pd.Series(s.dropna()).nunique())

        # force continuous by name (robust for your dataset)
        if col_norm in force_set:
            kind = "bin"
        else:
            # low-card numeric -> categorical token
            kind = "cat" if nunique <= max_unique_as_categorical else "bin"

        if kind == "cat":
            spec["columns"][col] = {"kind": "cat"}
        else:
            # build quantile edges on available numeric values
            vals = s.dropna().astype(float)
            if len(vals) < 20 or vals.nunique() < 4:
                # too few unique values -> fallback to categorical
                spec["columns"][col] = {"kind": "cat"}
            else:
                # quantile cut; duplicates='drop' handles repeated values
                try:
                    _, bins = pd.qcut(vals, q=n_bins, retbins=True, duplicates="drop")
                    bins = np.unique(bins).astype(float)
                    # ensure strictly increasing
                    if len(bins) >= 3:
                        spec["columns"][col] = {"kind": "bin", "bins": bins.tolist()}
                    else:
                        spec["columns"][col] = {"kind": "cat"}
                except Exception:
                    spec["columns"][col] = {"kind": "cat"}

    return spec

def _value_to_token(value: Any, col: str, col_spec: Dict[str, Any]) -> str:
    if pd.isna(value):
        return "NA"

    kind = col_spec.get("kind", "cat")

    # categorical tokenization (works for ints that are actually codes)
    if kind == "cat":
        v = str(value).strip()
        if v == "":
            return "NA"
        # compress long floats like 13.666666666 -> 13.67 (still as token)
        try:
            fv = float(v)
            # if it looks like an int, keep as int-like string
            if abs(fv - round(fv)) < 1e-9:
                v = str(int(round(fv)))
            else:
                v = f"{fv:.2f}"
        except Exception:
            pass
        return f"cat_{v}"

    # binned numeric tokenization
    if kind == "bin":
        bins = col_spec.get("bins")
        try:
            x = float(value)
        except Exception:
            return "NA"
        if not bins or len(bins) < 3:
            return "cat_NA"
        # digitize into bin index: 0..k-2
        edges = np.asarray(bins, dtype=float)
        idx = int(np.digitize([x], edges[1:-1], right=False)[0])  # ignore -inf/+inf
        idx = max(0, min(idx, len(edges) - 2))
        return f"bin_{idx}"

    # fallback
    return f"cat_{str(value).strip()}"

def row_to_text(row: pd.Series, target_col: Optional[str], spec: Dict[str, Any]) -> str:
    parts = []
    cols_spec = spec.get("columns", {})

    for col, value in row.items():
        if target_col is not None and col == target_col:
            continue
        col_disp = _clean_col_display(col)
        col_s = cols_spec.get(col, {"kind": "cat"})
        tok = _value_to_token(value, col, col_s)
        parts.append(f"{col_disp}={tok}")
    return "; ".join(parts)

def df_to_texts(df: pd.DataFrame, target_col: Optional[str], spec: Optional[Dict[str, Any]] = None) -> list[str]:
    """
    Convert dataframe rows to tokenized strings using a fitted spec.
    If spec is None, fit a spec on this df (OK for quick experiments, but train should save spec).
    """
    if spec is None:
        spec = fit_textify_spec(df, target_col)
    return [row_to_text(row, target_col, spec) for _, row in df.iterrows()]
