import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd


LABEL_MAP_STR = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
LABEL_MAP_INT = {0: 0, 1: 1, 2: 2}


def read_csv_auto(path: Path) -> pd.DataFrame:
    # Prefer semicolon; fallback to auto-detect.
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) >= 10:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python")


def detect_target_column(df: pd.DataFrame) -> str:
    preferred = ["final_result", "FinalResult", "target", "label", "class", "outcome", "result"]
    normalized_to_original = {
        str(c).replace("\ufeff", "").strip().lower(): c
        for c in df.columns
    }
    for candidate in preferred:
        key = str(candidate).strip().lower()
        if key in normalized_to_original:
            return normalized_to_original[key]
    preview_cols = list(df.columns)[:30]
    raise ValueError(
        f"Target column not found. First 30 columns: {preview_cols}. "
        "Pass --target_col <colname>"
    )


def map_labels(y_raw: pd.Series) -> np.ndarray:
    """
    Robust label mapping for 4-class Kuzilek labels:
    - String labels (case/space tolerant): Distinction/Fail/Pass/Withdrawn -> 0/1/2/3
    - Numeric labels: 0/1/2/3 (including numeric strings like "1", "2.0")
    """
    s_str = y_raw.astype(str).str.strip()
    s_norm = s_str.str.lower()

    missing_tokens = {"", "nan", "none", "null", "na"}
    is_missing = s_norm.isin(missing_tokens)

    str_map = {
        "distinction": 0,
        "fail": 1,
        "pass": 2,
        "withdrawn": 3,
    }
    non_missing = s_norm[~is_missing]

    # String-label route if all non-missing values match supported 4-class labels.
    if len(non_missing) > 0 and non_missing.isin(set(str_map.keys())).all():
        mapped = s_norm.map(str_map)
        mapped[is_missing] = np.nan
        if mapped.isna().any():
            bad_vals = sorted(set(s_str[mapped.isna()].tolist()))[:10]
            raise ValueError(
                f"Unknown/missing label values found (examples): {bad_vals}. "
                "Supported labels: Distinction/Fail/Pass/Withdrawn or numeric 0..3."
            )
        return mapped.astype(int).to_numpy()

    # Numeric route (also supports numeric strings).
    y_num = pd.to_numeric(y_raw, errors="coerce")
    y_num[is_missing] = np.nan
    if y_num.isna().any():
        bad_vals = sorted(set(s_str[y_num.isna()].tolist()))[:10]
        raise ValueError(
            f"Numeric label parsing failed (examples): {bad_vals}. "
            "Supported labels: Distinction/Fail/Pass/Withdrawn or numeric 0..3."
        )

    y_int = y_num.astype(int)
    invalid_mask = ~y_int.isin([0, 1, 2, 3])
    if invalid_mask.any():
        bad_vals = sorted(set(y_int[invalid_mask].tolist()))[:10]
        raise ValueError(
            f"Numeric labels must be in 0..3. Found invalid values (examples): {bad_vals}. "
            "Supported labels: Distinction/Fail/Pass/Withdrawn or numeric 0..3."
        )

    return y_int.to_numpy(dtype=int)

def split_columns(
    df: pd.DataFrame,
    max_unique_for_cat: int = 50,
) -> Tuple[List[str], List[str]]:
    # Heuristic:
    # - object -> categorical
    # - numeric with small cardinality -> categorical (codes)
    # - otherwise -> numeric
    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            cat_cols.append(c)
            continue

        s_num = pd.to_numeric(s, errors="coerce")
        nunique = int(pd.Series(s_num.dropna()).nunique())
        if nunique <= max_unique_for_cat:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return cat_cols, num_cols


def fit_preprocess(
    X: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
) -> Dict[str, Any]:
    prep: Dict[str, Any] = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_maps": {},
        "num_mean": {},
        "num_std": {},
    }

    # categorical: map to 0..K (reserve 0 for UNK)
    for c in cat_cols:
        vals = X[c].astype(str).fillna("NA").str.strip()
        uniq = sorted(vals.unique().tolist())
        m = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved for UNK
        prep["cat_maps"][c] = m

    # numeric: standardize
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce").astype(float)
        mean = float(np.nanmean(s))
        std = float(np.nanstd(s))
        if std < 1e-6:
            std = 1.0
        prep["num_mean"][c] = mean
        prep["num_std"][c] = std

    return prep


def transform(
    X: pd.DataFrame,
    prep: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    cat_cols = prep["cat_cols"]
    num_cols = prep["num_cols"]

    # categorical -> int64 ids
    if cat_cols:
        cat_arr = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            m = prep["cat_maps"][c]
            vals = X[c].astype(str).fillna("NA").str.strip().tolist()
            cat_arr[:, j] = np.array([m.get(v, 0) for v in vals], dtype=np.int64)
    else:
        cat_arr = np.zeros((len(X), 0), dtype=np.int64)

    # numeric -> float32 standardized
    if num_cols:
        num_arr = np.zeros((len(X), len(num_cols)), dtype=np.float32)
        for j, c in enumerate(num_cols):
            mean = prep["num_mean"][c]
            std = prep["num_std"][c]
            s = pd.to_numeric(X[c], errors="coerce").astype(float).to_numpy()
            s = np.where(np.isnan(s), mean, s)
            num_arr[:, j] = ((s - mean) / std).astype(np.float32)
    else:
        num_arr = np.zeros((len(X), 0), dtype=np.float32)

    return cat_arr, num_arr


def save_preprocess(prep: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(prep, f, indent=2)


def load_preprocess(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
