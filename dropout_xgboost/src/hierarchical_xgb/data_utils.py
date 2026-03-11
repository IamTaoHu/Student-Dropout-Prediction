from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_xgb.config import CANONICAL_LABEL_ORDER, TARGET_CANDIDATES


def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]
    if isinstance(obj, pd.Series):
        return [to_py(v) for v in obj.tolist()]
    if isinstance(obj, pd.Index):
        return [to_py(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, np.ndarray):
        return [to_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_py(obj), f, indent=2, sort_keys=True)
    return path


def _detect_target_column(columns: pd.Index) -> str:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def normalize_label(s: str) -> str:
    if s is None:
        return ""
    x = str(s).strip().lower()
    return {
        "distinction": "distinction", "dist": "distinction", "fail": "fail", "pass": "pass",
        "withdrawn": "withdrawn", "withdraw": "withdrawn", "withdrew": "withdrawn",
    }.get(x, x)


def load_data(input_path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, int]]:
    df = pd.read_csv(Path(input_path), sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    target_column = _detect_target_column(df.columns)
    y_raw = df[target_column].map(normalize_label)
    unknown = sorted(set(y_raw.dropna().unique().tolist()) - set(CANONICAL_LABEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}. Expected subset of {CANONICAL_LABEL_ORDER}.")

    label_mapping = {name: i for i, name in enumerate(CANONICAL_LABEL_ORDER)}
    valid = y_raw.notna()
    X_df = df.drop(columns=[target_column]).loc[valid].copy()
    y_int = y_raw.loc[valid].map(label_mapping).astype(int)
    return X_df, y_int, CANONICAL_LABEL_ORDER.copy(), label_mapping


def _is_feature_identifier(col: str) -> bool:
    low = col.lower().strip()
    return low in {"id", "id_student"} or low.startswith("id_") or low.endswith("_id")


def resolve_input_path(input_arg_raw: str) -> Path:
    input_arg = Path(input_arg_raw).expanduser()
    resolved = input_arg if input_arg.is_absolute() else (Path.cwd() / input_arg)
    resolved = resolved.resolve()
    print(f"Input CSV (resolved): {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {resolved}. "
            "Run from project root or pass an absolute path to --input."
        )
    return resolved
