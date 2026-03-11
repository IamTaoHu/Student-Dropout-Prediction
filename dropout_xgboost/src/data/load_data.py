from __future__ import annotations

from pathlib import Path

import pandas as pd

from hierarchical_xgb.data_utils import load_data as load_hierarchical_data
from hierarchical_xgb.data_utils import resolve_input_path
from utils.io_utils import load_csv


def load_csv_data(path: str | Path) -> pd.DataFrame:
    return load_csv(Path(path))


__all__ = ["load_csv_data", "load_hierarchical_data", "resolve_input_path"]
