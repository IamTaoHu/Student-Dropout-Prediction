import json
from pathlib import Path

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    # Ensure a directory exists and return its Path.
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_csv(path: str | Path) -> pd.DataFrame:
    # Load a CSV file or raise a clear error if missing.
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def save_json(obj: dict, path: str | Path) -> Path:
    # Save a JSON file with readable formatting.
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
    return json_path


def save_model(model, path: str | Path) -> Path:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def print_fixed_table(rows: list[list[str]], headers: list[str]) -> None:
    if not headers:
        return
    col_count = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for idx in range(col_count):
            cell = row[idx] if idx < len(row) else ""
            widths[idx] = max(widths[idx], len(str(cell)))

    def _format_row(cells: list[str]) -> str:
        padded = []
        for idx in range(col_count):
            cell = cells[idx] if idx < len(cells) else ""
            padded.append(str(cell).ljust(widths[idx]))
        return "  ".join(padded)

    print(_format_row(headers))
    for row in rows:
        print(_format_row([str(c) for c in row]))
