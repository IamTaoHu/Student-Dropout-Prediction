from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()

DATA_PATH = (PROJECT_ROOT / os.getenv("DATA_PATH", "./data/data.csv")).resolve()
OUTPUT_DIR = (PROJECT_ROOT / os.getenv("OUTPUT_DIR", "./outputs")).resolve()

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

if not 0 < TEST_SIZE < 1:
    raise ValueError("TEST_SIZE must be between 0 and 1.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
