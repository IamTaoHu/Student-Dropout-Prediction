from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(PROJECT_ROOT))).resolve()

DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

if not 0 < TEST_SIZE < 1:
    raise ValueError("TEST_SIZE must be between 0 and 1.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
