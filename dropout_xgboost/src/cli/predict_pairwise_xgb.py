from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.pairwise_xgb.predict_pairwise import main as run_main


def main() -> None:
    run_main()


if __name__ == "__main__":
    main()
