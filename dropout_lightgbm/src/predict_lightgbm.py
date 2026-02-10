from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from config import *

TARGET_CANDIDATES = [
    "Target",
    "target",
    "STATUS",
    "Status",
    "status",
    "Outcome",
    "outcome",
    "Class",
    "class",
]

CLASS_LABELS = ["dropout", "enrolled", "graduate"]
CLASS_ID_TO_LABEL = {0: "dropout", 1: "enrolled", 2: "graduate"}


def _drop_target_if_present(df: pd.DataFrame) -> pd.DataFrame:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return df.drop(columns=[candidate])
    return df


def main() -> None:
    model_path = OUTPUT_DIR / "lightgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Run: python train_lightgbm.py")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    X = _drop_target_if_present(df)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna("NA").astype(str)

    proba = model.predict_proba(X)
    proba_dropout = proba[:, 0]
    proba_enrolled = proba[:, 1]
    proba_graduate = proba[:, 2]
    pred_class_id = np.argmax(proba, axis=1)
    pred_label = [CLASS_ID_TO_LABEL[idx] for idx in pred_class_id]

    out_df = pd.DataFrame(
        {
            "proba_dropout": np.round(proba_dropout, 6),
            "proba_enrolled": np.round(proba_enrolled, 6),
            "proba_graduate": np.round(proba_graduate, 6),
            "pred_class_id": pred_class_id.astype(int),
            "pred_label": pred_label,
        }
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=10)
    args = parser.parse_args()

    print(out_df.head(args.head).to_string(index=False))

    out_path = OUTPUT_DIR / "predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
