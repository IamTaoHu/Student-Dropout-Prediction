from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.utils import load_csv, load_model, print_fixed_table


CLASS_LABELS = ["dropout", "enrolled", "graduate"]
CLASS_ID_TO_LABEL = {0: "dropout", 1: "enrolled", 2: "graduate"}


def _default_input_path() -> Path:
    candidate = Path("input/data.csv")
    if candidate.exists():
        return candidate
    return Path(config.DATA_PATH)


def _build_preview_rows(
    df: pd.DataFrame,
    pred_label: list[str],
    proba_cols: list[str],
    proba_values: dict[str, list[float]],
    limit: int = 15,
) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []

    if "id" in df.columns:
        headers.append("id")

    headers.append("pred_label")
    headers.extend(proba_cols)

    for i in range(min(limit, len(pred_label))):
        row: list[str] = []
        if "id" in df.columns:
            row.append(str(df["id"].iloc[i]))
        row.append(pred_label[i])
        for col in proba_cols:
            row.append(f"{proba_values[col][i]:.4f}")
        rows.append(row)

    return headers, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=str(_default_input_path()))
    parser.add_argument("--model_path", default=str(Path(config.OUTPUT_DIR) / "model.joblib"))
    parser.add_argument("--output_csv", default=str(Path(config.OUTPUT_DIR) / "predictions.csv"))
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    model_path = Path(args.model_path)
    output_csv = Path(args.output_csv)

    df = load_csv(input_csv)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    model = load_model(model_path)

    if args.task == "multiclass":
        proba = model.predict_proba(df)
        proba_values = {
            "proba_dropout": proba[:, 0],
            "proba_enrolled": proba[:, 1],
            "proba_graduate": proba[:, 2],
        }
        pred_class_id = proba.argmax(axis=1)
        pred_label = [CLASS_ID_TO_LABEL[idx] for idx in pred_class_id]
        proba_cols = ["proba_dropout", "proba_enrolled", "proba_graduate"]
    else:
        proba = model.predict_proba(df)[:, 1]
        proba_values = {"proba_dropout": proba}
        pred_class_id = (proba >= 0.5).astype(int)
        pred_label = ["dropout" if v == 1 else "enrolled" for v in pred_class_id]
        proba_cols = ["proba_dropout"]

    out_df = pd.DataFrame({"pred_label": pred_label})
    for col in proba_cols:
        out_df[col] = proba_values[col]
    out_df.to_csv(output_csv, index=False)

    headers, rows = _build_preview_rows(df, pred_label, proba_cols, proba_values)
    print_fixed_table(rows, headers)
    print(f"Saved predictions to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
