from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils_textify import detect_target_column, df_to_texts


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))

    def __getitem__(self, idx: int):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def resolve_column_name(df: pd.DataFrame, name: str):
    def _norm(value: str) -> str:
        return str(value).replace("\ufeff", "").strip().lower()

    if name in df.columns:
        return name

    mapping = {_norm(col): col for col in df.columns}
    return mapping.get(_norm(name))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "input" / "data.csv"
    default_output = base_dir / "outputs" / "predictions.csv"
    model_dir = base_dir / "outputs" / "model"

    parser = argparse.ArgumentParser(description="Predict dropout using encoder model.")
    parser.add_argument("--input_csv", type=str, default=str(default_input))
    parser.add_argument("--output_csv", type=str, default=str(default_output))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-col", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(args.input_csv, sep=";")
    if args.target_col is not None:
        target_col = resolve_column_name(df, args.target_col)
    else:
        target_col = detect_target_column(df)
    if target_col is None:
        raise ValueError(
            f"Target column not found. Available columns: {list(df.columns)}"
        )
    texts = df_to_texts(df, target_col)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    if args.threshold is not None:
        print("[WARN] --threshold is ignored for multi-class predictions.")

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    y_pred_class = probs.argmax(axis=1)
    label_map = {0: "Graduate", 1: "Dropout", 2: "Enrolled"}
    pred_label = [label_map[int(c)] for c in y_pred_class]

    out_df = pd.DataFrame(
        {
            "pred_class": y_pred_class,
            "pred_label": pred_label,
            "prob_0": probs[:, 0],
            "prob_1": probs[:, 1],
            "prob_2": probs[:, 2],
            "pred_proba_dropout": probs[:, 1],
        }
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
