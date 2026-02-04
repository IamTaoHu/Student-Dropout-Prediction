from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .metrics import (
    ID2LABEL,
    compute_metrics,
    compute_metrics_multiclass,
    map_labels,
    print_confusion_matrix,
    print_per_class_table,
    print_summary_table,
)
from .utils_textify import detect_target_column, df_to_texts


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
    parser.add_argument(
        "--task",
        type=str,
        choices=["multiclass", "binary"],
        default="multiclass",
    )
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
    print("[INFO] Loaded model from outputs/model")
    model.eval()

    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    if args.task == "multiclass":
        y_pred_class = probs.argmax(axis=1)
        pred_label = [ID2LABEL[int(c)] for c in y_pred_class]
    else:
        y_proba = probs[:, 1]
        y_pred_class = (y_proba >= args.threshold).astype(int)
        pred_label = [str(int(c)) for c in y_pred_class]

    y_true = None
    def _print_binary_table(metrics: dict) -> None:
        headers = [
            ("Model", 18),
            ("accuracy", 10),
            ("f1", 10),
            ("recall", 10),
            ("roc_auc", 10),
            ("pr_auc", 10),
            ("TN", 6),
            ("FP", 6),
            ("FN", 6),
            ("TP", 6),
        ]
        row = [
            "PLM-Encoder",
            metrics.get("accuracy"),
            metrics.get("f1"),
            metrics.get("recall"),
            metrics.get("roc_auc"),
            metrics.get("pr_auc"),
            metrics.get("TN"),
            metrics.get("FP"),
            metrics.get("FN"),
            metrics.get("TP"),
        ]

        def _fmt_local(v):
            if v is None:
                return "NA"
            if isinstance(v, (float, np.floating)):
                return f"{v:.4f}"
            return str(v)

        header_line = " | ".join(h.ljust(w) for h, w in headers)
        sep_line = "-+-".join("-" * w for _, w in headers)
        row_line = " | ".join(
            _fmt_local(val).ljust(w) for (val, (_, w)) in zip(row, headers)
        )
        print(header_line)
        print(sep_line)
        print(row_line)

    if target_col is not None:
        try:
            if args.task == "multiclass":
                y_true = map_labels(df[target_col])
                metrics = compute_metrics_multiclass(y_true, y_pred_class, probs)
                print_summary_table(model_name="PLM-Encoder", metrics=metrics)
                print_per_class_table(metrics, ID2LABEL)
                print_confusion_matrix(metrics, ID2LABEL)
            else:
                y_true = df[target_col].astype(int)
                metrics = compute_metrics(y_true, y_pred_class, probs[:, 1], num_classes=2)
                _print_binary_table(metrics)
        except Exception as exc:
            print(f"[WARN] Could not compute metrics: {exc}")

    if args.task == "multiclass":
        out_df = pd.DataFrame(
            {
                "pred_class": y_pred_class,
                "pred_label": pred_label,
                "prob_dropout": probs[:, 0],
                "prob_enrolled": probs[:, 1],
                "prob_graduate": probs[:, 2],
            }
        )
        if y_true is not None:
            out_df.insert(0, "y_true", y_true)
    else:
        out_df = pd.DataFrame(
            {
                "pred_class": y_pred_class,
                "pred_label": pred_label,
                "prob_1": probs[:, 1],
                "pred_proba_dropout": probs[:, 1],
            }
        )
        if y_true is not None:
            out_df.insert(0, "y_true", y_true)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
