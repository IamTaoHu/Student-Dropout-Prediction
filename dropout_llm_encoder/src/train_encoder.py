from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from metrics import compute_metrics
from utils_textify import detect_target_column, df_to_texts


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def resolve_column_name(df: pd.DataFrame, name: str) -> Optional[str]:
    def _norm(value: str) -> str:
        return str(value).replace("\ufeff", "").strip().lower()

    if name in df.columns:
        return name

    mapping = {_norm(col): col for col in df.columns}
    return mapping.get(_norm(name))


def _map_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique().tolist())
        if unique_vals.issubset({0, 1, 2}):
            return series.astype(int)
        raise ValueError(
            "Numeric target column must contain only {0, 1, 2} for multi-class "
            f"dropout prediction. Found: {sorted(unique_vals)}"
        )

    mapping = {"dropout": 1, "graduate": 0, "enrolled": 2}
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )
    if mapped.isna().any():
        bad = sorted(
            {str(v) for v in series.dropna().unique().tolist()}
        )
        raise ValueError(
            "Could not map target labels. Expected one of: Dropout, Graduate, Enrolled "
            f"(case-insensitive). Found: {bad}"
        )
    return mapped.astype(int)


def _build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    epochs: int,
):
    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        y_proba = probs[:, 1]
        y_pred = probs.argmax(axis=1)
        return compute_metrics(labels, y_pred, y_proba, num_classes=3)

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "logging_steps": 25,
        "seed": 42,
        "data_seed": 42,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
    }
    if "evaluation_strategy" not in supported and "eval_strategy" in supported:
        args_kwargs["eval_strategy"] = args_kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" not in supported:
        args_kwargs.pop("evaluation_strategy", None)
    if "save_strategy" not in supported:
        args_kwargs.pop("save_strategy", None)

    args_kwargs = {k: v for k, v in args_kwargs.items() if k in supported}
    args = TrainingArguments(**args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": hf_compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def _print_metrics_table(model_name: str, metrics: dict) -> None:
    row = [
        model_name,
        f"{metrics.get('f1'):.4f}" if metrics.get("f1") is not None else "NA",
        f"{metrics.get('recall'):.4f}" if metrics.get("recall") is not None else "NA",
        f"{metrics.get('roc_auc'):.4f}" if metrics.get("roc_auc") is not None else "NA",
        f"{metrics.get('pr_auc'):.4f}" if metrics.get("pr_auc") is not None else "NA",
        f"{metrics.get('accuracy'):.4f}" if metrics.get("accuracy") is not None else "NA",
        str(metrics.get("TN")),
        str(metrics.get("FP")),
        str(metrics.get("FN")),
        str(metrics.get("TP")),
    ]
    header = ["Model", "f1", "recall", "roc_auc", "pr_auc", "accuracy", "TN", "FP", "FN", "TP"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder-only dropout model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Explicit target column name (optional)",
    )
    args = parser.parse_args()

    _set_all_seeds(42)

    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "input" / "data.csv"
    output_dir = base_dir / "outputs"
    model_dir = output_dir / "model"
    metrics_path = output_dir / "metrics.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(input_path, sep=";")
        print("[INFO] Detected semicolon-delimited CSV. Reloaded with sep=';'.")
    if args.target_col is not None:
        target_col = resolve_column_name(df, args.target_col)
    else:
        target_col = detect_target_column(df)
    if target_col is None:
        if args.target_col is not None:
            provided = args.target_col
            raise ValueError(
                "Target column not found. "
                f"Provided target: {provided}. "
                f"Available columns: {list(df.columns)}. "
                "Hint: column names may contain spaces/BOM and matching is case-insensitive."
            )
        raise ValueError(
            "Target column not found. Expected one of: Target, target, label, y. "
            f"Available columns: {list(df.columns)}. "
            "Hint: column names may contain spaces/BOM and matching is case-insensitive."
        )

    labels = _map_labels(df[target_col])
    print(set(labels))
    texts = df_to_texts(df, target_col)

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    model_name = "prajjwal1/bert-tiny"
    # model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    train_ds = TextDataset(x_train, y_train.tolist(), tokenizer, max_length=args.max_length)
    val_ds = TextDataset(x_val, y_val.tolist(), tokenizer, max_length=args.max_length)
    test_ds = TextDataset(x_test, y_test.tolist(), tokenizer, max_length=args.max_length)

    model.to("cpu")
    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=model_dir,
        epochs=args.epochs,
    )
    trainer.train()

    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_proba = probs[:, 1]
    y_pred = probs.argmax(axis=1)
    metrics = compute_metrics(y_test, y_pred, y_proba, num_classes=3)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    _print_metrics_table(model_name, metrics)


if __name__ == "__main__":
    main()
