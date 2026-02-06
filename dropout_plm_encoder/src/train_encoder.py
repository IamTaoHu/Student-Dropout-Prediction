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

from .metrics import (
    LABEL2ID,
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


def _build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    epochs: int,
    task: str,
):
    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        if task == "multiclass":
            y_pred = probs.argmax(axis=1)
            return compute_metrics_multiclass(labels, y_pred, probs)
        y_proba = probs[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        return compute_metrics(labels, y_pred, y_proba, num_classes=2)

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    metric_key = "eval_f1_macro" if task == "multiclass" else "eval_f1"
    args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "logging_steps": 25,
        "seed": 42,
        "data_seed": 42,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_key,
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


def _print_binary_table(model_name: str, metrics: dict) -> None:
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
        model_name,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder-only dropout model.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["multiclass", "binary"],
        default="multiclass",
    )
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
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

    if args.task == "multiclass":
        labels = map_labels(df[target_col])
    else:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            unique_vals = set(pd.Series(df[target_col]).dropna().unique().tolist())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(
                    "Binary task requires numeric target values in {0, 1}. "
                    f"Found: {sorted(unique_vals)}"
                )
            labels = df[target_col].astype(int)
        else:
            mapping = {"dropout": 1, "graduate": 0}
            mapped = (
                df[target_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
            )
            if mapped.isna().any():
                bad = sorted({str(v) for v in df[target_col].dropna().unique().tolist()})
                raise ValueError(
                    "Binary task supports only Dropout/Graduate string labels "
                    f"(case-insensitive). Found: {bad}"
                )
            labels = mapped.astype(int)
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

    model_name = args.model_name
    # model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = 3 if args.task == "multiclass" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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
        task=args.task,
    )
    trainer.train()

    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    if args.task == "multiclass":
        y_pred = probs.argmax(axis=1)
        metrics = compute_metrics_multiclass(y_test, y_pred, probs)
    else:
        y_proba = probs[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba, num_classes=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    meta = {
        "task": args.task,
        "label2id": LABEL2ID if args.task == "multiclass" else None,
        "id2label": ID2LABEL if args.task == "multiclass" else None,
        "model_name": model_name,
        "max_length": args.max_length,
        "target_col": target_col,
    }
    with (model_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print_summary_table(model_name, metrics)
    if args.task == "multiclass":
        print_per_class_table(metrics, ID2LABEL)
        print_confusion_matrix(metrics, ID2LABEL)
    else:
        _print_binary_table(model_name, metrics)


if __name__ == "__main__":
    main()
