from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import inspect
import matplotlib.pyplot as plt
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from metrics import ID2LABEL, compute_metrics_multiclass, map_labels
from utils_textify import detect_target_column, df_to_texts, fit_textify_spec

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
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
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

    mapping = {_norm(c): c for c in df.columns}
    return mapping.get(_norm(name))


def _read_csv_auto(path: Path) -> pd.DataFrame:
    # Prefer semicolon-separated format (expected for this dataset).
    # If it doesn't parse well, fallback to auto-detect.
    try:
        df = pd.read_csv(path, sep=";")
        # If semicolon parsing gives a reasonable number of columns, accept it.
        if len(df.columns) >= 10:
            return df
    except Exception:
        pass
    # Fallback: auto-detect delimiter (comma/semicolon/tab)
    return pd.read_csv(path, sep=None, engine="python")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PLM encoder dropout model (3-class).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-col", type=str, default=None, help="Explicit target column name (optional)")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (CPU-friendly default)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (DeBERTa-small default)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--overfit-n", type=int, default=0,
                        help="If >0, use only first N samples for train/val/test (sanity overfit).")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weights (debug collapse).")
    parser.add_argument("--n-bins", type=int, default=10, help="Quantile bins for continuous numeric features")
    parser.add_argument("--max-unique-cat", type=int, default=25, help="<= this unique numeric values -> categorical token")
    args = parser.parse_args()

    _set_all_seeds(42)

    # Auto device (works on CPU-only machines and GPU machines)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model backbone: {args.model_name}")
    print(
        f"[INFO] batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
        f"lr={args.lr}, wd={args.weight_decay}, warmup={args.warmup_ratio}, gamma={args.gamma}"
    )

    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "input" / "data.csv"
    output_dir = base_dir / "outputs"
    model_dir = output_dir / "model"
    metrics_path = output_dir / "metrics.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_auto(input_path)

    if args.target_col is not None:
        target_col = resolve_column_name(df, args.target_col)
    else:
        target_col = detect_target_column(df)

    if target_col is None:
        raise ValueError(
            "Target column not found. "
            f"Available columns: {list(df.columns)}. "
            "Hint: use --target-col if your label column is not one of {target,label,y,status,outcome,class,result}."
        )

    print(f"Detected target column: {target_col}")

    # 3-class mapping: Dropout=0, Enrolled=1, Graduate=2 (supports strings or numeric {0,1,2})
    labels = map_labels(df[target_col])
    import numpy as np
    labels_arr_dbg = np.array(labels)
    print("[INFO] Label distribution (mapped) full:", {
        "0": int((labels_arr_dbg==0).sum()),
        "1": int((labels_arr_dbg==1).sum()),
        "2": int((labels_arr_dbg==2).sum()),
    })
    print()
    print("[INFO] Label set:", sorted(set(labels.tolist() if hasattr(labels, "tolist") else list(labels))))

    # Show class distribution after mapping (0/1/2)
    vc = pd.Series(labels).value_counts().sort_index()
    print("Class distribution after mapping:")
    print(pd.DataFrame({target_col: vc}))
    print()

    # Fit textify spec ONCE in training and save it for prediction consistency
    textify_spec = fit_textify_spec(
        df,
        target_col=target_col,
        n_bins=int(args.n_bins),
        max_unique_as_categorical=int(args.max_unique_cat),
    )

    textify_spec_path = model_dir / "textify_spec.json"
    with textify_spec_path.open("w", encoding="utf-8") as f:
        json.dump(textify_spec, f, indent=2)
    print(f"[INFO] Saved textify spec to: {textify_spec_path}")

    texts = df_to_texts(df, target_col, spec=textify_spec)
    if args.overfit_n and args.overfit_n > 0:
        n = int(args.overfit_n)
        labels_arr = np.array(labels)

        # indices per class
        idx0 = np.where(labels_arr == 0)[0]
        idx1 = np.where(labels_arr == 1)[0]
        idx2 = np.where(labels_arr == 2)[0]

        # take balanced counts
        per = max(1, n // 3)
        rng = np.random.default_rng(42)

        take0 = rng.choice(idx0, size=min(per, len(idx0)), replace=False) if len(idx0) else np.array([], dtype=int)
        take1 = rng.choice(idx1, size=min(per, len(idx1)), replace=False) if len(idx1) else np.array([], dtype=int)
        take2 = rng.choice(idx2, size=min(per, len(idx2)), replace=False) if len(idx2) else np.array([], dtype=int)

        take = np.concatenate([take0, take1, take2])
        rng.shuffle(take)

        texts = [texts[i] for i in take]
        labels = labels_arr[take]

        print(f"[INFO] Overfit mode ON (balanced): requested={n}, actual={len(texts)} "
              f"(0:{(labels==0).sum()} 1:{(labels==1).sum()} 2:{(labels==2).sum()})")
        print()

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

    if args.no_class_weights:
        class_weights = None
        print("[INFO] Class weights DISABLED")
        print()
    else:
        classes = np.array([0, 1, 2])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train).astype(np.float32)
        weights = np.power(weights, 0.5)
        weights = np.clip(weights, 0.5, 2.0)
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        print("[INFO] Soft class weights (sqrt+clip):", class_weights.detach().cpu().numpy().tolist())
        print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.to(device)

    train_ds = TextDataset(x_train, y_train.tolist(), tokenizer, max_length=args.max_length)
    val_ds = TextDataset(x_val, y_val.tolist(), tokenizer, max_length=args.max_length)
    test_ds = TextDataset(x_test, y_test.tolist(), tokenizer, max_length=args.max_length)

    def hf_compute_metrics(eval_pred):
        logits, y_true = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()  # (n,3)
        y_pred = probs.argmax(axis=1)
        m = compute_metrics_multiclass(y_true, y_pred, probs)
        # Return scalars for Trainer
        return {
            "accuracy": m.get("accuracy"),
            "f1_macro": m.get("f1_macro"),
            "recall_macro": m.get("recall_macro"),
            "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro"),
            "pr_auc_ovr_macro": m.get("pr_auc_ovr_macro"),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
            **kwargs,
        ):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            logits = logits.float()
            labels = labels.long()
            if class_weights is None:
                loss_fct = nn.CrossEntropyLoss()
            else:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights.float())
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    args_kwargs = {
        "output_dir": str(model_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "logging_steps": 25,
        "seed": 42,
        "data_seed": 42,
        "fp16": bool(torch.cuda.is_available()),
    }

    if "learning_rate" in supported:
        args_kwargs["learning_rate"] = float(args.lr)
    if "weight_decay" in supported:
        args_kwargs["weight_decay"] = float(args.weight_decay)
    if "warmup_steps" in supported:
        steps_per_epoch = max(
            1,
            int(
                np.ceil(
                    len(train_ds)
                    / max(1, args.batch_size * max(1, args.grad_accum))
                )
            ),
        )
        total_steps = steps_per_epoch * int(args.epochs)
        args_kwargs["warmup_steps"] = int(0.1 * total_steps)
    elif "warmup_ratio" in supported:
        args_kwargs["warmup_ratio"] = float(args.warmup_ratio)
    if "lr_scheduler_type" in supported:
        args_kwargs["lr_scheduler_type"] = "linear"
    if "gradient_accumulation_steps" in supported:
        args_kwargs["gradient_accumulation_steps"] = int(args.grad_accum)
    if "dataloader_num_workers" in supported:
        args_kwargs["dataloader_num_workers"] = 0
    if "dataloader_pin_memory" in supported and not torch.cuda.is_available():
        args_kwargs["dataloader_pin_memory"] = False

    # Some transformers versions use evaluation_strategy, others use eval_strategy, some have neither.
    has_eval_strategy = ("evaluation_strategy" in supported) or ("eval_strategy" in supported)
    has_save_strategy = ("save_strategy" in supported)

    if has_eval_strategy and has_save_strategy:
        # enable evaluation + saving each epoch
        if "evaluation_strategy" in supported:
            args_kwargs["evaluation_strategy"] = "epoch"
        else:
            args_kwargs["eval_strategy"] = "epoch"

        args_kwargs["save_strategy"] = "epoch"

        # load_best_model_at_end is only safe when eval & save strategies exist and match
        if "load_best_model_at_end" in supported:
            args_kwargs["load_best_model_at_end"] = True
        if "metric_for_best_model" in supported:
            args_kwargs["metric_for_best_model"] = "eval_f1_macro"
        if "greater_is_better" in supported:
            args_kwargs["greater_is_better"] = True
    else:
        # fallback for older versions: disable load_best_model_at_end to avoid ValueError
        if "load_best_model_at_end" in supported:
            args_kwargs["load_best_model_at_end"] = False
        print(
            "[WARN] TrainingArguments missing eval/save strategy in this transformers version. "
            "Disabled load_best_model_at_end for compatibility."
        )

    # Keep only supported keys
    args_kwargs = {k: v for k, v in args_kwargs.items() if k in supported}
    training_args = TrainingArguments(**args_kwargs)


    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": hf_compute_metrics,
    }

    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_supported = set(trainer_sig.parameters.keys())

    # tokenizer arg exists in many versions, but not all
    if "tokenizer" in trainer_supported:
        trainer_kwargs["tokenizer"] = tokenizer

    # callbacks arg exists in many versions, but not all
    if "callbacks" in trainer_supported:
        try:
            trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=2)]
        except Exception:
            # If EarlyStoppingCallback is missing or incompatible, skip it.
            print("[WARN] EarlyStoppingCallback not available in this transformers version. Continuing without it.")

    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()

    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)

    # --- sklearn-style report (matches XGBoost/CatBoost terminal look) ---
    label_order = [0, 1, 2]
    target_names = ["dropout", "enrolled", "graduate"]
    print("PLM-Encoder Metrics (3-class, sklearn):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=label_order,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=label_order)

    # --- Confusion Matrix FIGURE (matches screenshot style) ---
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix (3-class)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    print()

    metrics = compute_metrics_multiclass(y_test, y_pred, probs)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Saved model to: {model_dir}")

if __name__ == "__main__":
    main()
