from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging
import os

from metrics import ID2LABEL, map_labels, compute_metrics_multiclass
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
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def resolve_column_name(df: pd.DataFrame, name: str):
    def _norm(value: str) -> str:
        return str(value).replace("\ufeff", "").strip().lower()

    if name in df.columns:
        return name

    mapping = {_norm(c): c for c in df.columns}
    return mapping.get(_norm(name))


def _read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) >= 10:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "input" / "data.csv"
    default_output = base_dir / "outputs" / "predictions.csv"
    model_dir = base_dir / "outputs" / "model"

    parser = argparse.ArgumentParser(description="Predict dropout using PLM encoder (3-class).")
    parser.add_argument("--input_csv", type=str, default=str(default_input))
    parser.add_argument("--output_csv", type=str, default=str(default_output))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-col", type=str, default=None, help="Optional: column name for y_true metrics")
    parser.add_argument("--quiet", action="store_true", help="Suppress all console output")
    args = parser.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.quiet:
        hf_logging.set_verbosity_error()
    else:
        hf_logging.set_verbosity_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] Using device: {device}")

    df = _read_csv_auto(Path(args.input_csv))

    if args.target_col is not None:
        target_col = resolve_column_name(df, args.target_col)
    else:
        target_col = detect_target_column(df)

    if target_col is None:
        log("[WARN] Target column not found. Will run prediction only (no metrics).")
    spec_path = model_dir / "textify_spec.json"
    textify_spec = None
    if spec_path.exists():
        try:
            import json
            with spec_path.open("r", encoding="utf-8") as f:
                textify_spec = json.load(f)
            log(f"[INFO] Loaded textify spec from: {spec_path}")
        except Exception as exc:
            log(f"[WARN] Could not load textify spec: {exc}. Falling back to on-the-fly spec.")
    texts = df_to_texts(df, target_col, spec=textify_spec)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    log("[INFO] Loaded model from outputs/model")

    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    y_pred_class = probs.argmax(axis=1)
    pred_label = [ID2LABEL[int(c)] for c in y_pred_class]

    y_true = None
    if target_col is not None:
        try:
            y_true = map_labels(df[target_col])
            _ = compute_metrics_multiclass(y_true, y_pred_class, probs)
        except Exception as exc:
            print(f"[WARN] Could not compute metrics: {exc}")

    out_df = pd.DataFrame(
        {
            "proba_dropout": probs[:, 0],
            "proba_enrolled": probs[:, 1],
            "proba_graduate": probs[:, 2],
            "pred_class_id": y_pred_class,
            "pred_label": pred_label,
        }
    )
    if y_true is not None:
        out_df.insert(0, "y_true", y_true)

    # Print preview table like XGBoost (top 10)
    preview_cols = ["proba_dropout", "proba_enrolled", "proba_graduate", "pred_class_id", "pred_label"]
    log(out_df[preview_cols].head(10).to_string(index=False))
    log("")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    log(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
