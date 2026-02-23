import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils_data import read_csv_auto, detect_target_column, map_labels, load_preprocess, transform
from model_ft import FTTransformerLike


LABEL2ID = {"Distinction": 0, "Fail": 1, "Pass": 2, "Withdrawn": 3}
ID2LABEL = {0: "distinction", 1: "fail", 2: "pass", 3: "withdrawn"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_preview_table(rows, headers):
    # fixed-width-ish preview (no external deps)
    widths = []
    for j, h in enumerate(headers):
        max_len = len(h)
        for r in rows:
            max_len = max(max_len, len(str(r[j])))
        widths.append(max_len + 2)

    # header
    line = "".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    # rows
    for r in rows:
        print("".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, default="data/data.csv")
    p.add_argument("--model_dir", type=str, default="outputs/model")
    p.add_argument("--output_csv", type=str, default="outputs/predictions/predictions.csv")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--show_rows", type=int, default=10)
    args = p.parse_args()

    device = get_device()

    model_dir = Path(args.model_dir)
    bundle = torch.load(model_dir / "model.pt", map_location="cpu")
    cfg = bundle["config"]
    prep = load_preprocess(model_dir / "preprocess.json")
    expected_n_classes = 4
    if isinstance(cfg, dict) and "n_classes" in cfg:
        n_classes_ckpt = int(cfg["n_classes"])
    else:
        state_dict = bundle.get("state_dict", {})
        head_w = state_dict.get("head.weight")
        if head_w is None or getattr(head_w, "ndim", None) != 2:
            raise ValueError(
                "Could not infer checkpoint output classes. Missing/invalid 'head.weight' in state_dict."
            )
        n_classes_ckpt = int(head_w.shape[0])
    if n_classes_ckpt != expected_n_classes:
        raise ValueError(
            f"Checkpoint class mismatch: expected {expected_n_classes} classes, got {n_classes_ckpt}. "
            "Please use a 4-class checkpoint."
        )

    model = FTTransformerLike(
        cat_cardinalities=cfg["cat_cards"],
        n_num=cfg["n_num"],
        d_token=cfg["d_token"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        n_classes=3,
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    # Read input
    df = read_csv_auto(Path(args.input_csv))

    # If target exists, show mapping + distribution, then drop it for inference
    print(f"Label mapping: {LABEL2ID}")

    target_col = None
    try:
        target_col = detect_target_column(df)
    except Exception:
        target_col = None

    y = None
    if target_col is not None and target_col in df.columns:
        try:
            y = map_labels(df[target_col])
            # Print distribution like your other projects
            vals, counts = np.unique(y, return_counts=True)
            dist = {np.int64(v): np.int64(c) for v, c in zip(vals, counts)}
            print(f"Class distribution after mapping: {dist}")
        except Exception:
            # if mapping fails, just ignore
            y = None

        df = df.drop(columns=[target_col])

    print(f"Device used : {device}")

    # Transform
    xcat, xnum = transform(df, prep)

    # Predict
    proba_parts = []
    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            xc = torch.as_tensor(xcat[i:i+args.batch_size], device=device, dtype=torch.long)
            xn = torch.as_tensor(xnum[i:i+args.batch_size], device=device, dtype=torch.float32)
            logits = model(xc, xn)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            proba_parts.append(probs)

    proba = np.vstack(proba_parts)
    pred_class_id = proba.argmax(axis=1)
    pred_label = [ID2LABEL[int(i)] for i in pred_class_id]

    # Build output df
    out = pd.DataFrame({
        "proba_distinction": proba[:, 0],
        "proba_fail": proba[:, 1],
        "proba_pass": proba[:, 2],
        "proba_withdrawn": proba[:, 3],
        "pred_class_id": pred_class_id,
        "pred_label": pred_label,
    })

    # Print preview (first N rows)
    n_show = min(int(args.show_rows), len(out))
    preview = out.head(n_show)

    rows = []
    for _, r in preview.iterrows():
        rows.append([
            f"{r['proba_distinction']:.6f}",
            f"{r['proba_fail']:.6f}",
            f"{r['proba_pass']:.6f}",
            f"{r['proba_withdrawn']:.6f}",
            str(int(r["pred_class_id"])),
            str(r["pred_label"]),
        ])

    headers = ["proba_distinction", "proba_fail", "proba_pass", "proba_withdrawn", "pred_class_id", "pred_label"]
    _print_preview_table(rows, headers)

    # Save
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved predictions to: {out_path.resolve()}")
    

if __name__ == "__main__":
    main()
