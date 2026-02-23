import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils_data import (
    read_csv_auto,
    detect_target_column,
    map_labels,
    split_columns,
    fit_preprocess,
    transform,
    save_preprocess,
)
from model_ft import FTTransformerLike
from metrics_ft import compute_metrics, format_summary_row, format_confusion


class TabDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray):
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_num[idx], self.y[idx]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/data.csv")
    p.add_argument("--target_col", type=str, default="")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--d_token", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_unique_cat", type=int, default=50)
    p.add_argument("--drop_cols", type=str, default="id_student")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit_n", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--use_class_weights", type=int, default=1)
    p.add_argument("--balanced_sampler", type=int, default=1)
    p.add_argument("--class_weight_mode", type=str, default="sqrt_inv", choices=["none", "inv", "sqrt_inv"])
    p.add_argument("--class_weight_min", type=float, default=0.5)
    p.add_argument("--class_weight_max", type=float, default=3.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--save_best", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"[INFO] Using device: {device}")

    data_path = Path(args.data_path)
    df = read_csv_auto(data_path)
    print(f"[INFO] Columns: {list(df.columns)[:30]}")

    target_col = args.target_col.strip() or detect_target_column(df)
    print(f"[INFO] Using target_col: {target_col}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in columns: {list(df.columns)}")

    # Drop rows with missing target (robust)
    t = df[target_col].astype(str).str.strip()
    missing_tokens = {"", "nan", "none", "null", "na"}
    mask_missing = t.str.lower().isin(missing_tokens)
    if mask_missing.any():
        n_drop = int(mask_missing.sum())
        df = df.loc[~mask_missing].reset_index(drop=True)
        print(f"[WARN] Dropped {n_drop} rows with missing Target.")

    y = map_labels(df[target_col])
    print("[INFO] Label mapping used: Distinction(0), Fail(1), Pass(2), Withdrawn(3)")
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    print(f"[INFO] Class distribution after mapping: {dist}")
    X = df.drop(columns=[target_col])
    drop_cols = [c.strip() for c in str(args.drop_cols).split(",") if c.strip()]
    dropped_cols = [c for c in drop_cols if c in X.columns]
    if dropped_cols:
        X = X.drop(columns=dropped_cols)
    print(f"[INFO] Dropped columns: {dropped_cols}")

    # Optional small subset for quick tests (random sample)
    if args.overfit_n and args.overfit_n > 0:
        n = min(int(args.overfit_n), len(df))
        idx = rng.permutation(len(df))[:n]
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        print(f"[INFO] Overfit mode: n={n}")

    cat_cols, num_cols = split_columns(X, max_unique_for_cat=int(args.max_unique_cat))
    print(f"[INFO] n_rows={len(X)} | cat_cols={len(cat_cols)} | num_cols={len(num_cols)}")

    # Train/val/test split (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )

    prep = fit_preprocess(X_train, cat_cols, num_cols)
    xcat_train, xnum_train = transform(X_train, prep)
    xcat_val, xnum_val = transform(X_val, prep)
    xcat_test, xnum_test = transform(X_test, prep)

    # Embedding cardinalities (UNK=0 plus observed)
    cat_cards = [len(prep["cat_maps"][c]) + 1 for c in prep["cat_cols"]]

    model = FTTransformerLike(
        cat_cardinalities=cat_cards,
        n_num=len(prep["num_cols"]),
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_classes=4,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=3
        )
    else:
        scheduler = None

    if args.use_class_weights == 1 and args.class_weight_mode != "none":
        class_counts = np.bincount(y_train, minlength=4)
        base_w = 1.0 / (class_counts + 1e-8)
        if args.class_weight_mode == "inv":
            w = base_w
        else:
            w = np.sqrt(base_w)
        w = w / w.mean()
        w = np.clip(w, float(args.class_weight_min), float(args.class_weight_max))
        print(f"[INFO] Class counts: {class_counts.tolist()}")
        print(f"[INFO] Class weights (mode={args.class_weight_mode}): {w.tolist()}")
        weights_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        if args.label_smoothing > 0:
            loss_fn = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=args.label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        if args.label_smoothing > 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss()

    train_ds = TabDataset(xcat_train, xnum_train, y_train)
    if args.balanced_sampler == 1:
        class_counts = np.bincount(y_train, minlength=4)
        w_class = 1.0 / (class_counts + 1e-8)
        sample_w = w_class[y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
    val_loader = DataLoader(
        TabDataset(xcat_val, xnum_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TabDataset(xcat_test, xnum_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    def run_eval(loader):
        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for xc, xn, yy in loader:
                xc = torch.as_tensor(xc, device=device, dtype=torch.long)
                xn = torch.as_tensor(xn, device=device, dtype=torch.float32)
                logits = model(xc, xn)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probs.append(p)
                ys.append(np.array(yy))
        y_true = np.concatenate(ys)
        y_proba = np.concatenate(probs)
        return y_true, y_proba

    best_val = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    best_out_dir = Path("outputs") / "model"
    best_out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for xc, xn, yy in train_loader:
            xc = torch.as_tensor(xc, device=device, dtype=torch.long)
            xn = torch.as_tensor(xn, device=device, dtype=torch.float32)
            yy = torch.as_tensor(yy, device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            logits = model(xc, xn)
            loss = loss_fn(logits, yy)
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            opt.step()

            total_loss += float(loss.item()) * len(yy)
            n += len(yy)

        yv_true, yv_proba = run_eval(val_loader)
        val_metrics = compute_metrics(yv_true, yv_proba)
        val_f1 = val_metrics["f1_macro"]
        print(f"[EPOCH {epoch:02d}] loss={total_loss/max(1,n):.4f} | val_f1_macro={val_f1:.4f}")
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_f1)
            else:
                scheduler.step()

        if val_f1 > best_val + float(args.min_delta):
            best_val = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            if args.save_best == 1:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": {
                            "d_token": args.d_token,
                            "n_heads": args.n_heads,
                            "n_layers": args.n_layers,
                            "dropout": args.dropout,
                            "cat_cards": cat_cards,
                            "n_num": len(prep["num_cols"]),
                        },
                        "best_val_f1_macro": float(best_val),
                        "best_epoch": int(best_epoch),
                        "seed": int(args.seed),
                    },
                    best_out_dir / "best_model.pt",
                )
        else:
            patience_counter += 1
            if patience_counter >= int(args.patience):
                print(
                    f"[EARLY STOP] epoch={epoch} | best_epoch={best_epoch} | "
                    f"best_val_f1_macro={best_val:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yt_true, yt_proba = run_eval(test_loader)
    test_metrics = compute_metrics(yt_true, yt_proba)
    y_pred = yt_proba.argmax(axis=1)
    target_names = ["Distinction", "Fail", "Pass", "Withdrawn"]

    print("\n=== Classification Report (sklearn) ===")
    print(classification_report(yt_true, y_pred, labels=[0, 1, 2, 3], target_names=target_names, digits=4))
    report_dict_test = classification_report(
        yt_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=target_names,
        digits=4,
        output_dict=True,
    )
    recall_line = ", ".join([f"{lab}={report_dict_test[lab]['recall']:.4f}" for lab in target_names])
    precision_line = ", ".join([f"{lab}={report_dict_test[lab]['precision']:.4f}" for lab in target_names])
    print(f"[INFO] Per-class recall: {recall_line}")
    print(f"[INFO] Per-class precision: {precision_line}")

    cm = confusion_matrix(yt_true, y_pred, labels=[0, 1, 2, 3])

    print("=== Confusion Matrix (raw) ===")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - FT Transformer")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    # === Save metrics.json ===
    metrics_dir = Path("outputs") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    accuracy = float(accuracy_score(yt_true, y_pred))
    f1_macro = float(f1_score(yt_true, y_pred, average="macro"))
    recall_macro = float(recall_score(yt_true, y_pred, average="macro"))

    report_str = classification_report(
        yt_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=target_names,
        digits=4
    )

    metrics_json = {
        "model_name": "FT-Transformer",
        "num_features": int(X.shape[1]),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str
    }

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    print(f"[INFO] Saved metrics to: {metrics_dir / 'metrics.json'}")


    out_dir = Path("outputs") / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "d_token": args.d_token,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "cat_cards": cat_cards,
                "n_num": len(prep["num_cols"]),
            },
        },
        out_dir / "model.pt",
    )

    save_preprocess(prep, out_dir / "preprocess.json")

    meta = {"target_col": target_col}
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved model to: {out_dir / 'model.pt'}")
    print(f"[INFO] Saved preprocess to: {out_dir / 'preprocess.json'}")


if __name__ == "__main__":
    main()
