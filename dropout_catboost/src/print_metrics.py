import json
import sys
from pathlib import Path

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "catboost"

    if model == "catboost":
        metrics_path = Path("outputs/catboost_metrics.json")
    elif model == "lr":
        metrics_path = Path("../dropout_lr_baseline/outputs/metrics.json")
    else:
        raise ValueError("model must be 'catboost' or 'lr'")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"=== {model.upper()} Metrics ===")
    print(f"F1      : {metrics['f1']:.4f}")
    print(f"Recall  : {metrics['recall']:.4f}")
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
