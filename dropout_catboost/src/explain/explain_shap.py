"""SHAP explainability helpers (skeleton only; no SHAP logic yet)."""

from pathlib import Path

import pandas as pd
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

from src import config

TARGET_CANDIDATES = [
    "Target",
    "target",
    "STATUS",
    "Status",
    "status",
    "Outcome",
    "outcome",
    "Class",
    "class",
]


def load_model(model_path: Path) -> CatBoostClassifier:
    """Load a CatBoost model from the given path."""
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def ensure_outdir(path: Path) -> Path:
    """Ensure the output directory exists and return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_input_df(X: pd.DataFrame) -> None:
    """Validate the input dataframe for explainability computations."""
    if X is None or X.empty:
        raise ValueError("Input dataframe X must be non-empty.")
    if X.isnull().any().any():
        raise ValueError("Input dataframe X contains missing values.")


def preprocess_like_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features like the CatBoost training pipeline."""
    processed = X.copy()

    cat_cols = processed.select_dtypes(exclude=["number"]).columns
    target_cols = [c for c in TARGET_CANDIDATES if c in processed.columns]
    cat_cols = [c for c in cat_cols if c not in target_cols]

    for c in cat_cols:
        processed[c] = processed[c].astype(str).fillna("NA")

    return processed


def compute_global_shap_bar(
    X: pd.DataFrame,
    outdir: Path,
    max_samples: int = 1000,
) -> Path:
    """Compute and save a global SHAP bar plot for feature importance."""
    validate_input_df(X)
    processed = preprocess_like_catboost(X)

    if len(processed) > max_samples:
        processed = processed.sample(n=max_samples, random_state=config.RANDOM_STATE)

    model_path = Path(config.OUTPUT_DIR) / "catboost_model.cbm"
    model = load_model(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed)

    outdir = ensure_outdir(outdir)
    output_path = outdir / "shap_global_bar.png"
    shap.summary_plot(shap_values, processed, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def compute_local_shap_waterfall(
    X_row: pd.DataFrame,
    outdir: Path,
) -> Path:
    """Compute and save a local SHAP waterfall plot for a single row."""
    if len(X_row) != 1:
        raise ValueError("X_row must contain exactly one row.")

    processed = preprocess_like_catboost(X_row)

    model_path = Path(config.OUTPUT_DIR) / "catboost_model.cbm"
    model = load_model(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed)

    outdir = ensure_outdir(outdir)
    output_path = outdir / "shap_local_waterfall.png"
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
