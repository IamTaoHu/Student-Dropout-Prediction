"""LIME explainability helpers (optional module)."""

from pathlib import Path
from typing import Callable

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def preprocess_like_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with CatBoost-style categorical preprocessing applied."""
    processed = X.copy()
    cat_cols = processed.select_dtypes(exclude=["number"]).columns
    for c in cat_cols:
        processed[c] = processed[c].astype(str).fillna("NA")
    return processed


def compute_lime_local_html(
    X_train: pd.DataFrame,
    X_row: pd.DataFrame,
    predict_proba_fn: Callable,
    outdir: Path,
    num_features: int = 8,
) -> Path:
    """Generate a LIME local explanation for a single row and save as HTML.

    Args:
        X_train: Feature dataframe used to initialize the LIME explainer.
        X_row: Single-row dataframe to explain (must contain exactly one row).
        predict_proba_fn: Callable that maps an array-like of rows to class probabilities.
        outdir: Directory where the HTML explanation will be written.
        num_features: Number of top features to include in the explanation.
    """
    if len(X_row) != 1:
        raise ValueError("X_row must contain exactly one row.")

    train_processed = preprocess_like_catboost(X_train)
    row_processed = preprocess_like_catboost(X_row)

    explainer = LimeTabularExplainer(
        train_processed.values,
        feature_names=train_processed.columns.tolist(),
        mode="classification",
    )

    explanation = explainer.explain_instance(
        row_processed.iloc[0].values,
        predict_proba_fn,
        num_features=num_features,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "lime_local.html"
    explanation.save_to_file(str(output_path))
    return output_path
