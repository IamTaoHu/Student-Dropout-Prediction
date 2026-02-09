from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src import config
from src.utils import save_json

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
TARGET_MAPPING = {"dropout": 1, "enrolled": 0, "graduate": 0}

G3_KEYWORDS = ["Debtor", "Tuition fees up to date", "Scholarship holder"]
G4_KEYWORDS = ["1st sem", "2nd sem", "Curricular units"]


def _detect_target_column(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _map_target(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(TARGET_MAPPING)


def _keyword_filter(features: pd.Index, keywords: list[str]) -> list[str]:
    lowered = [feature.lower() for feature in features]
    patterns = [keyword.lower() for keyword in keywords]
    return [
        features[idx]
        for idx, name in enumerate(lowered)
        if any(pattern in name for pattern in patterns)
    ]


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    y = _map_target(df[target_col])
    X = df.drop(columns=[target_col])

    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("NA")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=getattr(config, "TEST_SIZE", 0.2),
        random_state=42,
        stratify=y,
    )

    feature_names = X_train.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_cols]

    model_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": 200,
        "allow_writing_files": False,
        "auto_class_weights": "Balanced",
    }

    model = CatBoostClassifier(**model_params)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    pool = Pool(X_test, cat_features=cat_feature_indices)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    expected_value = shap_values[:, -1]
    feature_shap = shap_values[:, :-1]

    mean_abs_shap = pd.Series(
        data=abs(feature_shap).mean(axis=0),
        index=X_test.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    g3_features = _keyword_filter(X_test.columns, G3_KEYWORDS)
    g4_features = _keyword_filter(X_test.columns, G4_KEYWORDS)

    top_overall = mean_abs_shap.head(20)
    top_g3 = mean_abs_shap.loc[g3_features].sort_values(ascending=False).head(10)
    top_g4 = mean_abs_shap.loc[g4_features].sort_values(ascending=False).head(10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_DIR) / "shap_full" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(feature_shap, X_test, show=False, max_display=20)
    summary_path = output_dir / "shap_summary_full.png"
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    top_overall_path = output_dir / "top_shap_overall.csv"
    top_g3_path = output_dir / "top_shap_G3.csv"
    top_g4_path = output_dir / "top_shap_G4.csv"

    top_overall.to_frame().to_csv(top_overall_path, index_label="feature")
    top_g3.to_frame().to_csv(top_g3_path, index_label="feature")
    top_g4.to_frame().to_csv(top_g4_path, index_label="feature")

    metadata = {
        "n_samples_test": int(len(X_test)),
        "n_features": int(X_test.shape[1]),
        "cat_feature_count": int(len(cat_feature_indices)),
        "model_params": model_params,
        "expected_value_mean": float(expected_value.mean()) if len(expected_value) else 0.0,
    }
    metadata_path = output_dir / "shap_metadata.json"
    save_json(metadata, metadata_path)

    print(f"Saved summary plot: {summary_path}")
    print(f"Saved top overall CSV: {top_overall_path}")
    print(f"Saved top G3 CSV: {top_g3_path}")
    print(f"Saved top G4 CSV: {top_g4_path}")
    print(f"Saved metadata JSON: {metadata_path}")
    print("Top 10 overall features by mean(|SHAP|):")
    print(top_overall.head(10).to_string())


if __name__ == "__main__":
    main()
