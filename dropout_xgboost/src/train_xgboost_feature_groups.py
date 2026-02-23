from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


# -----------------------
# Minimal local config
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (PROJECT_ROOT / "data" / "data.csv").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
RANDOM_STATE = 42
TEST_SIZE = 0.2


def save_json(obj: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


TARGET_CANDIDATES = [
    "Target", "target",
    "STATUS", "Status", "status",
    "Outcome", "outcome",
    "Class", "class",
]
TARGET_MAPPING = {"dropout": 0, "enrolled": 1, "graduate": 2}

GROUP_PATTERNS = {
    "G1_demographic": [
        "Gender",
        "Age at enrollment",
        "Nationality",
        "Marital status",
        "Displaced",
        "Educational special needs",
    ],
    "G2_admission_program": [
        "Course",
        "Application mode",
        "Application order",
        "Previous qualification",
        "Admission grade",
        "Daytime/evening attendance",
    ],
    "G3_financial_admin": [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ],
    "G4_academic_semesters": [
        "1st sem",
        "2nd sem",
        "Curricular units",
    ],
}

EXPERIMENTS = [
    ("A_G1", ["G1_demographic"]),
    ("B_G1_G2", ["G1_demographic", "G2_admission_program"]),
    ("C_G1_G2_G3", ["G1_demographic", "G2_admission_program", "G3_financial_admin"]),
    ("D_Full", ["G1_demographic", "G2_admission_program", "G3_financial_admin", "G4_academic_semesters"]),
    ("Full", ["G1_demographic", "G2_admission_program", "G3_financial_admin", "G4_academic_semesters"]),
    ("w/o_G1", ["G2_admission_program", "G3_financial_admin", "G4_academic_semesters"]),
    ("w/o_G2", ["G1_demographic", "G3_financial_admin", "G4_academic_semesters"]),
    ("w/o_G3", ["G1_demographic", "G2_admission_program", "G4_academic_semesters"]),
    ("w/o_G4", ["G1_demographic", "G2_admission_program", "G3_financial_admin"]),
]


def _detect_target_column(columns: pd.Index) -> str:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"Missing target column. Available columns: {columns.tolist()}")


def _map_target(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized.dropna().unique()) - set(TARGET_MAPPING))
    if unexpected:
        raise ValueError(f"Unexpected target labels in {column_name}: {unexpected}")
    return normalized.map(TARGET_MAPPING)


def _build_group_columns(columns: pd.Index) -> dict[str, list[str]]:
    group_columns: dict[str, list[str]] = {}
    lowered = [col.lower() for col in columns]
    for group_name, patterns in GROUP_PATTERNS.items():
        patt = [p.lower() for p in patterns]
        matches = [
            columns[i]
            for i, col_lower in enumerate(lowered)
            if any(p in col_lower for p in patt)
        ]
        group_columns[group_name] = matches
    return group_columns


def _select_features(group_columns: dict[str, list[str]], groups: list[str], exp: str) -> list[str]:
    selected: list[str] = []
    for g in groups:
        cols = group_columns.get(g, [])
        if not cols:
            print(f"Warning: {g} has 0 columns; skipping.")
            continue
        for c in cols:
            if c not in selected:
                selected.append(c)

    if not selected:
        raise ValueError(
            f"Feature set is empty for experiment '{exp}'. "
            "Check GROUP_PATTERNS vs your column names."
        )
    return selected


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def _run_experiment(X: pd.DataFrame, y: pd.Series, train_idx: pd.Index, test_idx: pd.Index, feat_cols: list[str]) -> dict:
    X_sel = X[feat_cols].copy()

    X_train = X_sel.loc[train_idx]
    X_test = X_sel.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # internal val split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    pre = _make_preprocessor(X_tr)
    X_tr_t = pre.fit_transform(X_tr)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    num_features_after = int(X_tr_t.shape[1])

    # NOTE:
    # Newer XGBoost sklearn interface removed early_stopping_rounds from .fit().
    # Use the same-name parameter in the constructor instead. :contentReference[oaicite:1]{index=1}
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=200,
    )

    clf.fit(
        X_tr_t,
        y_tr,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )

    proba = clf.predict_proba(X_test_t)
    pred = proba.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()
    report = classification_report(
        y_test,
        pred,
        target_names=["dropout", "enrolled", "graduate"],
        digits=4,
        output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "num_features": num_features_after,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    target_col = _detect_target_column(df.columns)
    y = _map_target(df[target_col], target_col)
    X = df.drop(columns=[target_col])

    valid = y.notna()
    y = y[valid]
    X = X.loc[valid]

    group_cols = _build_group_columns(X.columns)

    train_idx, test_idx = train_test_split(
        X.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: list[dict] = []
    for exp, groups in EXPERIMENTS:
        feat_cols = _select_features(group_cols, groups, exp)
        metrics = _run_experiment(X, y, train_idx, test_idx, feat_cols)
        results.append(
            {
                "experiment": exp,
                "num_features": metrics["num_features"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "groups": groups,
                "features": feat_cols,
                "confusion_matrix": metrics["confusion_matrix"],
                "classification_report": metrics["classification_report"],
            }
        )

    results_df = pd.DataFrame(results)
    summary_df = results_df[["experiment", "num_features", "accuracy", "macro_f1", "weighted_f1"]].copy()
    summary_df[["accuracy", "macro_f1", "weighted_f1"]] = summary_df[["accuracy", "macro_f1", "weighted_f1"]].astype(float).round(4)

    print("\nXGBoost Feature Group Performance Summary")
    print(summary_df.to_string(index=False))
    print("\nMarkdown table:\n")
    try:
        print(summary_df.to_markdown(index=False))
    except Exception:
        print(summary_df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT_DIR / "feature_groups" / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(outdir / "summary.csv", index=False)
    save_json(summary_df.to_dict(orient="records"), outdir / "summary.json")
    save_json(results, outdir / "details.json")


if __name__ == "__main__":
    main()
