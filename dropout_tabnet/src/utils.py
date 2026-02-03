import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import joblib

LABEL_MAPPING = {
    "Dropout": 0,
    "Graduate": 1,
    "Enrolled": 2,
}
ID2LABEL = {v: k for k, v in LABEL_MAPPING.items()}


@dataclass
class SplitData:
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df

def infer_target_column(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> str:
    if candidates is None:
        candidates = [
            "Target",
            "target",
            "Status",
            "status",
            "label",
            "dropout",
            "Outcome",
            "outcome",
        ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Target column not found in dataframe.")


def _map_target_to_int(y: pd.Series) -> np.ndarray:
    if y.dtype == object or pd.api.types.is_string_dtype(y.dtype):
        normalized = y.astype(str).str.strip().str.lower()
        if not hasattr(_map_target_to_int, "_printed_mapping"):
            print("Label mapping:", LABEL_MAPPING)
            _map_target_to_int._printed_mapping = True
        mapping = {k.lower(): v for k, v in LABEL_MAPPING.items()}
        mapped = normalized.map(mapping)
        if mapped.isna().any():
            unknown_values = sorted(set(normalized[mapped.isna()].tolist()))
            raise ValueError(f"Unknown target values: {unknown_values}")
        return mapped.astype(int).to_numpy()
    return y.to_numpy(dtype=int)


def preprocess_tabular(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    y = _map_target_to_int(y_raw)

    num_cols = X_raw.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X_raw.columns if c not in num_cols]

    X_filled = X_raw.copy()
    if len(num_cols) > 0:
        for col in num_cols:
            median_val = X_filled[col].median()
            X_filled[col] = X_filled[col].fillna(median_val)
    if len(cat_cols) > 0:
        for col in cat_cols:
            X_filled[col] = X_filled[col].fillna("missing")

    X_encoded = pd.get_dummies(X_filled, columns=cat_cols, drop_first=False)
    feature_names = list(X_encoded.columns)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    valid_ratio = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_ratio,
        random_state=random_state,
        stratify=y_train_valid,
    )

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return SplitData(
        X_train=X_train_scaled,
        X_valid=X_valid_scaled,
        X_test=X_test_scaled,
        y_train=np.asarray(y_train, dtype=int),
        y_valid=np.asarray(y_valid, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=feature_names,
        scaler=scaler,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_pred = np.argmax(y_prob, axis=1).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            roc_auc = None
        try:
            y_true_ovr = np.eye(y_prob.shape[1])[y_true]  # one-hot
            pr_auc = average_precision_score(y_true_ovr, y_prob, average="macro")
        except Exception:
            pr_auc = None
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        report = classification_report(y_true, y_pred, zero_division=0)
        return {
            "accuracy": acc,
            "f1": f1,
            "recall": rec,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm.tolist(),
            "TN": None,
            "FP": None,
            "FN": None,
            "TP": None,
            "classification_report": report,
        }

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1,
        "recall": rec,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "classification_report": report,
    }


def save_artifacts(model_dir: str, model, scaler: StandardScaler, meta: dict) -> Dict[str, str]:
    ensure_dir(model_dir)
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    meta_path = os.path.join(model_dir, "meta.json")

    joblib.dump(scaler, scaler_path)
    save_json(meta_path, meta)

    return {
        "scaler": scaler_path,
        "meta": meta_path,
    }


def load_artifacts(model_dir: str) -> Tuple[StandardScaler, dict]:
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    meta_path = os.path.join(model_dir, "meta.json")

    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return scaler, meta
