from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


SEEDS = [42, 43, 44, 45, 46]


def _default_experiment_configs(project_root: Path) -> dict[str, dict]:
    return {
        "baseline": {
            "family": "flat_xgb",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": ["--preset", "baseline_safe"],
        },
        "enhanced_features": {
            "family": "flat_xgb",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": ["--preset", "macro_f1_safe"],
        },
        "enhanced_features_with_calibration": {
            "family": "flat_xgb",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": ["--preset", "macro_f1_safe", "--calibration", "sigmoid"],
        },
        "flat_feature_phase3": {
            "family": "flat_xgb_phase3",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase3",
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase4": {
            "family": "flat_xgb_phase4",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase4",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase5": {
            "family": "flat_xgb_phase5",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase5",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase6": {
            "family": "flat_xgb_phase6",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase10": {
            "family": "flat_xgb_phase10",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase10",
                "--n_estimators",
                "900",
                "--learning_rate",
                "0.04",
                "--max_depth",
                "7",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_phase7_class_thresholds": {
            "family": "flat_xgb_phase7_thresholds",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
        },
        "flat_feature_phase5_ablate_module": {
            "family": "flat_xgb_phase5_ablation",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase5_no_module_features",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase5_ablate_tail": {
            "family": "flat_xgb_phase5_ablation",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase5_no_tail_features",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase5_ablate_late": {
            "family": "flat_xgb_phase5_ablation",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase5_no_late_features",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase5_ablate_consistency": {
            "family": "flat_xgb_phase5_ablation",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase5_no_consistency_features",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "flat_feature_phase5_candidate": {
            "family": "flat_xgb_phase5_candidate",
            "train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "output_root": "flat_xgb",
            "args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase4",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "active": False,
            "notes": "Phase5 candidate placeholder only; do not run as production benchmark yet.",
        },
        "ova_specialists": {
            "family": "ova_xgb",
            "train_script": str((project_root / "src" / "models" / "ova_xgb" / "train_ova.py").resolve()),
            "output_root": "ova_xgb",
            "args": ["--preset", "macro_f1_safe"],
        },
        "ova_tuned_small": {
            "family": "ova_xgb",
            "train_script": str((project_root / "src" / "models" / "ova_xgb" / "train_ova.py").resolve()),
            "output_root": "ova_xgb",
            "args": [
                "--preset",
                "macro_f1_safe",
                "--n_estimators",
                "300",
                "--max_depth",
                "6",
                "--learning_rate",
                "0.05",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "ova_tuned_large": {
            "family": "ova_xgb",
            "train_script": str((project_root / "src" / "models" / "ova_xgb" / "train_ova.py").resolve()),
            "output_root": "ova_xgb",
            "args": [
                "--preset",
                "macro_f1_safe",
                "--n_estimators",
                "600",
                "--max_depth",
                "7",
                "--learning_rate",
                "0.03",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
        },
        "pairwise_refinement": {
            "family": "pairwise_xgb",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "pairwise_train_script": str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
            "pairwise_predict_script": str((project_root / "src" / "models" / "pairwise_xgb" / "predict_pairwise.py").resolve()),
            "output_root": "pairwise_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "pairwise_args": ["--preset", "macro_f1_safe"],
            "predict_args": [],
        },
        "distinction_refinement": {
            "family": "distinction_refined_xgb",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "predict_args": ["--distinction_override_threshold", "0.6"],
        },
        "flat_phase8_distinction_refinement": {
            "family": "flat_xgb_phase8_distinction_refinement",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "predict_args": ["--distinction_threshold", "0.45"],
        },
        "flat_phase9_multi_specialist_refinement": {
            "family": "flat_xgb_phase9_multi_specialist_refinement",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "pairwise_train_script": str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
            "phase9_predict_script": str((project_root / "src" / "models" / "phase9_refinement" / "pipeline.py").resolve()),
            "output_root": "phase9_refinement",
            "flat_args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "pairwise_args": [
                "--preset",
                "macro_f1_safe",
                "--pairwise_n_estimators",
                "300",
                "--pairwise_learning_rate",
                "0.05",
                "--pairwise_max_depth",
                "5",
                "--pairwise_subsample",
                "0.8",
                "--pairwise_colsample_bytree",
                "0.8",
            ],
            "predict_args": [
                "--distinction_threshold",
                "0.45",
            ],
        },
        "flat_phase9_ablation_distinction_only": {
            "family": "flat_xgb_phase9_ablation",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "pairwise_train_script": str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
            "phase9_predict_script": str((project_root / "src" / "models" / "phase9_refinement" / "pipeline.py").resolve()),
            "output_root": "phase9_refinement",
            "flat_args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "pairwise_args": [
                "--preset",
                "macro_f1_safe",
                "--pairwise_n_estimators",
                "300",
                "--pairwise_learning_rate",
                "0.05",
                "--pairwise_max_depth",
                "5",
                "--pairwise_subsample",
                "0.8",
                "--pairwise_colsample_bytree",
                "0.8",
            ],
            "predict_args": [
                "--distinction_threshold",
                "0.45",
                "--phase9_disable_fail_specialist",
                "--phase9_disable_withdrawn_specialist",
            ],
        },
        "flat_phase9_ablation_distinction_fail": {
            "family": "flat_xgb_phase9_ablation",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "pairwise_train_script": str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
            "phase9_predict_script": str((project_root / "src" / "models" / "phase9_refinement" / "pipeline.py").resolve()),
            "output_root": "phase9_refinement",
            "flat_args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "pairwise_args": [
                "--preset",
                "macro_f1_safe",
                "--pairwise_n_estimators",
                "300",
                "--pairwise_learning_rate",
                "0.05",
                "--pairwise_max_depth",
                "5",
                "--pairwise_subsample",
                "0.8",
                "--pairwise_colsample_bytree",
                "0.8",
            ],
            "predict_args": [
                "--distinction_threshold",
                "0.45",
                "--phase9_disable_withdrawn_specialist",
            ],
        },
        "flat_phase9_ablation_distinction_withdrawn": {
            "family": "flat_xgb_phase9_ablation",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "pairwise_train_script": str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
            "phase9_predict_script": str((project_root / "src" / "models" / "phase9_refinement" / "pipeline.py").resolve()),
            "output_root": "phase9_refinement",
            "flat_args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "pairwise_args": [
                "--preset",
                "macro_f1_safe",
                "--pairwise_n_estimators",
                "300",
                "--pairwise_learning_rate",
                "0.05",
                "--pairwise_max_depth",
                "5",
                "--pairwise_subsample",
                "0.8",
                "--pairwise_colsample_bytree",
                "0.8",
            ],
            "predict_args": [
                "--distinction_threshold",
                "0.45",
                "--phase9_disable_fail_specialist",
            ],
        },
        "flat_phase9_margin_tuned": {
            "family": "flat_xgb_phase9_multi_specialist_refinement",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "pairwise_train_script": str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
            "phase9_predict_script": str((project_root / "src" / "models" / "phase9_refinement" / "pipeline.py").resolve()),
            "output_root": "phase9_refinement",
            "flat_args": [
                "--preset",
                "baseline_safe",
                "--feature_set",
                "phase6",
                "--n_estimators",
                "500",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
                "--optimize_class_thresholds",
            ],
            "distinction_args": [
                "--n_estimators",
                "400",
                "--learning_rate",
                "0.05",
                "--max_depth",
                "6",
                "--subsample",
                "0.8",
                "--colsample_bytree",
                "0.8",
            ],
            "pairwise_args": [
                "--preset",
                "macro_f1_safe",
                "--pairwise_n_estimators",
                "300",
                "--pairwise_learning_rate",
                "0.05",
                "--pairwise_max_depth",
                "5",
                "--pairwise_subsample",
                "0.8",
                "--pairwise_colsample_bytree",
                "0.8",
            ],
            "predict_args": [
                "--distinction_threshold",
                "0.45",
                "--phase9_margin_fail_pass",
                "0.20",
                "--phase9_margin_withdrawn_pass",
                "0.20",
                "--phase9_fail_threshold",
                "0.60",
                "--phase9_withdrawn_threshold",
                "0.60",
                "--phase9_flat_confidence_guard",
                "0.80",
            ],
        },
        "distinction_refinement_threshold_045": {
            "family": "distinction_threshold_sweep",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": ["--n_estimators", "400", "--learning_rate", "0.05", "--max_depth", "6", "--subsample", "0.8", "--colsample_bytree", "0.8"],
            "predict_args": ["--distinction_threshold", "0.45"],
        },
        "distinction_refinement_threshold_050": {
            "family": "distinction_threshold_sweep",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": ["--n_estimators", "400", "--learning_rate", "0.05", "--max_depth", "6", "--subsample", "0.8", "--colsample_bytree", "0.8"],
            "predict_args": ["--distinction_threshold", "0.50"],
        },
        "distinction_refinement_threshold_055": {
            "family": "distinction_threshold_sweep",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": ["--n_estimators", "400", "--learning_rate", "0.05", "--max_depth", "6", "--subsample", "0.8", "--colsample_bytree", "0.8"],
            "predict_args": ["--distinction_threshold", "0.55"],
        },
        "distinction_refinement_threshold_060": {
            "family": "distinction_threshold_sweep",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": ["--n_estimators", "400", "--learning_rate", "0.05", "--max_depth", "6", "--subsample", "0.8", "--colsample_bytree", "0.8"],
            "predict_args": ["--distinction_threshold", "0.60"],
        },
        "distinction_refinement_threshold_065": {
            "family": "distinction_threshold_sweep",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": ["--n_estimators", "400", "--learning_rate", "0.05", "--max_depth", "6", "--subsample", "0.8", "--colsample_bytree", "0.8"],
            "predict_args": ["--distinction_threshold", "0.65"],
        },
        "distinction_refinement_threshold_070": {
            "family": "distinction_threshold_sweep",
            "flat_train_script": str((project_root / "src" / "train_xgboost.py").resolve()),
            "distinction_train_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()
            ),
            "distinction_predict_script": str(
                (project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()
            ),
            "output_root": "distinction_refined_xgb",
            "flat_args": ["--preset", "macro_f1_safe"],
            "distinction_args": ["--n_estimators", "400", "--learning_rate", "0.05", "--max_depth", "6", "--subsample", "0.8", "--colsample_bytree", "0.8"],
            "predict_args": ["--distinction_threshold", "0.70"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Run flat and OVA XGBoost experiments and collect summary metrics.")
    p.add_argument(
        "--input",
        type=str,
        default=str((project_root / "data" / "data.csv").resolve()),
        help="Input CSV path passed to the selected training entrypoint.",
    )
    p.add_argument(
        "--train_script",
        type=str,
        default=str((project_root / "src" / "train_xgboost.py").resolve()),
        help="Optional override for the flat training entrypoint.",
    )
    p.add_argument(
        "--ova_train_script",
        type=str,
        default=str((project_root / "src" / "models" / "ova_xgb" / "train_ova.py").resolve()),
        help="Optional override for the OVA training entrypoint.",
    )
    p.add_argument(
        "--pairwise_train_script",
        type=str,
        default=str((project_root / "src" / "models" / "pairwise_xgb" / "train_pairwise.py").resolve()),
        help="Optional override for the pairwise training entrypoint.",
    )
    p.add_argument(
        "--pairwise_predict_script",
        type=str,
        default=str((project_root / "src" / "models" / "pairwise_xgb" / "predict_pairwise.py").resolve()),
        help="Optional override for the pairwise prediction entrypoint.",
    )
    p.add_argument(
        "--distinction_train_script",
        type=str,
        default=str((project_root / "src" / "models" / "distinction_specialist_xgb" / "train_distinction_specialist.py").resolve()),
        help="Optional override for the distinction specialist training entrypoint.",
    )
    p.add_argument(
        "--distinction_predict_script",
        type=str,
        default=str((project_root / "src" / "models" / "distinction_specialist_xgb" / "predict_distinction_specialist.py").resolve()),
        help="Optional override for the distinction specialist prediction entrypoint.",
    )
    p.add_argument(
        "--phase9_predict_script",
        type=str,
        default=str((project_root / "src" / "models" / "phase9_refinement" / "pipeline.py").resolve()),
        help="Optional override for the Phase9 multi-specialist prediction entrypoint.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str((project_root / "outputs" / "experiments").resolve()),
        help="Directory for experiment summary outputs.",
    )
    p.add_argument(
        "--run_prefix",
        type=str,
        default="xgb_exp",
        help="Prefix used for per-seed training run folders.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run training.",
    )
    p.add_argument(
        "--num_seeds",
        type=int,
        default=len(SEEDS),
        help="Number of seeds to run from seed_start.",
    )
    p.add_argument(
        "--seed_start",
        type=int,
        default=SEEDS[0],
        help="First seed used by the experiment runner.",
    )
    p.add_argument(
        "--extra_args",
        type=str,
        nargs=argparse.REMAINDER,
        default=[],
        help="Optional extra args forwarded to each selected train script.",
    )
    p.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=[
            "baseline",
            "enhanced_features",
            "enhanced_features_with_calibration",
            "ova_specialists",
            "ova_tuned_small",
            "ova_tuned_large",
            "pairwise_refinement",
            "distinction_refinement",
        ],
        help="Experiment config names to run.",
    )
    return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_final_metrics(metrics: dict) -> dict:
    return metrics.get("final_test_metrics") or metrics.get("aggregate_test_metrics") or {}


def save_confusion_matrix_csv(matrix: list[list[int]], labels: list[str], out_path: Path) -> Path:
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "true_label"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, encoding="utf-8")
    return out_path


def summarize_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "experiment" in df.columns and "experiment_name" not in df.columns:
        df["experiment_name"] = df["experiment"]
    ordered_cols = [
        "family",
        "experiment_name",
        "experiment",
        "seed",
        "selected_seed",
        "run_name",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "metrics_path",
        "run_dir",
        "confusion_matrix_csv",
        "confusion_matrix_plot",
    ]
    available = [col for col in ordered_cols if col in df.columns]
    remaining = [col for col in df.columns if col not in available]
    return df[available + remaining]


def print_summary(df: pd.DataFrame, family: str) -> None:
    if df.empty:
        print(f"No {family} experiment results were collected.")
        return
    print(f"\n{family} Experiment Summary")
    print(df[["experiment", "seed", "accuracy", "macro_f1", "balanced_accuracy"]].to_string(index=False))
    print("\nAggregate")
    print(df.groupby("experiment")[["accuracy", "macro_f1", "balanced_accuracy"]].agg(["mean", "std", "min", "max"]).round(6).to_string())
    best_row = df.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]
    print(
        "\nBest run"
        f"\nfamily={best_row['family']}"
        f"\nexperiment={best_row['experiment']}"
        f"\nseed={int(best_row['seed'])}"
        f" | accuracy={float(best_row['accuracy']):.6f}"
        f" | macro_f1={float(best_row['macro_f1']):.6f}"
        f" | balanced_accuracy={float(best_row['balanced_accuracy']):.6f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).expanduser().resolve()
    experiments_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    config_map = _default_experiment_configs(project_root)
    config_map["baseline"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["enhanced_features"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["enhanced_features_with_calibration"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase3"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase4"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase5"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase6"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase10"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_phase7_class_thresholds"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase5_ablate_module"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase5_ablate_tail"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase5_ablate_late"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase5_ablate_consistency"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_feature_phase5_candidate"]["train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["ova_specialists"]["train_script"] = str(Path(args.ova_train_script).expanduser().resolve())
    config_map["ova_tuned_small"]["train_script"] = str(Path(args.ova_train_script).expanduser().resolve())
    config_map["ova_tuned_large"]["train_script"] = str(Path(args.ova_train_script).expanduser().resolve())
    config_map["pairwise_refinement"]["pairwise_train_script"] = str(Path(args.pairwise_train_script).expanduser().resolve())
    config_map["pairwise_refinement"]["pairwise_predict_script"] = str(Path(args.pairwise_predict_script).expanduser().resolve())
    config_map["pairwise_refinement"]["flat_train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["distinction_refinement"]["distinction_train_script"] = str(
        Path(args.distinction_train_script).expanduser().resolve()
    )
    config_map["distinction_refinement"]["distinction_predict_script"] = str(
        Path(args.distinction_predict_script).expanduser().resolve()
    )
    config_map["distinction_refinement"]["flat_train_script"] = str(Path(args.train_script).expanduser().resolve())
    config_map["flat_phase8_distinction_refinement"]["distinction_train_script"] = str(
        Path(args.distinction_train_script).expanduser().resolve()
    )
    config_map["flat_phase8_distinction_refinement"]["distinction_predict_script"] = str(
        Path(args.distinction_predict_script).expanduser().resolve()
    )
    config_map["flat_phase8_distinction_refinement"]["flat_train_script"] = str(Path(args.train_script).expanduser().resolve())
    for phase9_exp in [
        "flat_phase9_multi_specialist_refinement",
        "flat_phase9_ablation_distinction_only",
        "flat_phase9_ablation_distinction_fail",
        "flat_phase9_ablation_distinction_withdrawn",
        "flat_phase9_margin_tuned",
    ]:
        config_map[phase9_exp]["distinction_train_script"] = str(Path(args.distinction_train_script).expanduser().resolve())
        config_map[phase9_exp]["pairwise_train_script"] = str(Path(args.pairwise_train_script).expanduser().resolve())
        config_map[phase9_exp]["phase9_predict_script"] = str(Path(args.phase9_predict_script).expanduser().resolve())
        config_map[phase9_exp]["flat_train_script"] = str(Path(args.train_script).expanduser().resolve())
    for threshold_exp in [
        "distinction_refinement_threshold_045",
        "distinction_refinement_threshold_050",
        "distinction_refinement_threshold_055",
        "distinction_refinement_threshold_060",
        "distinction_refinement_threshold_065",
        "distinction_refinement_threshold_070",
    ]:
        config_map[threshold_exp]["distinction_train_script"] = str(Path(args.distinction_train_script).expanduser().resolve())
        config_map[threshold_exp]["distinction_predict_script"] = str(Path(args.distinction_predict_script).expanduser().resolve())
        config_map[threshold_exp]["flat_train_script"] = str(Path(args.train_script).expanduser().resolve())

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    selected_experiments = [name for name in args.experiments if name in config_map]
    unknown_experiments = sorted(set(args.experiments) - set(selected_experiments))
    if unknown_experiments:
        raise ValueError(f"Unknown experiment configs: {unknown_experiments}")
    inactive_requested = [name for name in selected_experiments if not bool(config_map[name].get("active", True))]
    if inactive_requested:
        details = {name: config_map[name].get("notes", "inactive placeholder") for name in inactive_requested}
        raise ValueError(f"Inactive experiment configs requested: {details}")

    rows: list[dict] = []
    run_seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    for experiment_name in selected_experiments:
        experiment = config_map[experiment_name]
        family = str(experiment["family"])
        output_root = str(experiment["output_root"])
        confusion_dir = ensure_dir(experiments_dir / f"{family}_confusion_matrices")
        print(f"\n=== Experiment: {experiment_name} ({family}) ===")
        for seed in run_seeds:
            run_name = f"{args.run_prefix}_{experiment_name}_{seed}"
            run_dir = project_root / "outputs" / output_root / run_name
            metrics_path: Path
            confusion_plot_path: Path | None = None

            if family == "pairwise_xgb":
                flat_run_name = f"{run_name}_flat"
                pairwise_run_name = run_name
                flat_run_dir = project_root / "outputs" / "flat_xgb" / flat_run_name
                pairwise_run_dir = project_root / "outputs" / output_root / pairwise_run_name
                prediction_csv_path = pairwise_run_dir / "refined_predictions.csv"
                prediction_metrics_path = pairwise_run_dir / "refined_metrics.json"

                for path in [flat_run_dir, pairwise_run_dir]:
                    if path.exists():
                        shutil.rmtree(path)

                flat_train_script = Path(experiment["flat_train_script"]).expanduser().resolve()
                pairwise_train_script = Path(experiment["pairwise_train_script"]).expanduser().resolve()
                pairwise_predict_script = Path(experiment["pairwise_predict_script"]).expanduser().resolve()
                for script_path in [flat_train_script, pairwise_train_script, pairwise_predict_script]:
                    if not script_path.exists():
                        raise FileNotFoundError(f"Required script not found for {experiment_name}: {script_path}")

                flat_cmd = [
                    str(args.python),
                    str(flat_train_script),
                    "--input",
                    str(input_path),
                    "--num_seeds",
                    "1",
                    "--seed_start",
                    str(seed),
                    "--run_name",
                    flat_run_name,
                    "--no_plot",
                    *experiment["flat_args"],
                    *args.extra_args,
                ]
                pairwise_cmd = [
                    str(args.python),
                    str(pairwise_train_script),
                    "--input",
                    str(input_path),
                    "--pairwise_num_seeds",
                    "1",
                    "--pairwise_seed_start",
                    str(seed),
                    "--run_name",
                    pairwise_run_name,
                    "--no_plot",
                    *experiment["pairwise_args"],
                ]
                predict_cmd = [
                    str(args.python),
                    str(pairwise_predict_script),
                    "--input",
                    str(input_path),
                    "--flat_model",
                    str((flat_run_dir / "artifacts" / "xgboost_model.joblib").resolve()),
                    "--pairwise_model",
                    str((pairwise_run_dir / "artifacts" / "pairwise_models.joblib").resolve()),
                    "--output",
                    str(prediction_csv_path.resolve()),
                    "--metrics_output",
                    str(prediction_metrics_path.resolve()),
                    "--split_seed",
                    str(seed),
                    "--split_scope",
                    "test",
                    *experiment["predict_args"],
                ]
                print(f"\nRunning family={family} experiment={experiment_name} seed={seed}")
                for cmd in [flat_cmd, pairwise_cmd, predict_cmd]:
                    print("Command:", " ".join(cmd))
                    subprocess.run(cmd, cwd=project_root, check=True)
                run_dir = pairwise_run_dir
                metrics_path = prediction_metrics_path
            elif family in {"distinction_refined_xgb", "distinction_threshold_sweep", "flat_xgb_phase8_distinction_refinement"}:
                flat_run_name = f"{run_name}_flat"
                distinction_run_name = f"{run_name}_specialist"
                refined_run_name = run_name
                flat_run_dir = project_root / "outputs" / "flat_xgb" / flat_run_name
                distinction_run_dir = project_root / "outputs" / "distinction_specialist_xgb" / distinction_run_name
                refined_run_dir = project_root / "outputs" / output_root / refined_run_name
                prediction_csv_path = refined_run_dir / "refined_predictions.csv"
                prediction_metrics_path = refined_run_dir / "refined_metrics.json"

                for path in [flat_run_dir, distinction_run_dir, refined_run_dir]:
                    if path.exists():
                        shutil.rmtree(path)
                refined_run_dir.mkdir(parents=True, exist_ok=True)

                flat_train_script = Path(experiment["flat_train_script"]).expanduser().resolve()
                distinction_train_script = Path(experiment["distinction_train_script"]).expanduser().resolve()
                distinction_predict_script = Path(experiment["distinction_predict_script"]).expanduser().resolve()
                for script_path in [flat_train_script, distinction_train_script, distinction_predict_script]:
                    if not script_path.exists():
                        raise FileNotFoundError(f"Required script not found for {experiment_name}: {script_path}")

                flat_cmd = [
                    str(args.python),
                    str(flat_train_script),
                    "--input",
                    str(input_path),
                    "--num_seeds",
                    "1",
                    "--seed_start",
                    str(seed),
                    "--run_name",
                    flat_run_name,
                    "--no_plot",
                    *experiment["flat_args"],
                    *args.extra_args,
                ]
                distinction_cmd = [
                    str(args.python),
                    str(distinction_train_script),
                    "--input",
                    str(input_path),
                    "--num_seeds",
                    "1",
                    "--seed_start",
                    str(seed),
                    "--run_name",
                    distinction_run_name,
                    *experiment["distinction_args"],
                ]
                predict_cmd = [
                    str(args.python),
                    str(distinction_predict_script),
                    "--input",
                    str(input_path),
                    "--flat_model",
                    str((flat_run_dir / "artifacts" / "xgboost_model.joblib").resolve()),
                    "--distinction_model",
                    str((distinction_run_dir / "artifacts" / "distinction_model.joblib").resolve()),
                    "--output",
                    str(prediction_csv_path.resolve()),
                    "--metrics_output",
                    str(prediction_metrics_path.resolve()),
                    "--split_seed",
                    str(seed),
                    "--split_scope",
                    "test",
                    *experiment["predict_args"],
                ]
                print(f"\nRunning family={family} experiment={experiment_name} seed={seed}")
                for cmd in [flat_cmd, distinction_cmd, predict_cmd]:
                    print("Command:", " ".join(cmd))
                    subprocess.run(cmd, cwd=project_root, check=True)
                run_dir = refined_run_dir
                metrics_path = prediction_metrics_path
            elif family in {"flat_xgb_phase9_multi_specialist_refinement", "flat_xgb_phase9_ablation"}:
                flat_run_name = f"{run_name}_flat"
                distinction_run_name = f"{run_name}_distinction"
                pairwise_run_name = f"{run_name}_pairwise"
                refined_run_name = run_name
                flat_run_dir = project_root / "outputs" / "flat_xgb" / flat_run_name
                distinction_run_dir = project_root / "outputs" / "distinction_specialist_xgb" / distinction_run_name
                pairwise_run_dir = project_root / "outputs" / "pairwise_xgb" / pairwise_run_name
                refined_run_dir = project_root / "outputs" / output_root / refined_run_name
                prediction_csv_path = refined_run_dir / "refined_predictions.csv"
                prediction_metrics_path = refined_run_dir / "refined_metrics.json"

                for path in [flat_run_dir, distinction_run_dir, pairwise_run_dir, refined_run_dir]:
                    if path.exists():
                        shutil.rmtree(path)
                refined_run_dir.mkdir(parents=True, exist_ok=True)

                flat_train_script = Path(experiment["flat_train_script"]).expanduser().resolve()
                distinction_train_script = Path(experiment["distinction_train_script"]).expanduser().resolve()
                pairwise_train_script = Path(experiment["pairwise_train_script"]).expanduser().resolve()
                phase9_predict_script = Path(experiment["phase9_predict_script"]).expanduser().resolve()
                for script_path in [flat_train_script, distinction_train_script, pairwise_train_script, phase9_predict_script]:
                    if not script_path.exists():
                        raise FileNotFoundError(f"Required script not found for {experiment_name}: {script_path}")

                flat_cmd = [
                    str(args.python),
                    str(flat_train_script),
                    "--input",
                    str(input_path),
                    "--num_seeds",
                    "1",
                    "--seed_start",
                    str(seed),
                    "--run_name",
                    flat_run_name,
                    "--no_plot",
                    *experiment["flat_args"],
                    *args.extra_args,
                ]
                distinction_cmd = [
                    str(args.python),
                    str(distinction_train_script),
                    "--input",
                    str(input_path),
                    "--num_seeds",
                    "1",
                    "--seed_start",
                    str(seed),
                    "--run_name",
                    distinction_run_name,
                    *experiment["distinction_args"],
                ]
                pairwise_cmd = [
                    str(args.python),
                    str(pairwise_train_script),
                    "--input",
                    str(input_path),
                    "--pairwise_num_seeds",
                    "1",
                    "--pairwise_seed_start",
                    str(seed),
                    "--run_name",
                    pairwise_run_name,
                    "--no_plot",
                    *experiment["pairwise_args"],
                ]
                predict_cmd = [
                    str(args.python),
                    str(phase9_predict_script),
                    "--input",
                    str(input_path),
                    "--flat_model",
                    str((flat_run_dir / "artifacts" / "xgboost_model.joblib").resolve()),
                    "--distinction_model",
                    str((distinction_run_dir / "artifacts" / "distinction_model.joblib").resolve()),
                    "--pairwise_model",
                    str((pairwise_run_dir / "artifacts" / "pairwise_models.joblib").resolve()),
                    "--output",
                    str(prediction_csv_path.resolve()),
                    "--metrics_output",
                    str(prediction_metrics_path.resolve()),
                    "--split_seed",
                    str(seed),
                    "--split_scope",
                    "test",
                    *experiment["predict_args"],
                ]
                print(f"\nRunning family={family} experiment={experiment_name} seed={seed}")
                for cmd in [flat_cmd, distinction_cmd, pairwise_cmd, predict_cmd]:
                    print("Command:", " ".join(cmd))
                    subprocess.run(cmd, cwd=project_root, check=True)
                run_dir = refined_run_dir
                metrics_path = prediction_metrics_path
            else:
                train_script = Path(experiment["train_script"]).expanduser().resolve()
                if not train_script.exists():
                    raise FileNotFoundError(f"Training script not found for {experiment_name}: {train_script}")
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                cmd = [
                    str(args.python),
                    str(train_script),
                    "--input",
                    str(input_path),
                    "--num_seeds",
                    "1",
                    "--seed_start",
                    str(seed),
                    "--run_name",
                    run_name,
                    "--no_plot",
                    *experiment["args"],
                    *args.extra_args,
                ]
                print(f"\nRunning family={family} experiment={experiment_name} seed={seed}")
                print("Command:", " ".join(cmd))
                subprocess.run(cmd, cwd=project_root, check=True)
                metrics_path = run_dir / "metrics.json"
                confusion_plot_path = run_dir / "confusion_matrix_plot.png"

            if not metrics_path.exists():
                raise FileNotFoundError(f"Expected metrics.json not found after experiment={experiment_name} seed={seed}: {metrics_path}")

            metrics = load_metrics(metrics_path)
            final_metrics = _extract_final_metrics(metrics)
            labels = metrics.get("labels", [])
            confusion_matrix = metrics.get("confusion_matrix") or final_metrics.get("confusion_matrix")
            confusion_csv_path = confusion_dir / f"{family}_confusion_matrix_{experiment_name}_seed_{seed}.csv"
            if confusion_matrix is not None and labels:
                save_confusion_matrix_csv(confusion_matrix, labels, confusion_csv_path)

            rows.append(
                {
                    "family": family,
                    "experiment_name": experiment_name,
                    "experiment": experiment_name,
                    "seed": int(seed),
                    "selected_seed": metrics.get("selected_seed"),
                    "run_name": run_name,
                    "accuracy": final_metrics.get("accuracy"),
                    "macro_f1": final_metrics.get("macro_f1"),
                    "balanced_accuracy": final_metrics.get("balanced_accuracy"),
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "confusion_matrix_csv": str(confusion_csv_path) if confusion_csv_path.exists() else None,
                    "confusion_matrix_plot": str(confusion_plot_path) if confusion_plot_path and confusion_plot_path.exists() else None,
                }
            )

    summary_df = summarize_results(rows)
    for family in sorted(summary_df["family"].unique().tolist()) if not summary_df.empty else []:
        family_df = summary_df[summary_df["family"] == family].reset_index(drop=True)
        summary_path = experiments_dir / f"{family}_summary.csv"
        family_df.to_csv(summary_path, index=False, encoding="utf-8")
        print(f"\nSaved {family} summary CSV to: {summary_path}")
        print_summary(family_df, family)
        if family == "ova_xgb":
            tuned_df = family_df[family_df["experiment"].isin(["ova_tuned_small", "ova_tuned_large"])].reset_index(drop=True)
            if not tuned_df.empty:
                tuned_summary_path = experiments_dir / "ova_xgb_summary_tuned.csv"
                tuned_df.to_csv(tuned_summary_path, index=False, encoding="utf-8")
                print(f"Saved tuned OVA summary CSV to: {tuned_summary_path}")
    if not summary_df.empty:
        phase3_df = summary_df[summary_df["experiment"] == "flat_feature_phase3"].reset_index(drop=True)
        if not phase3_df.empty:
            phase3_summary_path = experiments_dir / "flat_feature_phase3_summary.csv"
            phase3_df.to_csv(phase3_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase3 summary CSV to: {phase3_summary_path}")
        phase4_df = summary_df[summary_df["experiment"] == "flat_feature_phase4"].reset_index(drop=True)
        if not phase4_df.empty:
            phase4_summary_path = experiments_dir / "flat_feature_phase4_summary.csv"
            phase4_df.to_csv(phase4_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase4 summary CSV to: {phase4_summary_path}")
        phase5_df = summary_df[summary_df["experiment"] == "flat_feature_phase5"].reset_index(drop=True)
        if not phase5_df.empty:
            phase5_summary_path = experiments_dir / "flat_feature_phase5_summary.csv"
            phase5_df.to_csv(phase5_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase5 summary CSV to: {phase5_summary_path}")
        phase6_df = summary_df[summary_df["experiment"] == "flat_feature_phase6"].reset_index(drop=True)
        if not phase6_df.empty:
            phase6_summary_path = experiments_dir / "flat_feature_phase6_summary.csv"
            phase6_df.to_csv(phase6_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase6 summary CSV to: {phase6_summary_path}")
        phase10_df = summary_df[summary_df["experiment"] == "flat_feature_phase10"].reset_index(drop=True)
        if not phase10_df.empty:
            phase10_summary_path = experiments_dir / "flat_feature_phase10_summary.csv"
            phase10_df.to_csv(phase10_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase10 summary CSV to: {phase10_summary_path}")
        phase7_threshold_df = summary_df[summary_df["experiment"] == "flat_phase7_class_thresholds"].reset_index(drop=True)
        if not phase7_threshold_df.empty:
            phase7_threshold_summary_path = experiments_dir / "flat_xgb_phase7_thresholds_summary.csv"
            phase7_threshold_df.to_csv(phase7_threshold_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase7 thresholds summary CSV to: {phase7_threshold_summary_path}")
        phase5_ablation_df = summary_df[
            summary_df["experiment"].isin(
                [
                    "flat_feature_phase5_ablate_module",
                    "flat_feature_phase5_ablate_tail",
                    "flat_feature_phase5_ablate_late",
                    "flat_feature_phase5_ablate_consistency",
                ]
            )
        ].reset_index(drop=True)
        if not phase5_ablation_df.empty:
            phase5_ablation_summary_path = experiments_dir / "flat_xgb_phase5_ablation_summary.csv"
            phase5_ablation_df.to_csv(phase5_ablation_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase5 ablation summary CSV to: {phase5_ablation_summary_path}")
        distinction_df = summary_df[summary_df["experiment"] == "distinction_refinement"].reset_index(drop=True)
        if not distinction_df.empty:
            distinction_summary_path = experiments_dir / "distinction_refined_summary.csv"
            distinction_df.to_csv(distinction_summary_path, index=False, encoding="utf-8")
            print(f"Saved distinction refined summary CSV to: {distinction_summary_path}")
        phase8_distinction_df = summary_df[summary_df["experiment"] == "flat_phase8_distinction_refinement"].reset_index(drop=True)
        if not phase8_distinction_df.empty:
            phase8_distinction_summary_path = experiments_dir / "flat_xgb_phase8_distinction_refinement_summary.csv"
            phase8_distinction_df.to_csv(phase8_distinction_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase8 distinction refinement summary CSV to: {phase8_distinction_summary_path}")
        phase9_df = summary_df[summary_df["experiment"] == "flat_phase9_multi_specialist_refinement"].reset_index(drop=True)
        if not phase9_df.empty:
            phase9_summary_path = experiments_dir / "flat_xgb_phase9_multi_specialist_summary.csv"
            phase9_df.to_csv(phase9_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase9 multi-specialist summary CSV to: {phase9_summary_path}")
            phase9_conf_src = experiments_dir / "flat_xgb_phase9_multi_specialist_refinement_confusion_matrices"
            phase9_conf_alias = experiments_dir / "flat_xgb_phase9_multi_specialist_confusion_matrices"
            phase9_conf_alias.mkdir(parents=True, exist_ok=True)
            if phase9_conf_src.exists():
                for csv_path in phase9_conf_src.glob("*.csv"):
                    shutil.copy2(csv_path, phase9_conf_alias / csv_path.name)
                print(f"Saved flat phase9 confusion-matrix alias CSVs to: {phase9_conf_alias}")
        phase9_ablation_df = summary_df[
            summary_df["experiment"].isin(
                [
                    "flat_phase9_ablation_distinction_only",
                    "flat_phase9_ablation_distinction_fail",
                    "flat_phase9_ablation_distinction_withdrawn",
                ]
            )
        ].reset_index(drop=True)
        if not phase9_ablation_df.empty:
            phase9_ablation_summary_path = experiments_dir / "flat_xgb_phase9_ablation_summary.csv"
            phase9_ablation_df.to_csv(phase9_ablation_summary_path, index=False, encoding="utf-8")
            print(f"Saved flat phase9 ablation summary CSV to: {phase9_ablation_summary_path}")

        phase9_routing_source_df = summary_df[
            summary_df["experiment"].isin(
                [
                    "flat_phase9_multi_specialist_refinement",
                    "flat_phase9_margin_tuned",
                    "flat_phase9_ablation_distinction_only",
                    "flat_phase9_ablation_distinction_fail",
                    "flat_phase9_ablation_distinction_withdrawn",
                ]
            )
        ].reset_index(drop=True)
        routing_rows: list[dict] = []
        for _, row in phase9_routing_source_df.iterrows():
            metrics_path = Path(str(row.get("metrics_path", "")))
            if not metrics_path.exists():
                continue
            metrics = load_metrics(metrics_path)
            stats = metrics.get("routing_stats", {})
            routing_rows.append(
                {
                    "experiment": str(row.get("experiment")),
                    "seed": int(row.get("seed")),
                    "distinction_override_rate": float(stats.get("distinction_override_rate", 0.0)),
                    "fail_specialist_trigger_rate": float(stats.get("fail_specialist_trigger_rate", stats.get("fail_pairwise_trigger_rate", 0.0))),
                    "withdrawn_specialist_trigger_rate": float(
                        stats.get("withdrawn_specialist_trigger_rate", stats.get("withdrawn_pairwise_trigger_rate", 0.0))
                    ),
                    "fail_override_rate": float(stats.get("fail_override_rate", 0.0)),
                    "withdrawn_override_rate": float(stats.get("withdrawn_override_rate", 0.0)),
                    "prediction_change_rate": float(stats.get("prediction_change_rate", stats.get("total_prediction_change_rate", 0.0))),
                    "avg_margin_routed": float(stats.get("avg_margin_routed", 0.0)),
                }
            )
        if routing_rows:
            routing_df = pd.DataFrame(routing_rows).sort_values(["experiment", "seed"]).reset_index(drop=True)
            routing_payload = {
                "by_seed": routing_df.to_dict(orient="records"),
                "distinction_override_rate": float(routing_df["distinction_override_rate"].mean()),
                "fail_specialist_trigger_rate": float(routing_df["fail_specialist_trigger_rate"].mean()),
                "withdrawn_specialist_trigger_rate": float(routing_df["withdrawn_specialist_trigger_rate"].mean()),
                "fail_override_rate": float(routing_df["fail_override_rate"].mean()),
                "withdrawn_override_rate": float(routing_df["withdrawn_override_rate"].mean()),
                "prediction_change_rate": float(routing_df["prediction_change_rate"].mean()),
                "avg_margin_routed": float(routing_df["avg_margin_routed"].mean()),
            }
            routing_path = experiments_dir / "phase9_routing_stats.json"
            with routing_path.open("w", encoding="utf-8") as handle:
                json.dump(routing_payload, handle, indent=2, sort_keys=True)
            print(f"Saved phase9 routing stats JSON to: {routing_path}")
        distinction_threshold_df = summary_df[
            summary_df["experiment"].isin(
                [
                    "distinction_refinement_threshold_045",
                    "distinction_refinement_threshold_050",
                    "distinction_refinement_threshold_055",
                    "distinction_refinement_threshold_060",
                    "distinction_refinement_threshold_065",
                    "distinction_refinement_threshold_070",
                ]
            )
        ].reset_index(drop=True)
        if not distinction_threshold_df.empty:
            threshold_summary_path = experiments_dir / "distinction_threshold_sweep_summary.csv"
            distinction_threshold_df.to_csv(threshold_summary_path, index=False, encoding="utf-8")
            print(f"Saved distinction threshold sweep summary CSV to: {threshold_summary_path}")


if __name__ == "__main__":
    main()
