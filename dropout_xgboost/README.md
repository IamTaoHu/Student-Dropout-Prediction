# XGBoost Dropout Prediction

## Project Status
- This repository is under active development.
- Primary workflow: `flat_xgb`
- The current canonical workflow is the flat multiclass XGBoost pipeline.
- Historical and experimental code paths are still present in the repository.

## Primary Workflow
- Canonical pipeline:
  - `src/models/flat_xgb/train.py`
- Backward-compatible train entrypoint:
  - `src/train_xgboost.py`
- Backward-compatible predict entrypoint:
  - `src/predict_xgboost.py`
- Experimental pipeline:
  - `hierarchical_xgb`
  - `src/models/hierarchical_xgb/`
  - `src/hierarchical_xgb/`

## Quickstart
### Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Data
- Place a CSV dataset in `data/`.
- The repository currently includes:
  - `data/data.csv`
  - `data/kuzilek_student_features.csv`

### Verify CLI
```powershell
python src/train_xgboost.py --help
```

## Training
### Canonical flat training command
```powershell
python src/train_xgboost.py ^
--input data/kuzilek_student_features.csv ^
--class_weight_mode balanced
```

### Notes
- `src/train_xgboost.py` delegates to the flat pipeline implementation in `src/models/flat_xgb/train.py`.
- The flat trainer supports additional options such as presets, calibration, threshold mode, and run naming.
- Use `python src/train_xgboost.py --help` to inspect the current CLI.

## Prediction
### Flat prediction command
```powershell
python src/predict_xgboost.py ^
--input data/kuzilek_student_features.csv ^
--model outputs/flat_xgb/<run_name>/artifacts/xgboost_model.joblib ^
--output outputs/flat_xgb/<run_name>/predictions_from_cli.csv
```

### Notes
- The predictor rebuilds the engineered feature set to match the saved model bundle.
- Replace `<run_name>` with the folder created by the corresponding training run.

## Outputs
- Flat canonical runs save to:
  - `outputs/flat_xgb/<run_name>/`
- Typical flat run artifacts:
  - `metrics.json`
  - `classification_report.txt`
  - `confusion_matrix_plot.png`
  - `feature_importance.csv`
  - `predictions.csv`
  - `seed_summary.csv`
  - `run_config.json`
  - `artifacts/xgboost_model.json`
  - `artifacts/xgboost_model.joblib`
- Hierarchical and older experiments also exist under other folders in `outputs/`, but they are not the primary documented workflow here.

## Source Structure
- `src/models/flat_xgb/`
  - Canonical flat train/predict implementation
- `src/models/hierarchical_xgb/`
  - Experimental hierarchical train/predict wrappers
- `src/hierarchical_xgb/`
  - Core hierarchical pipeline, feature engineering, thresholding, reporting
- `src/features/`
  - Feature-engineering entrypoints and re-exports
- `src/data/`
  - Data loading and split bridges
- `src/config/`
  - Paths and environment-driven settings
- `src/utils/`
  - Logging, seed, and I/O helpers
- `src/old/`
  - Legacy experiments; not part of the canonical workflow
