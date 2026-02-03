# dropout_tabnet

## Overview
TabNet training and inference for tabular student dropout prediction.

## Folder structure
- `data/`: input datasets
- `outputs/`: models, metrics, predictions
- `src/`: training and prediction scripts

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Data placement
- Primary file: `code/dropout_tabnet/data/data.csv`
- Recommended target column: `Target` (auto-inferred if present)
- Dropout mapping (string targets): values containing "drop" => 1, else 0

## Train
Run from `code/dropout_tabnet`:
```bash
python -m src.train_tabnet --data_path "data/data.csv"
```

## Predict
```bash
python -m src.predict_tabnet --data_path "data/data.csv"
```

## Outputs
- `outputs/models/tabnet_model.zip`, `outputs/models/scaler.joblib`, `outputs/models/meta.json`
- `outputs/metrics/metrics.json` and `outputs/metrics/metrics_predict.json`
- `outputs/predictions/test_predictions.csv` and `outputs/predictions/predictions.csv`

## Metrics included
F1, Recall, ROC-AUC, PR-AUC, Accuracy, Confusion Matrix (TN, FP, FN, TP).