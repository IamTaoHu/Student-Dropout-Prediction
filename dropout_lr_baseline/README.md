# Logistic Regression Baseline for Dropout Prediction

This project provides a simple logistic regression baseline to predict student dropout.

## Setup
1) Create a virtual environment
```powershell
python -m venv .venv
```
2) Activate it
```powershell
.venv\Scripts\Activate.ps1
```
3) Install dependencies
```powershell
pip install -r requirements.txt
```

## Data
Place your dataset at `dropout_lr_baseline/data/data.csv` (see `DATA_PATH` in `.env.example`).

## How to Run
1) Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```
2) Run the baseline logistic regression
```powershell
python -m src.train_logreg
```
3) Run the feature-group study
```powershell
python -m src.train_logreg_feature_groups
```

## Outputs
Training runs write artifacts and evaluation outputs to `outputs/` (e.g., `outputs/metrics.json` and
`outputs/feature_group_results.json`).
