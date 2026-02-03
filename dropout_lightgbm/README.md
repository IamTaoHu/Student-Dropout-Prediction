# LightGBM Dropout Prediction

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data
Copy the dataset to `data/data.csv`.

## Run
```powershell
python -m src.train_lightgbm
python -m src.train_lightgbm_feature_groups
```

## Outputs
- `outputs/metrics.json`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/feature_group_results.csv`
- `outputs/feature_group_results.json`