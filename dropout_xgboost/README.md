# XGBoost Dropout Prediction

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
py .\src\train_xgboost.py --input .\data\kuzilek_clean_plus.csv --num_seeds 15 --seed_start 30 --learning_rate 0.02 --n_estimators 25000 --max_depth 8 --min_child_weight 2 --subsample 0.9 --colsample_bytree 0.9 --reg_lambda 2 --reg_alpha 0 --gamma 0 --early_stopping_rounds 700 --select_by accuracy
python src\train_xgboost_feature_groups
```

## Outputs
- `outputs/metrics.json`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/feature_group_results.csv`
- `outputs/feature_group_results.json`