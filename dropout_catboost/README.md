# CatBoost Dropout Prediction

## How to Run
1) Activate the virtual environment (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```
2) Train the baseline CatBoost model
```powershell
python -m src.train_catboost
```
3) Run the feature group study
```powershell
python -m src.train_catboost_feature_groups
```
4) Run explainability (if available)
```powershell
python -m src.run_explain
```

## SHAP (Full Model)
```powershell
python -m src.run_shap_full_catboost
```
Outputs are saved under `outputs/shap_full/run_YYYYMMDD_HHMMSS/`.

## Outputs
Artifacts and evaluation outputs are saved under `outputs/`, with per-run folders for feature group runs.
