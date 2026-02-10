# Logistic Regression Baseline

Minimal logistic regression baseline for student dropout prediction.

## Data
Place your dataset at `data/data.csv`. The target column is auto-detected from common names
(`Target`, `Status`, `Outcome`, `Class`, etc.).

3-class mapping:
- dropout = 0
- enrolled = 1
- graduate = 2

## Training
```powershell
py src\train_logreg.py
```

## Prediction
```powershell
py src\predict_logreg.py --input data\data.csv
```

## Outputs
- `outputs/model.joblib`
- `outputs/metrics.json`
- `outputs/predictions.csv`
