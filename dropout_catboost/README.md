# CatBoost Student Dropout Prediction (Minimal)

This folder contains a **minimal, self-contained CatBoost pipeline** for **3-class student status prediction**:

- `dropout` → 0  
- `enrolled` → 1  
- `graduate` → 2  

The goal is to keep the code **simple, readable, and runnable from a single folder** without package-style imports.

---

## Folder Structure
dropout_catboost/
├── train_catboost.py # Train CatBoost (3-class) + sklearn metrics
├── predict_catboost.py # Predict with trained model (CLI supported)
├── data/
│ └── data.csv # Input dataset
├── outputs/
│ ├── catboost_model.cbm # Trained model
│ ├── metrics.json # Saved evaluation metrics
│ └── predictions.csv # Prediction results
├── _optional/ # Advanced / non-minimal scripts (SHAP, explainability)
├── requirements.txt
└── README.md

---

## Environment Setup (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1

Install dependencies if needed:
pip install -r requirements.txt

Run training from this folder:
py train_catboost.py

Run Prediction (Default)

Use the trained model to predict on the default dataset:
py predict_catboost.py

 Run Prediction (Custom Input via CLI)
You can override the input file using --input:
py predict_catboost.py --input data/data.csv

Optional arguments:
py predict_catboost.py --input data/data.csv --output outputs/preds.csv
py predict_catboost.py --model outputs/catboost_model.cbm
```
