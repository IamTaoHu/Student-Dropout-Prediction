## Dropout LLM Encoder

This module trains and uses an encoder-only LLM (no decoder) to produce features for student dropout prediction.

### Folder structure
```
dropout_llm_encoder/
  .venv/
  input/
    data.csv
  src/
    train_encoder.py
    predict_encoder.py
    utils_textify.py
    metrics.py
  outputs/
    model/
    metrics.json
  requirements.txt
  README.md
```

### Setup and training (Windows PowerShell)
From the repo root:
```
cd code/dropout_llm_encoder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_encoder --epochs 8 --max-length 128 --task multiclass --use_class_weights --model-name distilbert-base-uncased
python -m src.predict_encoder --input_csv "input/data.csv" --output_csv "outputs/predictions.csv" --task multiclass
```

### Predict (Windows PowerShell)
```
python -m src.predict_encoder --input_csv "input/data.csv" --output_csv "outputs/predictions.csv" --task multiclass
```

### Label mapping
- Dropout = 0
- Enrolled = 1
- Graduate = 2

### Outputs
- `outputs/model/`: saved encoder model artifacts
- `outputs/metrics.json`: training/evaluation metrics
- `outputs/predictions.csv`: prediction outputs

The `model_comparator` can read `outputs/metrics.json` because the folder name is `dropout_llm_encoder`.
