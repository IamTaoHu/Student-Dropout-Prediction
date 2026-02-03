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
python src\train_encoder.py
```

### Outputs
- `outputs/model/`: saved encoder model artifacts
- `outputs/metrics.json`: training/evaluation metrics

The `model_comparator` can read `outputs/metrics.json` because the folder name is `dropout_llm_encoder`.
