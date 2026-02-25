# CatBoost Student Dropout Prediction

## Environment Setup (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1

Install dependencies if needed:
pip install -r requirements.txt

Run training stage 1:
py .\src\train_stage1.py ^
  --input .\data\kuzilek_student_features_clean.csv ^
  --target final_result ^
  --stage1_mode ova4 ^
  --route_policy margin_gate ^
  --optimize_joint 1 ^
  --grid_t_pass "0.50,0.55,0.60" ^
  --grid_t_notpass "0.30,0.35,0.40" ^
  --grid_t_margin "0.03,0.05,0.08" ^
  --pass_recall_min 0.83
Output:
outputs/hierarchical/stage1_model.cbm
outputs/hierarchical/stage1_threshold.json
outputs/hierarchical/stage1_joint_sweep_results.json

Run training stage 2:
py .\src\train_stage2.py ^
  --input .\data\kuzilek_student_features_clean.csv ^
  --target final_result ^
  --mode ova_sweep ^
  --loss_mode ova ^
  --base_weight_mode balanced ^
  --sweep_depth "9,10" ^
  --sweep_lr "0.03" ^
  --sweep_l2 "9,12" ^
  --sweep_bag_temp "0.6" ^
  --sweep_rstrength "2" ^
  --sweep_iters "5000"

Output:
outputs/hierarchical/stage2_model.cbm
outputs/hierarchical/stage2_metrics.json
outputs/hierarchical/stage2_labels.json


Run Prediction (Default)
py .\src\predict_hierarchical.py `
  --input .\data\kuzilek_student_features_clean.csv `
  --target final_result
Output:
outputs/hierarchical/predictions_4class.csv
outputs/hierarchical/hierarchical_metrics_4class.json

Run Prediction (Custom Input via CLI)
You can override the input file using --input:
py src\predict_catboost.py --input data/data.csv

------------------------------------------------------------
EXPECTED FINAL PERFORMANCE
------------------------------------------------------------
Macro F1 ≈ 0.70–0.72
Accuracy ≈ 0.74
Balanced Accuracy ≈ 0.74
```
