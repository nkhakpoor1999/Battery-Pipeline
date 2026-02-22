# 🔋 Project 2 — Feature-Based RUL Prediction (MLP)

A machine learning pipeline for RUL prediction using engineered cycle-level features and a fully connected neural network.

---

## 🎯 Objective

Given battery cycle data (`.npz` files), the pipeline:

- Computes engineered degradation features per cycle
- Defines EOL based on SOH threshold
- Converts cycle index to normalized RUL target
- Performs battery-level cross-validation
- Selects optimal feature groups via ablation + search
- Trains a final regression model

---

## 📊 Validation Strategy

- GroupKFold split (group = battery ID)
- Prevents cycle-level leakage
- Feature group ablation + search
- Median optimal epoch from CV used for final training

---

## 🔬 Feature Engineering

Features are extracted per cycle using:

- Voltage statistics
- dV/dSOC
- dQ/dV
- dT/dV (if temperature available)
- V(SOC), including voltage at 20%, 50%, and 80% SOC

Feature groups are evaluated through systematic ablation and subset search.

---

## 🚀 Training

```bash
python project_2_rul_feature_mlp/train.py
```

Dataset configuration is controlled via:

```python
DATASET_KEY = "NASA" # MIT / OXFORD / Lab-Li-LCO / Lab-Li-NMC / Lab-Li-EVE*
```

---

## 📁 Outputs

```
artifacts_3/<DATASET>/
├── model (Keras)
├── scaler
├── selected feature indices
├── ablation results (CSV)
├── search results (CSV)
├── Excel summary
└── training report.txt
```

## 🔎 Evaluate on New Battery


```bash
python project_2_rul_feature_mlp/evaluate.py \
    --model_dir project_2_rul_feature_mlp/artifacts_3/NASA \
    --battery "PATH_TO_NEW_BATTERY.npz"
```

### 📈 Output

- Predicted RUL ratio per cycle  
- Predicted remaining cycles  
- R² / MAE / RMSE metrics  
- `true_vs_pred_rul_ratio.png` visualization  

This enables practical deployment of the trained model for real battery prognosis.


### 📊 Example Results

#### Oxford Dataset

![Oxford True vs Pred](outputs/true_vs_pred_rul_ratio-OXFORD.png)  
![Oxford Ablation](outputs/ablation_r2_mean_bar-OXFORD.png)

📄 Full Report: [report-OXFORD.txt](examples/report-OXFORD.txt)

---

#### Lab-Li-EVE Dataset

![Lab True vs Pred](outputs/true_vs_pred_rul_ratio-Lab-Li-EVE.png)  
![Lab Ablation](outputs/ablation_r2_mean_bar-Lab-Li-EVE.png)

📄 Full Report: [report-Lab-Li-EVE.txt](examples/report-Lab-Li-EVE.txt)
