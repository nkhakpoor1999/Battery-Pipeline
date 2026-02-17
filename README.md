# 🔋 Battery RUL Prediction Pipeline

A modular machine learning framework for battery dataset classification and Remaining Useful Life (RUL) prediction.

This repository includes three complementary modeling stages:

1. **Project 1: Dataset Classification** 
2. **Project 2: Feature-based RUL Regression**  
3. **Project 3: Sequence-based RUL Prediction** 

---

## 📊 Data & Preprocessing

This project integrates multiple heterogeneous battery datasets, including:

- **NASA Battery Dataset**
- **Oxford Battery Degradation Dataset**
- **MIT Battery Dataset**
- **Laboratory datasets collected and tested under controlled experimental conditions**
  - Lab-Li-EVE
  - Lab-Li-NMC
  - Lab-Li-LCO

  ### 🔬 Preprocessing Overview

All datasets were systematically preprocessed before modeling.

Only the **discharge phase** of each cycle was retained, as it contains the most informative degradation behavior.

Preprocessing included:

- Cycle segmentation  
- SOH computation and EOL detection  
- Derivative signal construction (dV/dSOC, dQ/dV, dT/dV)  
- Noise filtering and smoothing  
- Curve length standardization via interpolation  
- Consistent conversion to structured `.npz` format  

All models operate on standardized `.npz` files.  
Dataset-specific preprocessing scripts are intentionally excluded.

## 📂 Project Structure

```
project_1_brand_classifier/      # Dataset classification
project_2_rul_feature_mlp/       # Feature-based RUL regression
project_3_rul_lstm/              # Sequence-based RUL (LSTM)
```
Detailed explanations for each module are provided in the respective project-level README files.

---

## 🛠 Tech Stack

Python · TensorFlow/Keras · NumPy · SciPy · Scikit-learn · Pandas

---

## 🔍 Engineering Highlights

- Battery-level data leakage prevention  
- GroupKFold cross-validation  
- Feature ablation and subset search  
- Reproducible training  
- Model + scaler artifact persistence  

---

This repository demonstrates structured battery analytics workflows from feature engineering to deep sequence modeling.
