# Feature-Based Battery RUL Prediction (MLP)

This project predicts **Remaining Useful Life (RUL)** of lithium-ion batteries using a **Multi-Layer Perceptron (MLP)** regressor. The model uses features extracted from each discharge cycle to estimate how close the battery is to its **End-of-Life (EOL)**.

---

## Method Overview

- **State of Health (SoH)** is computed as `SoH = Q / Qmax`.
- **End-of-Life (EOL)** is defined as the cycle where `SoH ≤ eol_threshold`.
- The model predicts a **normalized RUL**: `RUL_ratio = (EOL - cycle) / EOL`.

### Feature Engineering
Features include voltage statistics, dQ/dV, dV/dSOC, and dT/dV (if available). These are used to train the MLP model.

### Model
An MLP with ReLU activation and a linear output layer is used to predict the normalized RUL.

### Validation
**GroupKFold** is used for cross-validation, ensuring no data leakage from the same battery between train and validation.

---

## How to Run

- Train the model:  
  ```bash
  python train.py
