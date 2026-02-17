# Sequence-Based Battery RUL Prediction (LSTM)

This project predicts **Remaining Useful Life (RUL)** of lithium-ion batteries using a **Long Short-Term Memory (LSTM)** network.  
Unlike the feature-based MLP approach, this model learns directly from **cycle-level time-series sequences**.

## Method Overview

- **State of Health (SoH)** is computed as `SoH = Q / Qmax`.
- **End-of-Life (EOL)** is defined as the cycle where `SoH ≤ eol_threshold`.
- The model predicts a normalized target:  
  `RUL_ratio = (EOL - cycle) / EOL`.

Each cycle is represented as a multivariate sequence including voltage and differential signals (e.g., dQ/dV, dV/dSOC, dT/dV).

## Model

A stacked LSTM architecture is used:

- LSTM layers for temporal modeling
- Dense layer with ReLU activation
- Linear output layer for regression

Loss: Mean Squared Error (MSE)  
Metric: Mean Absolute Error (MAE)

## Validation

**GroupKFold** is applied at the battery level to prevent data leakage and ensure realistic generalization across unseen batteries.

## How to Run

Train:
```bash
python train.py
