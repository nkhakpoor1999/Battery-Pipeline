# evaluate.py
import os
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from artifacts import load_rul_artifacts
from dataset import load_battery_npz
from preprocessing import compute_eol
from features import extract_rul_features_one_cycle

def evaluate_new_battery_like_before_2(new_battery_path, dataset_name, out_dir, save_pred=True, clip_ratio_pred=True, eol_threshold=0.8):
    rul_model, x_scaler, feat_idx, chosen_groups, eol_thr = load_rul_artifacts(out_dir, dataset_name)

    V, Q, SOC, DQV, dVdS, SOH, dTdV = load_battery_npz(new_battery_path)

    eol_idx = compute_eol(SOH, eol_threshold)
    if eol_idx is None or eol_idx <= 0:
        raise ValueError(f"EOL not found or invalid for {os.path.basename(new_battery_path)}")

    X_list, y_ratio_list = [], []
    for c in range(eol_idx + 1):
        X_list.append(extract_rul_features_one_cycle(V, Q, SOC, DQV, dVdS, SOH, dTdV, c))
        y_ratio_list.append((eol_idx - c) / eol_idx)

    X = np.asarray(X_list, dtype=float)[:, feat_idx]
    y_true_ratio = np.asarray(y_ratio_list, dtype=float).reshape(-1)

    X_sc = x_scaler.transform(X)
    y_pred_ratio = rul_model.predict(X_sc, verbose=0).reshape(-1)

    if clip_ratio_pred:
        y_pred_ratio = np.clip(y_pred_ratio, 0.0, 1.0)

    r2_ratio   = r2_score(y_true_ratio, y_pred_ratio)
    mae_ratio  = mean_absolute_error(y_true_ratio, y_pred_ratio)
    rmse_ratio = np.sqrt(mean_squared_error(y_true_ratio, y_pred_ratio))

    y_true_cycles = y_true_ratio * eol_idx
    y_pred_cycles = y_pred_ratio * eol_idx

    r2_c   = r2_score(y_true_cycles, y_pred_cycles)
    mae_c  = mean_absolute_error(y_true_cycles, y_pred_cycles)
    rmse_c = np.sqrt(mean_squared_error(y_true_cycles, y_pred_cycles))

    bn = os.path.basename(new_battery_path)

    print(f"\nRUL on battery {bn} (ratio space):")
    print(f"ACTUAL EOL: {eol_idx} cycles")
    print(f"  R2   : {r2_ratio:.4f}")
    print(f"  MAE  : {mae_ratio:.4f}")
    print(f"  RMSE : {rmse_ratio:.4f}")

    print(f"\nRUL on battery {bn} (cycles, derived using actual EOL):")
    print(f"  R2   : {r2_c:.4f}")
    print(f"  MAE  : {mae_c:.2f} cycles")
    print(f"  RMSE : {rmse_c:.2f} cycles")

    pred_csv = None
    if save_pred:
        ds_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)
        base = os.path.splitext(bn)[0]
        pred_csv = os.path.join(ds_dir, f"pred_eval_{base}.csv")

        abs_err_cycles = np.abs(y_true_cycles - y_pred_cycles)

        pd.DataFrame({
            "cycle": np.arange(len(y_pred_ratio)),
            "rul_true_ratio": y_true_ratio,
            "rul_pred_ratio": y_pred_ratio,
            "rul_true_cycles": y_true_cycles,
            "rul_pred_cycles": y_pred_cycles,
            "abs_error_cycles": abs_err_cycles,
        }).to_csv(pred_csv, index=False)

        print("\nsaved:", pred_csv)

    return {
        "cycle": np.arange(len(y_pred_ratio)),
        "rul_true_ratio": y_true_ratio,
        "rul_pred_ratio": y_pred_ratio,
        "rul_true_cycles": y_true_cycles,
        "rul_pred_cycles": y_pred_cycles,
        "battery": bn,
        "dataset": dataset_name,
        "chosen_groups": chosen_groups,
        "eol_threshold": float(eol_thr),
        "actual_eol": int(eol_idx),
        "metrics_ratio": {"R2": float(r2_ratio), "MAE": float(mae_ratio), "RMSE": float(rmse_ratio)},
        "metrics_cycles": {"R2": float(r2_c), "MAE": float(mae_c), "RMSE": float(rmse_c)},
        "pred_csv": pred_csv
    }

if __name__ == "__main__":
    out = evaluate_new_battery_like_before_2(
        new_battery_path=r"D:\Uni\Thesis\code & data\Battery\Battery Project\new\OXFORD_8.npz",
        dataset_name="OXFORD",
        out_dir="artifacts_3",
        save_pred=True,
        clip_ratio_pred=True,
        eol_threshold=0.8
    )
