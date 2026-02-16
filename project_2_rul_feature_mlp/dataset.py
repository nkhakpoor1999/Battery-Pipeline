# dataset.py
import os
import glob
import numpy as np

from config import FS, CUTOFF, FILTER_ORDER, WINDOW_SIZE
from preprocessing import dvdsoc, filter_dT_dV, compute_eol
from features import extract_rul_features_one_cycle

def load_battery_npz(file_path, fs=FS, cutoff=CUTOFF, order=FILTER_ORDER, window_size=WINDOW_SIZE):
    d = np.load(file_path, allow_pickle=True)

    V   = np.asarray(d["voltage"], dtype=float)
    Q   = np.asarray(d["max_capacity"], dtype=float)
    SOC = np.asarray(d["soc"], dtype=float)
    DQV = np.asarray(d["dqdv"], dtype=float)

    T = np.asarray(d["temperature"], dtype=float) if "temperature" in d.files else None

    dVdS = dvdsoc(V, SOC)

    dTdV = None
    if T is not None:
        dT = np.diff(T, axis=1)
        dV = np.diff(V, axis=1)
        dV[np.abs(dV) < 1e-9] = np.nan

        dTdV = dT / dV
        dTdV = np.hstack([dTdV, dTdV[:, -1:]])
        dTdV = np.nan_to_num(dTdV, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(dTdV.shape[0]):
            dTdV[i] = filter_dT_dV(dTdV[i], window_size)

        dTdV = -1.0 * dTdV

    qmax = float(np.nanmax(Q))
    SOH = Q / qmax

    return V, Q, SOC, DQV, dVdS, SOH, dTdV

def build_rul_dataset_with_groups(folder, eol_threshold=0.8, logger=None):
    folder = os.path.abspath(folder)
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))

    X_list, y_list, g_list = [], [], []

    if logger is not None:
        logger.write("RUL training batteries:")
        for f in files:
            logger.write(f"- {os.path.basename(f)}")
        logger.write()

    for f in files:
        battery_id = os.path.basename(f)
        V, Q, SOC, DQV, dVdS, SOH, dTdV = load_battery_npz(f)

        eol_idx = compute_eol(SOH, eol_threshold=eol_threshold)

        if eol_idx is None or eol_idx <= 0:
            if logger is not None:
                logger.write(f"{battery_id}: cycles={len(SOH)}, EOL={eol_idx}  --> SKIPPED (invalid EOL)")
            continue

        if logger is not None:
            logger.write(f"{battery_id}: cycles={len(SOH)}, EOL={eol_idx}")

        for c in range(eol_idx + 1):
            X_list.append(extract_rul_features_one_cycle(V, Q, SOC, DQV, dVdS, SOH, dTdV, c))
            y_list.append((eol_idx - c) / eol_idx)
            g_list.append(battery_id)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float).reshape(-1, 1)
    groups = np.asarray(g_list, dtype=object)

    if logger is not None:
        logger.write()
        logger.write("RUL dataset:")
        logger.write(f"X shape: {X.shape}")
        logger.write(f"y shape: {y.shape}")
        logger.write(f"#unique batteries: {len(np.unique(groups))}")
        logger.write()

    return X, y, groups

def predict_rul_all_cycles(new_battery_path, model, x_scaler, feat_idx):
    V, Q, SOC, DQV, dVdS, SOH, dTdV = load_battery_npz(new_battery_path)

    X_list = []
    for c in range(V.shape[0]):
        X_list.append(extract_rul_features_one_cycle(V, Q, SOC, DQV, dVdS, SOH, dTdV, c))

    X_new = np.asarray(X_list, dtype=float)[:, feat_idx]
    X_new_sc = x_scaler.transform(X_new)

    y_pred_ratio = model.predict(X_new_sc, verbose=0).reshape(-1)
    return y_pred_ratio
