from __future__ import annotations

import os
import glob
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def dvdsoc(voltage: np.ndarray, soc: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    dV = np.diff(voltage, axis=1)
    dS = np.diff(soc, axis=1)
    dS[np.abs(dS) < eps] = np.nan
    out = dV / dS
    out = np.hstack([out, out[:, -1:]])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def low_pass_filter(data: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def filter_1d_savgol(x: np.ndarray, window_size: int, poly: int) -> np.ndarray:
    if len(x) < window_size:
        return x
    w = window_size if window_size % 2 == 1 else window_size - 1
    if w < 3:
        return x
    return savgol_filter(x, w, poly)


def load_battery_npz(
    file_path: Path,
    fs: float = 1.0,
    cutoff: float = 0.05,
    lowpass_order: int = 3,
    savgol_window: int = 21,
    savgol_poly: int = 5,
) -> Dict[str, np.ndarray]:
    d = np.load(file_path, allow_pickle=True)

    V   = np.asarray(d["voltage"], dtype=float)
    SOC = np.asarray(d["soc"], dtype=float)
    Q   = np.asarray(d["max_capacity"], dtype=float)
    DQV = np.asarray(d["dqdv"], dtype=float)

    T = np.asarray(d["temperature"], dtype=float) if "temperature" in d.files else None

    dVdS = dvdsoc(V, SOC)

    dTdV = None
    if T is not None:
        T_f = np.copy(T)

        # low-pass each cycle (row-wise) if long enough
        for i in range(T_f.shape[0]):
            if T_f.shape[1] > 3 * (lowpass_order + 1):
                T_f[i] = low_pass_filter(T_f[i], fs, cutoff, lowpass_order)

        dT = np.diff(T_f, axis=1)
        dV = np.diff(V, axis=1)
        dV[np.abs(dV) < 1e-9] = np.nan

        dTdV = dT / dV
        dTdV = np.hstack([dTdV, dTdV[:, -1:]])
        dTdV = np.nan_to_num(dTdV, nan=0.0, posinf=0.0, neginf=0.0)

        # savgol each cycle
        for i in range(dTdV.shape[0]):
            dTdV[i] = filter_1d_savgol(dTdV[i], savgol_window, savgol_poly)

        dTdV = -1.0 * dTdV

    qmax = float(np.nanmax(Q))
    SOH = Q / qmax if qmax > 0 else np.zeros_like(Q)

    return {
        "voltage": V,
        "soc": SOC,
        "max_capacity": Q,
        "soh": SOH,
        "dqdv": DQV,
        "dv_dsoc": dVdS,
        "dtdv": dTdV,  # may be None
    }


def build_brand_dataset(
    database_dir: Path,
    labels: Dict[str, Tuple[str, int]],
    fs: float = 1.0,
    cutoff: float = 0.05,
    lowpass_order: int = 3,
    savgol_window: int = 21,
    savgol_poly: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    database_dir = Path(database_dir).resolve()
    X_list, y_list, battery_list = [], [], []

    for prefix, (class_name, class_id) in labels.items():
        pattern = str(database_dir / f"{prefix}*.npz")
        for p in glob.glob(pattern):
            path = Path(p)
            d = load_battery_npz(
                path,
                fs=fs,
                cutoff=cutoff,
                lowpass_order=lowpass_order,
                savgol_window=savgol_window,
                savgol_poly=savgol_poly,
            )
            V = d["voltage"]
            dVdS = d["dv_dsoc"]

            n_cycles = V.shape[0]
            for c in range(n_cycles):
                v_c = V[c]
                dvds_c = dVdS[c]

                feats = [
                    float(np.nanmean(v_c)),  float(np.nanstd(v_c)),  float(np.nanmax(v_c)),
                    float(np.nanmean(dvds_c)), float(np.nanstd(dvds_c)), float(np.nanmax(dvds_c)),
                ]
                X_list.append(feats)
                y_list.append(int(class_id))
                battery_list.append(path.name)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=int)
    groups = np.asarray(battery_list, dtype=object)

    id_to_name = {cid: cname for (_, (cname, cid)) in labels.items() if isinstance(cid, int)}
    # above line not perfect because multiple prefixes map to same id; fix:
    id_to_name = {cid: cname for (cname, cid) in set(labels.values())}

    return X, y, groups, id_to_name
