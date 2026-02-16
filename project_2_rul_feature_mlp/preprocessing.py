# preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

def dvdsoc(voltage, soc, eps=1e-9):
    dV = np.diff(voltage, axis=1)
    dS = np.diff(soc, axis=1)
    dS[np.abs(dS) < eps] = np.nan
    out = dV / dS
    out = np.hstack([out, out[:, -1:]])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def low_pass_filter(data, fs, cutoff, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

def filter_dT_dV(x, window_size):
    if len(x) >= window_size:
        w = window_size if window_size % 2 == 1 else window_size - 1
        return savgol_filter(x, w, 5)
    return x

def compute_eol(soh, eol_threshold):
    idx = np.where(soh <= eol_threshold)[0]
    if len(idx) == 0:
        return len(soh) - 1
    return int(idx[0])
