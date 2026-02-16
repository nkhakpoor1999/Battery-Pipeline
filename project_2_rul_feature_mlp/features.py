# features.py
import numpy as np

FEATURE_GROUPS = {
    "V_stats":  [0, 1, 2],
    "dQdV":     [3, 4, 5],
    "dVdSOC":   [6, 7, 8],
    "dTdV":     [9, 10, 11],
    "V_at_SOC": [12, 13, 14],
}
ALL_GROUPS = list(FEATURE_GROUPS.keys())

def feature_idx_from_groups(groups_to_keep, feature_groups=FEATURE_GROUPS):
    idx = []
    for g in groups_to_keep:
        idx.extend(feature_groups[g])
    return np.array(sorted(set(idx)), dtype=int)

def extract_rul_features_one_cycle(V, Q, SOC, DQV, dVdS, SOH, dTdV, c):
    v_c = V[c]
    dqdv_c = DQV[c]
    s_c = SOC[c]
    dvds_c = dVdS[c]

    mean_v, std_v, max_v = np.nanmean(v_c), np.nanstd(v_c), np.nanmax(v_c)
    mean_dqdv, std_dqdv, max_dqdv = np.nanmean(dqdv_c), np.nanstd(dqdv_c), np.nanmax(dqdv_c)
    mean_dvds, std_dvds, max_dvds = np.nanmean(dvds_c), np.nanstd(dvds_c), np.nanmax(dvds_c)

    if dTdV is None:
        mean_dtdv = std_dtdv = max_dtdv = 0.0
    else:
        dtdv_c = dTdV[c]
        mean_dtdv = np.nanmean(dtdv_c)
        std_dtdv  = np.nanstd(dtdv_c)
        max_dtdv  = np.nanmax(dtdv_c)

    order = np.argsort(s_c)
    s_sorted = s_c[order]
    v_sorted = v_c[order]

    v20 = np.interp(0.2, s_sorted, v_sorted)
    v50 = np.interp(0.5, s_sorted, v_sorted)
    v80 = np.interp(0.8, s_sorted, v_sorted)

    return np.array([
        mean_v, std_v, max_v,
        mean_dqdv, std_dqdv, max_dqdv,
        mean_dvds, std_dvds, max_dvds,
        mean_dtdv, std_dtdv, max_dtdv,
        v20, v50, v80
    ], dtype=float)
