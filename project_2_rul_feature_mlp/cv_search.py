# cv_search.py
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from features import feature_idx_from_groups, ALL_GROUPS
from model import build_rul_model

def groupkfold_eval_subset(X, y, groups, groups_to_keep, n_splits, epochs, batch_size, l2_reg=1e-4, verbose=0):
    feat_idx = feature_idx_from_groups(groups_to_keep)
    gkf = GroupKFold(n_splits=n_splits)

    rows = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr = X[tr][:, feat_idx]
        X_va = X[va][:, feat_idx]
        y_tr = np.asarray(y[tr]).reshape(-1, 1)
        y_va = np.asarray(y[va]).reshape(-1, 1)

        x_scaler = StandardScaler()
        X_tr_sc = x_scaler.fit_transform(X_tr)
        X_va_sc = x_scaler.transform(X_va)

        model = build_rul_model(input_dim=X_tr_sc.shape[1], l2_reg=l2_reg)
        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0)

        history = model.fit(
            X_tr_sc, y_tr,
            validation_data=(X_va_sc, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        best_epoch = len(history.history["loss"])

        y_pred = model.predict(X_va_sc, verbose=0).reshape(-1)
        y_true = y_va.reshape(-1)

        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        rows.append({
            "fold": fold,
            "best_epoch": int(best_epoch),
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse),
        })

    df = pd.DataFrame(rows)
    median_best_epoch = int(np.median(df["best_epoch"].values))

    summary = {
        "n_splits": int(n_splits),
        "groups_to_keep": tuple(groups_to_keep),
        "dim": int(len(feat_idx)),
        "median_best_epoch": median_best_epoch,
        "R2_mean": float(df["R2"].mean()),
        "R2_std": float(df["R2"].std(ddof=1)),
        "MAE_mean": float(df["MAE"].mean()),
        "MAE_std": float(df["MAE"].std(ddof=1)),
        "RMSE_mean": float(df["RMSE"].mean()),
        "RMSE_std": float(df["RMSE"].std(ddof=1)),
    }
    return summary, df

def run_ablation(X, y, groups, n_splits, epochs, batch_size, l2_reg=1e-4, verbose=0):
    results = []
    base_summary, _ = groupkfold_eval_subset(
        X, y, groups, ALL_GROUPS,
        n_splits=n_splits, epochs=epochs, batch_size=batch_size, l2_reg=l2_reg, verbose=verbose
    )
    results.append({"setting": "ALL", **base_summary})

    for g in ALL_GROUPS:
        keep = [x for x in ALL_GROUPS if x != g]
        s, _ = groupkfold_eval_subset(
            X, y, groups, keep,
            n_splits=n_splits, epochs=epochs, batch_size=batch_size, l2_reg=l2_reg, verbose=verbose
        )
        results.append({"setting": f"Without_{g}", **s})

    return pd.DataFrame(results).sort_values("setting").reset_index(drop=True)

def choose_groups_from_ablation_and_search(X, y, groups, n_splits, epochs, batch_size, l2_reg=1e-4, verbose=0, logger=None):
    ablation_df = run_ablation(X, y, groups, n_splits, epochs, batch_size, l2_reg=l2_reg, verbose=verbose)

    base = ablation_df.loc[ablation_df["setting"] == "ALL"].iloc[0]
    ablation_df["ΔR2"] = ablation_df["R2_mean"] - base["R2_mean"]
    ablation_df["ΔMAE"] = ablation_df["MAE_mean"] - base["MAE_mean"]
    ablation_df["ΔRMSE"] = ablation_df["RMSE_mean"] - base["RMSE_mean"]

    if logger is not None:
        logger.write("=== Ablation summary (GroupKFold mean) ===")
        logger.write_df("", ablation_df[["setting","R2_mean","MAE_mean","RMSE_mean","ΔR2","ΔMAE","ΔRMSE"]])

    rows = []
    for r in range(1, len(ALL_GROUPS) + 1):
        for subset in itertools.combinations(ALL_GROUPS, r):
            s, _ = groupkfold_eval_subset(
                X, y, groups, subset,
                n_splits=n_splits, epochs=epochs, batch_size=batch_size, l2_reg=l2_reg, verbose=verbose
            )
            rows.append(s)

    search_df = pd.DataFrame(rows).sort_values(
        ["MAE_mean", "RMSE_mean", "R2_mean"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    chosen_groups = tuple(search_df.iloc[0]["groups_to_keep"])

    if logger is not None:
        logger.write("=== Best subset by GroupKFold (min MAE, tie-break RMSE then R2) ===")
        logger.write_df("", search_df.head(10)[["groups_to_keep","dim","R2_mean","MAE_mean","RMSE_mean","R2_std","MAE_std","RMSE_std"]])
        logger.write(f"✅ CHOSEN GROUPS: {chosen_groups}")
        logger.write()

    return chosen_groups, ablation_df, search_df
