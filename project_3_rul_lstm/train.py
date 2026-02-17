from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
from pathlib import Path

import joblib
import pandas as pd

from utils import set_seed, ensure_dir, save_json, now_tag
from dataset import build_rul_dataset_seq_lstm
from model import build_rul_model_seq_lstm
from dataset_configs import DATASET_CONFIGS, DATA_ROOT


from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def groupkfold_train_eval(
    data_path: Path,
    eol_threshold: float,
    n_splits: int,
    epochs: int,
    batch_size: int,
    verbose: int,
    seed: int,
    savgol_window: int,
    savgol_poly: int,
    l2_reg: float,
    early_stop_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    min_lr: float,
    features: list,
):
    set_seed(seed)

    X, y, groups = build_rul_dataset_seq_lstm(
        folder=data_path,
        eol_threshold=eol_threshold,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        features=features
    )
    N, W, F = X.shape

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
        X_va_sc = scaler.transform(X_va.reshape(-1, F)).reshape(X_va.shape)

        model = build_rul_model_seq_lstm(W, F, l2_reg=l2_reg)

        early_stop = EarlyStopping(monitor="val_loss", patience=early_stop_patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr
        )

        hist = model.fit(
            X_tr_sc, y_tr,
            validation_data=(X_va_sc, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        ran_epochs = len(hist.history["loss"])

        y_pred = model.predict(X_va_sc, verbose=0).reshape(-1)
        y_true = y_va.reshape(-1)

        rows.append({
            "fold": fold,
            "epochs_ran": ran_epochs,
            "R2": float(r2_score(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float((mean_squared_error(y_true, y_pred) ** 0.5)),
        })

    df = pd.DataFrame(rows)

    summary = {
        "n_splits": int(n_splits),
        "n_samples": int(N),
        "W": int(W),
        "F": int(F),
        "R2_mean": float(df["R2"].mean()),
        "MAE_mean": float(df["MAE"].mean()),
        "RMSE_mean": float(df["RMSE"].mean()),
        "median_best_epoch": int(df["epochs_ran"].median()),
    }
    return summary, df


def train_final_model(
    data_path: Path,
    eol_threshold: float,
    final_epochs: int,
    batch_size: int,
    verbose: int,
    seed: int,
    savgol_window: int,
    savgol_poly: int,
    l2_reg: float,
    features: list,
):
    set_seed(seed)

    X, y, groups = build_rul_dataset_seq_lstm(
        folder=data_path,
        eol_threshold=eol_threshold,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        features=features
    )
    N, W, F = X.shape

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X.reshape(-1, F)).reshape(X.shape)

    model = build_rul_model_seq_lstm(W, F, l2_reg=l2_reg)
    hist = model.fit(X_sc, y, epochs=int(final_epochs), batch_size=batch_size, verbose=verbose)

    y_pred = model.predict(X_sc, verbose=0).reshape(-1)
    y_true = y.reshape(-1)

    metrics = {
        "train_R2": float(r2_score(y_true, y_pred)),
        "train_MAE": float(mean_absolute_error(y_true, y_pred)),
        "train_RMSE": float((mean_squared_error(y_true, y_pred) ** 0.5)),
        "epochs_ran": int(len(hist.history["loss"])),
        "n_samples": int(N),
        "n_batteries": int(len(set(groups))),
        "W": int(W),
        "F": int(F),
    }

    return model, scaler, metrics, {"W": W, "F": F, "batteries": sorted(set(groups))}


# =========================
# Run (VSCode friendly)
# =========================
if __name__ == "__main__":
    set_seed(42)

    DATASET_KEY = "NASA"   

    if DATASET_KEY not in DATASET_CONFIGS:
        raise KeyError(
            f"DATASET_KEY='{DATASET_KEY}' not found in DATASET_CONFIGS.\n"
            f"Available keys: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[DATASET_KEY]

    data_path = DATA_ROOT / cfg["folder"]
    run_id = f"{DATASET_KEY}_eol{cfg['eol_threshold']}_{now_tag()}"

    BASE_DIR = Path(__file__).resolve().parent
    results_dir = ensure_dir(BASE_DIR / "results" / run_id)
    models_dir  = ensure_dir(BASE_DIR / "saved_models" / run_id)

    # 1) Cross-validation
    summary, fold_df = groupkfold_train_eval(
        data_path=data_path,
        eol_threshold=cfg["eol_threshold"],
        n_splits=cfg["n_splits"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=cfg["verbose"],
        seed = cfg.get("seed", 42),
        savgol_window=cfg.get("savgol_window", 21),
        savgol_poly=cfg.get("savgol_poly", 5),
        l2_reg=cfg.get("l2_reg", 0.0),
        early_stop_patience=cfg.get("early_stop_patience", 10),
        reduce_lr_patience=cfg.get("reduce_lr_patience", 5),
        reduce_lr_factor=cfg.get("reduce_lr_factor", 0.5),
        min_lr=cfg.get("min_lr", 1e-6),
        features=cfg.get("features", ["voltage", "dqdv", "dv_dsoc", "dtdv"]),
    )

    fold_df.to_csv(results_dir / "cv_folds.csv", index=False)
    save_json(summary, results_dir / "cv_summary.json")

    final_epochs = summary["median_best_epoch"]

    # 2) Final train on ALL data
    model, scaler, train_metrics, extra = train_final_model(
        data_path=data_path,
        eol_threshold=cfg["eol_threshold"],
        final_epochs=final_epochs,
        batch_size=cfg["batch_size"],
        verbose=cfg["verbose"],
        seed=cfg.get("seed", 42),
        savgol_window=cfg.get("savgol_window", 21),
        savgol_poly=cfg.get("savgol_poly", 5),
        l2_reg=cfg.get("l2_reg", 0.0),
        features=cfg.get("features", ["voltage", "dqdv", "dv_dsoc", "dtdv"]),
    )

    # Save model + scaler + meta
    model.save(models_dir / "rul_lstm.keras")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    meta = {
        "dataset_key": DATASET_KEY,
        "data_path": str(data_path),
        "eol_threshold": cfg["eol_threshold"],
        "features": cfg.get("features", ["voltage", "dqdv", "dv_dsoc", "dtdv"]),
        "final_epochs": int(final_epochs),
        "seed": cfg.get("seed", 42),
        "train_metrics": train_metrics,
        **extra
    }
    joblib.dump(meta, models_dir / "meta.joblib")
    save_json(meta, models_dir / "meta.json")

    print("\n=== CV Summary ===")
    print(summary)
    print("\n=== Final Train Metrics ===")
    print(train_metrics)
    print(f"\nSaved to:\n  results: {results_dir}\n  models : {models_dir}\n")

