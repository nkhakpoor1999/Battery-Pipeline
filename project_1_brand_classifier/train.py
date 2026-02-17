from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, log_loss,
    confusion_matrix, classification_report
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from config import (
    LABELS, RANDOM_SEED, FS, CUTOFF, LOWPASS_ORDER,
    SAVGOL_WINDOW, SAVGOL_POLY, DEFAULT_ARTIFACTS_DIR
)
from utils import set_seed, ensure_dir, save_json, require_path
from dataset import build_brand_dataset
from model import build_brand_model


def make_class_weight(y: np.ndarray, power: float = 0.5) -> dict:
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    w = np.power(w, power)
    return dict(zip(classes, w))


def save_artifacts(out_dir: Path, model, scaler, id_to_name: dict, metrics: dict, cm: np.ndarray, cm_norm: np.ndarray):
    out_dir = ensure_dir(out_dir)

    model_path = out_dir / "brand_model.keras"
    model.save(model_path)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    save_json({str(k): v for k, v in id_to_name.items()}, out_dir / "id_to_name.json")
    save_json(metrics, out_dir / "metrics.json")

    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm_norm).to_csv(out_dir / "confusion_matrix_norm.csv", index=False)

    print("\nSaved to:", out_dir.resolve())
    print(" -", model_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to database folder containing *.npz")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--weight_power", type=float, default=0.5)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = require_path(args.data_dir)
    out_dir = Path(args.out_dir)

    X, y, groups, id_to_name = build_brand_dataset(
        database_dir=data_dir,
        labels=LABELS,
        fs=FS,
        cutoff=CUTOFF,
        lowpass_order=LOWPASS_ORDER,
        savgol_window=SAVGOL_WINDOW,
        savgol_poly=SAVGOL_POLY,
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    num_classes = int(len(np.unique(y)))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    class_weight = make_class_weight(y_train, power=args.weight_power)

    model = build_brand_model(input_dim=X_train_sc.shape[1], num_classes=num_classes, l2_reg=args.l2_reg)

    early_stop = EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0)

    model.fit(
        X_train_sc, y_train_cat,
        validation_data=(X_val_sc, y_val_cat),
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    val_probs = model.predict(X_val_sc, verbose=0)
    val_pred = val_probs.argmax(axis=1)

    all_labels = list(range(num_classes))
    target_names = [id_to_name.get(i, f"Class-{i}") for i in all_labels]

    acc = accuracy_score(y_val, val_pred)
    macro_f1 = f1_score(y_val, val_pred, average="macro")
    weighted_f1 = f1_score(y_val, val_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_val, val_pred)
    ll = log_loss(y_val, val_probs, labels=all_labels)

    cm = confusion_matrix(y_val, val_pred, labels=all_labels)
    den = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), den, out=np.zeros_like(cm, dtype=float), where=den != 0)
    cm_norm = np.nan_to_num(cm_norm)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc),
        "log_loss": float(ll),
        "labels": all_labels,
        "target_names": target_names,
        "train_batteries": int(len(np.unique(groups[train_idx]))),
        "val_batteries": int(len(np.unique(groups[val_idx]))),
        "train_cycles": int(len(train_idx)),
        "val_cycles": int(len(val_idx)),
        "seed": int(args.seed),
    }

    print("\nValidation metrics:")
    print(metrics)
    print("\nClassification report:")
    print(classification_report(y_val, val_pred, labels=all_labels, target_names=target_names, digits=4, zero_division=0))

    save_artifacts(out_dir, model, scaler, id_to_name, metrics, cm, cm_norm)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend([
            "--data_dir", r"D:\Uni\Thesis\code & data\Battery\Battery Project\database",
            "--out_dir", r"artifacts_brand"
        ])

    main()

