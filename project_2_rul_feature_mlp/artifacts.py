# artifacts.py
import os
import json
import joblib
import pandas as pd
from datetime import datetime

class RunLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        dir_ = os.path.dirname(filepath)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"RUL REPORT\nGenerated at: {datetime.now().isoformat(timespec='seconds')}\n\n")

    def write(self, text=""):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def write_df(self, title, df):
        if title:
            self.write(title)
        self.write(df.to_string(index=False))
        self.write()

def save_rul_artifacts(out_dir, dataset_name, model, x_scaler, feat_idx, chosen_groups, eol_threshold):
    ds_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    model.save(os.path.join(ds_dir, "rul_model.keras"))
    joblib.dump(x_scaler, os.path.join(ds_dir, "x_scaler.pkl"))

    meta = {
        "dataset": dataset_name,
        "eol_threshold": float(eol_threshold),
        "chosen_groups": list(chosen_groups),
        "feat_idx": [int(i) for i in feat_idx],
        "target_type": "ratio",
    }
    with open(os.path.join(ds_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_rul_artifacts(out_dir, dataset_name):
    import numpy as np
    import tensorflow as tf

    ds_dir = os.path.join(out_dir, dataset_name)

    model = tf.keras.models.load_model(os.path.join(ds_dir, "rul_model.keras"))
    x_scaler = joblib.load(os.path.join(ds_dir, "x_scaler.pkl"))

    with open(os.path.join(ds_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_idx = np.array(meta["feat_idx"], dtype=int)
    chosen_groups = tuple(meta["chosen_groups"])
    eol_threshold = float(meta.get("eol_threshold"))

    return model, x_scaler, feat_idx, chosen_groups, eol_threshold

def save_tables_csv(out_dir, dataset_name, ablation_df, search_df):
    ds_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    ab_path = os.path.join(ds_dir, "ablation_summary.csv")
    se_path = os.path.join(ds_dir, "feature_search.csv")

    ablation_df.to_csv(ab_path, index=False)
    search_df.to_csv(se_path, index=False)
    return ab_path, se_path

def save_tables_excel(out_dir, dataset_name, ablation_df, search_df):
    ds_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    xlsx_path = os.path.join(ds_dir, "tables.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        ablation_df.to_excel(writer, sheet_name="ablation", index=False)
        search_df.to_excel(writer, sheet_name="search", index=False)

    return xlsx_path

def load_tables(out_dir, dataset_name):
    ds_dir = os.path.join(out_dir, dataset_name)
    ab_path = os.path.join(ds_dir, "ablation_summary.csv")
    se_path = os.path.join(ds_dir, "feature_search.csv")
    ablation_df = pd.read_csv(ab_path) if os.path.exists(ab_path) else None
    search_df = pd.read_csv(se_path) if os.path.exists(se_path) else None
    return ablation_df, search_df
