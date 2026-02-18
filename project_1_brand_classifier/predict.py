from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from utils import load_json, require_path
from dataset import dvdsoc


def brand_features_for_new(filepath: Path, scaler, eps: float = 1e-9) -> np.ndarray:
    d = np.load(filepath, allow_pickle=True)
    V = np.asarray(d["voltage"], dtype=float)
    SOC = np.asarray(d["soc"], dtype=float)

    dVdS = dvdsoc(V, SOC, eps=eps)

    X_list = []
    for c in range(V.shape[0]):
        v_c = V[c]
        dvds_c = dVdS[c]
        X_list.append([
            float(np.nanmean(v_c)),  float(np.nanstd(v_c)),  float(np.nanmax(v_c)),
            float(np.nanmean(dvds_c)), float(np.nanstd(dvds_c)), float(np.nanmax(dvds_c)),
        ])

    X = np.asarray(X_list, dtype=float)
    return scaler.transform(X)


def load_artifacts(artifacts_dir: Path):
    artifacts_dir = Path(artifacts_dir)

    model = load_model(artifacts_dir / "brand_model.keras")
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    id_to_name_raw = load_json(artifacts_dir / "id_to_name.json")
    id_to_name = {int(k): v for k, v in id_to_name_raw.items()}
    metrics = load_json(artifacts_dir / "metrics.json")
    return model, scaler, id_to_name, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, required=True)
    parser.add_argument("--file", type=str, required=True, help="Path to a single .npz file")
    args = parser.parse_args()

    artifacts_dir = require_path(args.artifacts_dir)
    file_path = require_path(args.file)

    model, scaler, id_to_name, _ = load_artifacts(artifacts_dir)

    X_new = brand_features_for_new(file_path, scaler)
    probs = model.predict(X_new, verbose=0)

    mean_probs = probs.mean(axis=0)
    final_id = int(mean_probs.argmax())
    final_label = id_to_name.get(final_id, str(final_id))
    final_conf = float(mean_probs[final_id])

    print("\nBattery class probabilities (avg over cycles):")
    for cid, p in enumerate(mean_probs):
        print(f"  {id_to_name.get(cid, f'Class-{cid}')}: {p:.4f}")

    print(f"\nFinal class: {final_label} (confidence={final_conf:.4f})")

    class_ids = probs.argmax(axis=1)
    class_names = [id_to_name.get(int(i), f"Class-{int(i)}") for i in class_ids]
    print("\nPer-cycle summary:", Counter(class_names))

    print("\nPer-cycle summary:", Counter(class_names))

    # -------- Save results to file --------
    out_dir = Path("outputs_brand")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / f"pred_{file_path.stem}.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("Battery class probabilities (avg over cycles):\n")
        for cid, p in enumerate(mean_probs):
            f.write(f"  {id_to_name.get(cid, f'Class-{cid}')}: {p:.4f}\n")

        f.write(f"\nFinal class: {final_label} (confidence={final_conf:.4f})\n")
        f.write(f"\nPer-cycle summary: {Counter(class_names)}\n")

    print(f"\nSaved results to: {out_file.resolve()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--artifacts_dir", r"artifacts_brand",
            "--file", r"D:\Uni\Thesis\code & data\Battery\Battery Project\new\EVE_8_B.npz",  #new battery
    ])
        
        
    main()