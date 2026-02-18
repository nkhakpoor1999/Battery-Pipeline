import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = r"artifacts_3/Lab-Li-EVE/pred_eval_EVE_8_B.csv"
OUT_DIR  = r"artifacts_3/Lab-Li-EVE"

os.makedirs(OUT_DIR, exist_ok=True)

ABL_CSV = r"artifacts_3/Lab-Li-EVE/ablation_summary.csv"
OUT_DIR = r"artifacts_3/Lab-Li-EVE"

ab = pd.read_csv(ABL_CSV)


df = pd.read_csv(PRED_CSV)

required_cols = {"cycle", "rul_true_ratio", "rul_pred_ratio"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in pred csv: {missing}. Found: {list(df.columns)}")

cycle = df["cycle"].to_numpy() #FOR OXFORD: * 100
y_true = df["rul_true_ratio"].to_numpy()
y_pred = df["rul_pred_ratio"].to_numpy()
err = y_pred - y_true
abs_err = np.abs(err)

# --- Plot 1: True vs Pred over cycles (line plot)
plt.figure()
plt.plot(cycle, y_true, label="True RUL ", color='black')
plt.scatter(cycle, y_pred, label="Pred RUL ")
plt.xlabel("Cycle")
plt.ylabel("RUL (ratio)")
plt.title("True vs Pred RUL For Unseen Battery (Lab-Li-EVE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "true_vs_pred_rul_ratio.png")
plt.savefig(p1, dpi=200)
plt.close()


print("Saved:")
print(p1)


required_cols = {"setting", "R2_mean"}
missing = required_cols - set(ab.columns)
if missing:
    raise ValueError(f"Missing columns in ablation csv: {missing}. Found: {list(ab.columns)}")

ab_plot = ab.sort_values("R2_mean", ascending=True)

plt.figure(figsize=(10, 5))
plt.bar(ab_plot["setting"].astype(str), ab_plot["R2_mean"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("R2_mean")
plt.title("Ablation Results (GroupKFold Mean R2) - Lab-Li-EVE")
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ablation_r2_mean_bar.png")
plt.savefig(out_path, dpi=200)
plt.close()

print("Saved:", out_path)
