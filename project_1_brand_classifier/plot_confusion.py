from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cm_path = Path("artifacts_brand/confusion_matrix_norm.csv")

cm = pd.read_csv(cm_path).values

class_names = ["NASA", "OXFORD", "MIT", "Lab-Li-LCO", "Lab-Li-NMC", "Lab-Li-EVE"] 
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.colorbar()
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 ha="center", va="center")

plt.tight_layout()

out_path = Path("outputs_brand/cm_norm_heatmap.png")
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=200)
plt.close()

print("Saved to:", out_path.resolve())
