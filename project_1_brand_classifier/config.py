from __future__ import annotations
from pathlib import Path

# Default settings (can be overridden from CLI)
RANDOM_SEED = 42

# Feature extraction params
FS = 1.0
CUTOFF = 0.05
LOWPASS_ORDER = 3

SAVGOL_WINDOW = 21
SAVGOL_POLY = 5

# Output
DEFAULT_ARTIFACTS_DIR = Path("artifacts_brand")

LABELS = {
    "NASA":   ("NASA", 0),
    "OXFORD": ("OXFORD", 1),
    "MIT":    ("MIT", 2),

    "V44":  ("Lab-Li-LCO", 3),
    "V45":  ("Lab-Li-LCO", 3),
    "V43C": ("Lab-Li-LCO", 3),

    "V42":  ("Lab-Li-NMC", 4),
    "V435": ("Lab-Li-NMC", 4),

    "EVE": ("Lab-Li-EVE", 5),
}
