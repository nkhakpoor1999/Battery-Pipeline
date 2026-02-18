# configs.py

from pathlib import Path

DATA_ROOT = Path(r"D:\Uni\Thesis\code & data\Battery\Battery Project")

DATASET_CONFIGS = {
    "MIT": {
        "folder": "MIT",
        "dataset_name": "MIT",
        "out_dir": "artifacts_3",
        "eol_threshold": 0.85,
        "n_splits": 5,
        "epochs": 100,
        "batch_size": 64,
        "verbose": 0,
    },

    "NASA": {
        "folder": "NASA",
        "dataset_name": "NASA",
        "out_dir": "artifacts_3",
        "eol_threshold": 0.80,
        "n_splits": 3,
        "epochs": 100,
        "batch_size": 64,
        "verbose": 0,
    },

    "OXFORD": {
        "folder": "OXFORD",
        "dataset_name": "OXFORD",
        "out_dir": "artifacts_3",
        "eol_threshold": 0.80,
        "n_splits": 5,
        "epochs": 100,
        "batch_size": 64,
        "verbose": 0,
    },

    "Lab-Li-LCO": {
        "folder": "Lab-Li-LCO",
        "dataset_name": "Lab-Li-LCO",
        "out_dir": "artifacts_3",
        "eol_threshold": 0.85,
        "n_splits": 4,
        "epochs": 100,
        "batch_size": 64,
        "verbose": 0,
    },

    "Lab-Li-NMC": {
        "folder": "Lab-Li-NMC",
        "dataset_name": "Lab-Li-NMC",
        "out_dir": "artifacts_3",
        "eol_threshold": 0.80,
        "n_splits": 3,
        "epochs": 100,
        "batch_size": 64,
        "verbose": 0,
    },

    "Lab-Li-EVE": {
        "folder": "Lab-Li-EVE",
        "dataset_name": "Lab-Li-EVE",
        "out_dir": "artifacts_3",
        "eol_threshold": 0.80,
        "n_splits": 3,
        "epochs": 100,
        "batch_size": 64,
        "verbose": 0,
    },
}
