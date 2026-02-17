from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_path(p: str) -> Path:
    """Expand and validate a path from CLI/env."""
    pp = Path(os.path.expandvars(os.path.expanduser(p))).resolve()
    if not pp.exists():
        raise FileNotFoundError(f"Path not found: {pp}")
    return pp
