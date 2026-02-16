# utils.py
import numpy as np
import tensorflow as tf
from config import RANDOM_SEED

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def to_np(arr):
    arr = np.array(arr, dtype=object)
    return np.vstack([np.asarray(a, dtype=float) for a in arr])
