"""Pixel validation and a repeatable train/test split."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(labels, seed=42):
    return train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed
    )


def validate_pixels(values):
    """Accept a batch of flattened 8x8 images, each with intensities 0..16."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 64 or len(array) == 0:
        raise ValueError("Expected a nonempty array of rows, each containing 64 pixels.")
    if not np.isfinite(array).all() or (array < 0).any() or (array > 16).any():
        raise ValueError("Pixels must be finite numbers between 0 and 16 inclusive.")
    return array
