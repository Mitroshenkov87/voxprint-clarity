"""Per-scene colour cleanup. One gain for the whole chunk, so the picture does not pump."""
from __future__ import annotations

import numpy as np


def grey_world(frames: np.ndarray) -> np.ndarray:
    """Scale each channel so the chunk's mean matches the mean of the three means.

    ``frames`` is NHWC float in 0..1. A channel with no energy is left alone.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("expected NHWC RGB")
    image = frames.astype(np.float32, copy=False)
    means = image.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    scale = np.ones(3, dtype=np.float32)
    for channel in range(3):
        if means[channel] > 1e-6:
            scale[channel] = target / float(means[channel])
    return np.clip(image * scale.reshape(1, 1, 1, 3), 0.0, 1.0)


def percentile_stretch(frames: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Stretch each channel using one percentile pair measured on the whole chunk."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("expected NHWC RGB")
    image = frames.astype(np.float32, copy=False)
    flat = image.reshape(-1, 3)
    lo = np.percentile(flat, low, axis=0).astype(np.float32)
    hi = np.percentile(flat, high, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, 1e-6)
    stretched = (image - lo.reshape(1, 1, 1, 3)) / span.reshape(1, 1, 1, 3)
    return np.clip(stretched, 0.0, 1.0)


def cleanup(frames: np.ndarray, *, white_balance: bool, levels: bool) -> np.ndarray:
    result = frames
    if white_balance:
        result = grey_world(result)
    if levels:
        result = percentile_stretch(result)
    return result
