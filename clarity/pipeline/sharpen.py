"""Classical unsharp mask. ``amount`` 0 leaves the frames untouched."""
from __future__ import annotations

import numpy as np


def _box_blur(frame: np.ndarray) -> np.ndarray:
    """3x3 box blur. Edges are replicated."""
    padded = np.pad(frame, ((1, 1), (1, 1), (0, 0)), mode="edge")
    total = np.zeros_like(frame, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
    return total / 9.0


def unsharp(frame: np.ndarray, amount: float) -> np.ndarray:
    """``frame`` is HWC float. ``amount`` is 0..2."""
    if amount <= 0:
        return frame.astype(np.float32, copy=False)
    image = frame.astype(np.float32, copy=False)
    blurred = _box_blur(image)
    return np.clip(image + float(amount) * (image - blurred), 0.0, 1.0)


def sharpen_frames(frames: np.ndarray, amount: float) -> np.ndarray:
    """Run the unsharp mask on every frame.

    Args:
        frames: NHWC float RGB in 0..1.
        amount: Strength from 0 to 2. ``0`` returns the frames unchanged.

    Returns:
        NHWC float RGB in 0..1.
    """
    if amount <= 0:
        return frames.astype(np.float32, copy=False)
    return np.stack([unsharp(frame, amount) for frame in frames], axis=0)
