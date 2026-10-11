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


def _histogram_percentile(hist: np.ndarray, total: int, percent: float) -> float:
    """The smallest bin whose cumulative count reaches ``percent`` of ``total``."""
    if total <= 0:
        return 0.0
    target = (percent / 100.0) * float(total)
    running = 0
    for value, count in enumerate(hist.tolist()):
        running += int(count)
        if running >= target:
            return float(value)
    return 255.0


def uint8_colour_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and the 1st and 99th percentile of an NHWC uint8 clip.

    One frame is read at a time, so a long chunk does not become a second RAM copy.
    The histogram is exact for 8-bit values.

    Args:
        frames: NHWC uint8, including a memory map.

    Returns:
        Channel means and the low and high percentiles, each in 0..1.
    """
    count = int(frames.shape[0])
    sums = np.zeros(3, dtype=np.float64)
    hist = np.zeros((3, 256), dtype=np.int64)
    pixels = 0
    for index in range(count):
        flat = np.ascontiguousarray(frames[index]).reshape(-1, 3)
        sums += flat.sum(axis=0, dtype=np.float64)
        pixels += int(flat.shape[0])
        for channel in range(3):
            hist[channel] += np.bincount(flat[:, channel], minlength=256)
    if pixels < 1:
        ones = np.ones(3, dtype=np.float32)
        return ones * 0.5, np.zeros(3, dtype=np.float32), ones
    means = (sums / float(pixels) / 255.0).astype(np.float32)
    lo = np.zeros(3, dtype=np.float32)
    hi = np.ones(3, dtype=np.float32)
    for channel in range(3):
        lo[channel] = _histogram_percentile(hist[channel], pixels, 1.0) / 255.0
        hi[channel] = _histogram_percentile(hist[channel], pixels, 99.0) / 255.0
    return means, lo, hi


def apply_colour_frame(
    frame: np.ndarray,
    means: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    white_balance: bool,
    levels: bool,
) -> np.ndarray:
    """Apply one chunk's colour gain to a single HWC float frame.

    Args:
        frame: HWC float RGB in 0..1.
        means: Per-channel means in 0..1.
        lo: Low percentile in 0..1.
        hi: High percentile in 0..1.
        white_balance: Apply the grey-world gain when true.
        levels: Apply the percentile stretch when true.

    Returns:
        HWC float RGB in 0..1.
    """
    image = frame.astype(np.float32, copy=False)
    if white_balance:
        target = float(means.mean())
        scale = np.ones(3, dtype=np.float32)
        for channel in range(3):
            if float(means[channel]) > 1e-6:
                scale[channel] = target / float(means[channel])
        image = np.clip(image * scale.reshape(1, 1, 3), 0.0, 1.0)
    if levels:
        span = np.maximum(hi - lo, 1e-6)
        image = np.clip((image - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3), 0.0, 1.0)
    return image


def cleanup(frames: np.ndarray, *, white_balance: bool, levels: bool) -> np.ndarray:
    """Apply the colour steps that are switched on.

    Args:
        frames: NHWC float RGB in 0..1.
        white_balance: Run grey-world when true.
        levels: Run the percentile stretch when true.

    Returns:
        The frames after the selected steps, still in 0..1.
    """
    result = frames
    if white_balance:
        result = grey_world(result)
    if levels:
        result = percentile_stretch(result)
    return result
