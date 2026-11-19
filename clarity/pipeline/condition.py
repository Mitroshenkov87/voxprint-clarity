"""The extra channel DRUNet expects. Values follow the DPIR scripts."""
from __future__ import annotations

import numpy as np

from clarity.models import ClarityModel


def deblock_level(quality_factor: float) -> float:
    """DPIR deblocking: ``(100 - quality) / 100``."""
    quality = min(100.0, max(0.0, float(quality_factor)))
    return (100.0 - quality) / 100.0


def sigma_level(sigma: float) -> float:
    """DPIR denoise: the map is ``sigma / 255``."""
    return min(255.0, max(0.0, float(sigma))) / 255.0


def noise_map(count: int, height: int, width: int, level: float) -> np.ndarray:
    """A constant noise-level channel whose batch axis matches the image.

    Spandrel's DRUNet call builds ``(1, 1, H, W)`` and concatenates it with
    the batch. A batch of N then fails with "Expected size N but got size 1".
    This map uses ``count`` so the concatenation stays on the channel axis.

    Args:
        count: Batch size. One row per frame or tile.
        height: Map height.
        width: Map width.
        level: The DPIR level written into every pixel.

    Returns:
        Float32 ``(count, 1, height, width)``.

    Raises:
        ValueError: When ``count``, ``height``, or ``width`` is below 1.
    """
    if count < 1 or height < 1 or width < 1:
        raise ValueError("the noise map needs a positive batch and size")
    return np.full((count, 1, height, width), float(level), dtype=np.float32)


def condition_nchw(frames: np.ndarray, model: ClarityModel, *, quality_factor: float, sigma: float) -> np.ndarray:
    """NHWC RGB float -> NCHW, with the model's condition channel when it has one."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("expected NHWC RGB")
    image = np.transpose(frames.astype(np.float32, copy=False), (0, 3, 1, 2))
    if model.condition == "none":
        return np.ascontiguousarray(image)
    count, _channels, height, width = image.shape
    if model.condition == "deblock":
        level = deblock_level(quality_factor)
    elif model.condition == "sigma":
        level = sigma_level(sigma)
    else:
        raise ValueError(f"unknown condition {model.condition}")
    extra = noise_map(count, height, width, level)
    return np.ascontiguousarray(np.concatenate([image, extra], axis=1))
