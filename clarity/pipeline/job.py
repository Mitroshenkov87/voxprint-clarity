"""One restore job: chunks, thermal pauses, spill, models, then ffmpeg commands.

Frame processing is real numpy. The GPU lock, VRAM numbers, and thermal rule
come from ``voxprint_engine.api`` when a hook is left as None. Tests pass
hooks so they do not need a GPU. Restoring frames still needs the engine.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from clarity.engine import (
    choose_tile as engine_choose_tile,
    empty_cuda_cache,
    halve_tile as engine_halve_tile,
    note_spill as engine_note_spill,
    restore_frames,
    sysmem_spill,
    thermal_decide,
)
from clarity.errors import ClarityError
from clarity.models import ClarityModel, clean_models, get_model, strong_model, upscale_model
from clarity.pipeline.colour import cleanup
from clarity.pipeline.condition import deblock_level, sigma_level
from clarity.pipeline.encode import (
    Container,
    VideoCodec,
    decode_command,
    encode_command,
    select_encoder,
)
from clarity.pipeline.probe import Probe, require_sdr
from clarity.pipeline.resume import ChunkRecord, pending
from clarity.pipeline.scenes import plan_chunks
from clarity.pipeline.sharpen import sharpen_frames

Infer = Callable[[ClarityModel, np.ndarray], np.ndarray]


@dataclass
class Settings:
    """Restore choices shared by the window, the CLI, and a ``.vxclarity`` file."""

    mode: str = "clean"
    upscale: bool = False
    anime: bool = False
    upscale_strength: float = 0.0
    sharpen: float = 0.35
    white_balance: bool = True
    levels: bool = True
    quality_factor: float = 40.0
    sigma: float = 15.0
    tile: int = 0
    batch: int = 2
    overlap: int = 16
    container: Container = Container.MKV
    codec: VideoCodec = VideoCodec.AV1
    copy_audio: bool = True
    max_frames: int = 1000

    def digest_payload(self) -> dict[str, Any]:
        """Return the settings fields that change the pixels or the container.

        Returns:
            A JSON-ready dictionary. The resume digest is the hash of this object.
        """
        return {
            "mode": self.mode,
            "upscale": self.upscale,
            "anime": self.anime,
            "upscale_strength": self.upscale_strength,
            "sharpen": self.sharpen,
            "white_balance": self.white_balance,
            "levels": self.levels,
            "quality_factor": self.quality_factor,
            "sigma": self.sigma,
            "tile": self.tile,
            "batch": self.batch,
            "container": self.container.value,
            "codec": self.codec.value,
            "copy_audio": self.copy_audio,
            "overlap": self.overlap,
            "max_frames": self.max_frames,
        }


@dataclass
class Hooks:
    """Callables a restore uses for the engine, the clock, and the CUDA cache.

    Tests replace these so a job can be checked without a GPU.
    """

    infer: Infer
    decide: Callable[[float, float | None, float, bool], Any] = thermal_decide
    elapsed: Callable[[], float] = lambda: 0.0
    sensors: Callable[[], tuple[float | None, float]] = lambda: (None, 0.0)
    sleep: Callable[[float], None] = lambda _seconds: None
    budget_gb: Callable[[], float] = lambda: 8.0
    choose_tile: Callable[[float, int, int], int] = engine_choose_tile
    note_spill: Callable[[int, bool], int] = engine_note_spill
    halve_tile: Callable[[int], int | None] = engine_halve_tile
    sysmem_spill: Callable[[int | None, int | None], bool] | None = None
    shared_usage: Callable[[], int | None] = lambda: None
    empty_cache: Callable[[], None] = empty_cuda_cache
    pauses: list[str] = field(default_factory=list)


def _action(decision: object) -> str:
    return str(getattr(decision, "action", decision))


def wait_for_thermal(hooks: Hooks, paused: bool, *, limit: int = 1000) -> bool:
    """Pause between chunks while the suite rule says so. Returns the new paused flag."""
    for _ in range(limit):
        median, throttle = hooks.sensors()
        decision = hooks.decide(hooks.elapsed(), median, throttle, paused)
        action = _action(decision)
        if action in {"run", "resume"}:
            return False
        if action != "pause":
            raise ClarityError(f"unexpected thermal action {action}")
        paused = True
        reason = str(getattr(decision, "reason", "pause"))
        hooks.pauses.append(reason)
        hooks.sleep(1.0)
    raise ClarityError("the GPU stayed above the thermal limit")


def _with_condition(model: ClarityModel, nchw: np.ndarray, settings: Settings) -> np.ndarray:
    if model.condition == "none":
        return nchw
    count, _channels, height, width = nchw.shape
    if model.condition == "deblock":
        level = deblock_level(settings.quality_factor)
    elif model.condition == "sigma":
        level = sigma_level(settings.sigma)
    else:
        raise ClarityError(f"unknown condition {model.condition}")
    extra = np.full((count, 1, height, width), level, dtype=np.float32)
    return np.concatenate([nchw.astype(np.float32, copy=False), extra], axis=1)


def _direct_model(frames: np.ndarray, model: ClarityModel, settings: Settings, infer: Infer, tile: int, batch: int) -> np.ndarray:
    """Apply ``infer`` to the whole chunk. Used when a test injects the network."""
    del tile, batch
    nchw = np.ascontiguousarray(np.transpose(frames.astype(np.float32, copy=False), (0, 3, 1, 2)))
    output = np.asarray(infer(model, _with_condition(model, nchw, settings)), dtype=np.float32)
    if output.ndim != 4 or output.shape[1] < 3:
        raise ClarityError(f"{model.id} must return NCHW with 3 channels, got {output.shape}")
    return np.ascontiguousarray(np.transpose(output[:, :3], (0, 2, 3, 1)))


def _run_model(frames: np.ndarray, model: ClarityModel, settings: Settings, infer: Infer, tile: int, batch: int) -> np.ndarray:
    def wrapped(nchw: np.ndarray) -> np.ndarray:
        conditioned = _with_condition(model, nchw, settings)
        output = np.asarray(infer(model, conditioned), dtype=np.float32)
        if output.ndim != 4 or output.shape[1] != 3:
            raise ClarityError(f"{model.id} must return NCHW with 3 channels, got {output.shape}")
        return output

    return np.asarray(
        restore_frames(
            frames.astype(np.float32, copy=False),
            wrapped,
            tile=tile,
            overlap=settings.overlap,
            scale=model.scale,
            batch=batch,
            length=1,
            window_overlap=0,
        )
    )


def _content_token(frames: np.ndarray) -> str:
    """A stable digest of a stride through the chunk, not a hash of every byte."""
    flat = np.ascontiguousarray(frames.reshape(-1))
    step = max(1, flat.size // 4096)
    return hashlib.sha256(flat[::step].astype(np.float32).tobytes()).hexdigest()


def downscale_half(frames: np.ndarray) -> np.ndarray:
    """Area-average 2x downscale. 4x compact output becomes the optional 2x result."""
    count, height, width, channels = frames.shape
    even_h = height - (height % 2)
    even_w = width - (width % 2)
    image = frames[:, :even_h, :even_w]
    return image.reshape(count, even_h // 2, 2, even_w // 2, 2, channels).mean(axis=(2, 4)).astype(np.float32)


def models_for(settings: Settings) -> list[ClarityModel]:
    """List the catalogue models a job will run, in order.

    Args:
        settings: Mode, upscale switch, and anime switch.

    Returns:
        The clean pair or the strong net, then the upscale nets when upscale is on.

    Raises:
        ClarityError: When ``mode`` is neither ``clean`` nor ``strong``.
    """
    if settings.mode == "clean":
        chosen = list(clean_models())
    elif settings.mode == "strong":
        chosen = [strong_model()]
    else:
        raise ClarityError("mode must be clean or strong")
    if settings.upscale:
        chosen.append(upscale_model(anime=settings.anime))
        if settings.upscale_strength > 0 and not settings.anime:
            chosen.append(get_model("realesr-general-wdn-x4v3"))
    return chosen


def process_frames(
    frames: np.ndarray,
    settings: Settings,
    infer: Infer,
    *,
    tile: int,
    batch: int,
    use_engine: bool = True,
) -> np.ndarray:
    """Run the selected models, then classical colour and sharpen, on one chunk.

    ``use_engine`` sends each model through ``voxprint_engine`` tiling. A test
    control leaves it false and applies ``infer`` to the whole frame.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ClarityError("frames must be NHWC RGB")
    step = _run_model if use_engine else _direct_model
    current = frames.astype(np.float32, copy=False)
    if settings.mode == "clean":
        deblock, denoise = clean_models()
        current = step(current, deblock, settings, infer, tile, batch)
        current = step(current, denoise, settings, infer, tile, batch)
    elif settings.mode == "strong":
        current = step(current, strong_model(), settings, infer, tile, batch)
    else:
        raise ClarityError("mode must be clean or strong")
    if settings.upscale:
        primary = upscale_model(anime=settings.anime)
        upscaled = step(current, primary, settings, infer, tile, batch)
        strength = 0.0 if settings.anime else min(1.0, max(0.0, settings.upscale_strength))
        if strength > 0:
            twin = step(current, get_model("realesr-general-wdn-x4v3"), settings, infer, tile, batch)
            upscaled = (1.0 - strength) * upscaled + strength * twin
        current = downscale_half(upscaled)
    current = cleanup(current, white_balance=settings.white_balance, levels=settings.levels)
    return sharpen_frames(current, settings.sharpen)


def _tile_for(settings: Settings, hooks: Hooks) -> int:
    if settings.tile > 0:
        return settings.tile
    return int(hooks.choose_tile(float(hooks.budget_gb()), 0, 0))


def restore_ranges(
    ranges: list[tuple[int, int]],
    done: dict[int, Any],
    load_range: Callable[[int, int], np.ndarray],
    settings: Settings,
    hooks: Hooks,
) -> list[ChunkRecord]:
    """Restore each unfinished chunk. Re-measures, pauses, and halves the batch on a spill."""
    if hooks.sysmem_spill is None:
        hooks.sysmem_spill = sysmem_spill
    finished: list[ChunkRecord] = []
    paused = False
    batch = max(1, settings.batch)
    baseline = hooks.shared_usage()
    for index, start, stop in pending(ranges, done):
        paused = wait_for_thermal(hooks, paused)
        tile = _tile_for(settings, hooks)
        usage = hooks.shared_usage()
        if hooks.sysmem_spill(usage, baseline):
            hooks.empty_cache()
            batch = int(hooks.note_spill(batch, True))
            smaller = hooks.halve_tile(tile)
            if smaller is not None:
                tile = int(smaller)
            baseline = usage
        frames = load_range(start, stop)
        restored = process_frames(frames, settings, hooks.infer, tile=tile, batch=batch)
        finished.append(ChunkRecord(index, start, stop, _content_token(restored)))
    return finished


def commands_for(
    probe: Probe,
    settings: Settings,
    *,
    ffmpeg: str,
    source: str,
    destination: str,
    encoders: set[str],
) -> dict[str, Any]:
    """The ffmpeg commands for this probe. HDR is refused here."""
    require_sdr(probe)
    encoder = select_encoder(settings.codec, encoders)
    width = probe.width * (2 if settings.upscale else 1)
    height = probe.height * (2 if settings.upscale else 1)
    return {
        "decode": decode_command(ffmpeg, source, deinterlace=probe.interlaced),
        "encode": encode_command(
            ffmpeg,
            source,
            container=settings.container,
            encoder=encoder,
            destination=destination,
            copy_audio=settings.copy_audio,
            width=width,
            height=height,
            fps=probe.fps,
        ),
        "chunks": [list(pair) for pair in plan_chunks(max(probe.frames, 1), [], settings.max_frames)],
    }
