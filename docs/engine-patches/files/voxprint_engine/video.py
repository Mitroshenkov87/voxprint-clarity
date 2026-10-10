"""Additive video helpers for ``voxprint_engine`` (API stays 1).

Clarity imports this module. It has no Qt import and does not open a device
on import. Image tiling in ``voxprint_engine.tile`` is unchanged; the feather
here matches that file at enlarge commit 6d00d44.
"""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray
Infer = Callable[[Array], Array]
SPILL_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class TemporalWindow:
    """One model window.

    ``input_start`` is the source index of window slot 0 and may be negative.
    Slots before 0 and at or past ``frame_count`` are padding. ``keep_start``
    and ``keep_stop`` are slots whose outputs are real frames. Across the
    list, those frames cover ``0 .. frame_count`` once.
    """

    index: int
    input_start: int
    length: int
    overlap: int
    pad_left: int
    pad_right: int
    keep_start: int
    keep_stop: int
    frame_count: int


@dataclass(frozen=True)
class RestoreBatchRequest:
    """JSON line ``op`` is ``restore_batch``. Sent to the engine worker."""

    id: str
    frames_npy: str
    out_npy: str
    onnx: str
    tile: int = 0
    overlap: int = 16
    batch: int = 1
    scale: int = 1
    length: int = 1
    window_overlap: int = 0
    device: str = "auto"
    owner: str = "clarity"

    def to_message(self) -> dict[str, Any]:
        return {
            "op": "restore_batch",
            "id": self.id,
            "frames_npy": self.frames_npy,
            "out_npy": self.out_npy,
            "onnx": self.onnx,
            "tile": self.tile,
            "overlap": self.overlap,
            "batch": self.batch,
            "scale": self.scale,
            "length": self.length,
            "window_overlap": self.window_overlap,
            "device": self.device,
            "owner": self.owner,
        }


def temporal_windows(frame_count: int, length: int, overlap: int) -> list[TemporalWindow]:
    """Slide a window of ``length`` with ``overlap`` frames of left context.

    ``length`` 1 and ``overlap`` 0 is the frame-wise case: one window per
    frame and nothing is discarded. A longer window pads with edge frames
    outside the clip. The kept slots of each window are the frames that have
    not already been emitted.
    """
    count = int(frame_count)
    span = int(length)
    context = int(overlap)
    if count < 1:
        raise ValueError("frame_count must be positive")
    if span < 1:
        raise ValueError("length must be positive")
    if context < 0 or context >= span:
        raise ValueError("overlap must be >= 0 and < length")
    step = span - context
    windows: list[TemporalWindow] = []
    index = 0
    emit = 0
    while emit < count:
        stop = min(count, emit + step)
        input_start = emit - context
        pad_left = max(0, -input_start)
        pad_right = max(0, input_start + span - count)
        keep_count = stop - emit
        keep_start = context
        keep_stop = keep_start + keep_count
        if keep_stop > span - pad_right:
            raise ValueError("window keep range ran into padding")
        windows.append(
            TemporalWindow(
                index=index,
                input_start=input_start,
                length=span,
                overlap=context,
                pad_left=pad_left,
                pad_right=pad_right,
                keep_start=keep_start,
                keep_stop=keep_stop,
                frame_count=count,
            )
        )
        index += 1
        emit = stop
    return windows


def gather_window(frames: Array, window: TemporalWindow) -> Array:
    """``T,H,W,C`` (or ``T,...``) for one window, edge-padded."""
    if frames.ndim < 1 or frames.shape[0] != window.frame_count:
        raise ValueError("frames do not match the window")
    slots: list[Array] = []
    last = window.frame_count - 1
    for slot in range(window.length):
        source = window.input_start + slot
        source = min(last, max(0, source))
        slots.append(frames[source])
    return np.stack(slots, axis=0)


def to_nchw(frames_nhwc: Array) -> Array:
    """``N,H,W,C`` float -> contiguous ``N,C,H,W``."""
    if frames_nhwc.ndim != 4:
        raise ValueError("expected NHWC")
    return np.ascontiguousarray(np.transpose(frames_nhwc, (0, 3, 1, 2)))


def to_nhwc(batch_nchw: Array) -> Array:
    """``N,C,H,W`` -> ``N,H,W,C``."""
    if batch_nchw.ndim != 4:
        raise ValueError("expected NCHW")
    return np.transpose(batch_nchw, (0, 2, 3, 1))


def _session_dtype(session: object) -> np.dtype[np.generic]:
    inputs = getattr(session, "get_inputs", None)
    if not callable(inputs):
        return np.dtype(np.float32)
    specs = inputs()
    if not specs:
        return np.dtype(np.float32)
    kind = str(getattr(specs[0], "type", ""))
    if "float16" in kind:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def run_batch(session: object, batch: Array) -> Array:
    """Run one NCHW batch. ``session`` is an ONNX Runtime session or a double.

    The returned array is float32 NCHW. A fp16 session receives fp16 input.
    """
    if batch.ndim != 4:
        raise ValueError("batch must be NCHW")
    array = np.ascontiguousarray(batch.astype(_session_dtype(session), copy=False))
    getter = getattr(session, "get_inputs", None)
    if not callable(getter):
        raise ValueError("session has no inputs")
    specs = getter()
    if not specs:
        raise ValueError("session has no inputs")
    name = str(getattr(specs[0], "name", ""))
    runner = getattr(session, "run", None)
    if not callable(runner):
        raise ValueError("session has no run")
    try:
        result = runner(None, {name: array})
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not result:
        raise RuntimeError("the network returned nothing")
    output = np.asarray(result[0])
    if output.ndim != 4:
        raise RuntimeError(f"unexpected network output shape {output.shape}")
    return output.astype(np.float32, copy=False)


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last = length - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


@dataclass(frozen=True)
class _Box:
    x: int
    y: int
    width: int
    height: int


def _plan(width: int, height: int, tile: int, overlap: int) -> list[_Box]:
    if width < 1 or height < 1:
        raise ValueError("the frame is empty")
    if tile < 1:
        raise ValueError("tile must be positive")
    overlap = max(0, min(int(overlap), int(tile) // 2))
    stride = max(1, int(tile) - overlap)
    boxes: list[_Box] = []
    for y in _starts(height, tile, stride):
        for x in _starts(width, tile, stride):
            boxes.append(_Box(x, y, min(tile, width - x), min(tile, height - y)))
    return boxes


def _feather(
    height: int,
    width: int,
    overlap: int,
    *,
    fade_left: bool,
    fade_right: bool,
    fade_top: bool,
    fade_bottom: bool,
) -> Array:
    weight_y = np.ones(height, dtype=np.float32)
    weight_x = np.ones(width, dtype=np.float32)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / float(overlap)
        if fade_top and overlap <= height:
            weight_y[:overlap] *= ramp
        if fade_bottom and overlap <= height:
            weight_y[-overlap:] *= ramp[::-1]
        if fade_left and overlap <= width:
            weight_x[:overlap] *= ramp
        if fade_right and overlap <= width:
            weight_x[-overlap:] *= ramp[::-1]
    return weight_y[:, None] * weight_x[None, :]


def tile_frames(
    frames: Array,
    infer: Infer,
    tile: int,
    overlap: int,
    *,
    scale: int = 1,
    batch: int = 1,
) -> Array:
    """Per-frame tiles, batched across N.

    ``frames`` is NHWC float. ``infer`` receives NCHW and returns NCHW at
    ``scale`` times the tile. Edge tiles are padded by edge replication and
    the padding is cropped off the network output. Interior overlaps are
    feathered so the two weights sum to 1.
    """
    if frames.ndim != 4:
        raise ValueError("expected NHWC")
    if scale < 1:
        raise ValueError("scale must be >= 1")
    count, height, width, channels = frames.shape
    boxes = _plan(width, height, tile, overlap)
    out_h, out_w = height * scale, width * scale
    acc = np.zeros((count, out_h, out_w, channels), dtype=np.float32)
    cover = np.zeros((count, out_h, out_w, 1), dtype=np.float32)
    group = max(1, int(batch))
    pending_img: list[Array] = []
    pending_where: list[tuple[int, _Box]] = []

    def flush() -> None:
        if not pending_img:
            return
        stacked = np.stack(pending_img, axis=0)
        produced = np.asarray(infer(to_nchw(stacked)), dtype=np.float32)
        if produced.shape[0] != stacked.shape[0]:
            raise ValueError("the network changed the batch count")
        for sample, (frame_index, box) in enumerate(pending_where):
            tile_out = to_nhwc(produced[sample : sample + 1])[0]
            # The network sees the padded tile. Crop back to the real box.
            crop_h = box.height * scale
            crop_w = box.width * scale
            if tile_out.shape[0] < crop_h or tile_out.shape[1] < crop_w or tile_out.shape[2] != channels:
                raise ValueError(
                    f"the network returned {tile_out.shape}, expected at least {(crop_h, crop_w, channels)}"
                )
            tile_out = tile_out[:crop_h, :crop_w]
            weight = _feather(
                tile_out.shape[0],
                tile_out.shape[1],
                overlap * scale,
                fade_left=box.x > 0,
                fade_right=box.x + box.width < width,
                fade_top=box.y > 0,
                fade_bottom=box.y + box.height < height,
            )
            y = box.y * scale
            x = box.x * scale
            acc[frame_index, y : y + crop_h, x : x + crop_w] += tile_out * weight[..., None]
            cover[frame_index, y : y + crop_h, x : x + crop_w] += weight[..., None]
        pending_img.clear()
        pending_where.clear()

    for frame_index in range(count):
        image = frames[frame_index]
        for box in boxes:
            crop = image[box.y : box.y + box.height, box.x : box.x + box.width]
            pad_h = max(0, tile - box.height)
            pad_w = max(0, tile - box.width)
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            pending_img.append(crop)
            pending_where.append((frame_index, box))
            if len(pending_img) >= group:
                flush()
    flush()
    return np.clip(acc / np.maximum(cover, 1e-6), 0.0, 1.0)


def restore_frames(
    frames: Array,
    infer: Infer,
    *,
    tile: int,
    overlap: int,
    scale: int = 1,
    batch: int = 1,
    length: int = 1,
    window_overlap: int = 0,
) -> Array:
    """Restore ``frames`` (NHWC). Frame-wise models use the defaults.

    A temporal window is gathered, run as an NCHW batch through ``infer``
    (one source frame per batch row, padding included), and only the kept
    rows are written back. Padding rows are discarded.
    """
    count = int(frames.shape[0])
    windows = temporal_windows(count, length, window_overlap if length > 1 else 0)
    if length <= 1:
        return tile_frames(frames, infer, tile, overlap, scale=scale, batch=batch)
    output: Array | None = None
    for window in windows:
        gathered = gather_window(frames, window)
        restored = tile_frames(gathered, infer, tile, overlap, scale=scale, batch=batch)
        kept = restored[window.keep_start : window.keep_stop]
        if output is None:
            output = np.zeros((count, kept.shape[1], kept.shape[2], kept.shape[3]), dtype=np.float32)
        start = window.input_start + window.keep_start
        output[start : start + kept.shape[0]] = kept
    if output is None:
        raise ValueError("no frames")
    return output


def sysmem_spill(
    shared_now: int | None,
    shared_baseline: int | None,
    threshold: int = SPILL_BYTES,
) -> bool:
    """True when GPU-process shared (system) memory grew by more than ``threshold``."""
    if shared_now is None or shared_baseline is None:
        return False
    return int(shared_now) - int(shared_baseline) > int(threshold)


_COUNTER_NUMBER = re.compile(r"(\d+)\s*$")


def parse_shared_usage(text: str) -> int | None:
    """Pull the last integer out of a Windows GPU-process counter sample."""
    found: int | None = None
    for line in text.splitlines():
        match = _COUNTER_NUMBER.search(line.strip().strip('"'))
        if match:
            found = int(match.group(1))
    return found


def read_shared_usage(pid: int | None = None) -> int | None:
    """Bytes of shared GPU memory for ``pid``. None when the counter is unavailable.

    The suite rule watches ``GPU Process Memory(pid_*)\\Shared Usage`` on
    Windows. Other platforms, and a missing counter, return None so the
    caller does not invent a spill.
    """
    if os.name != "nt":
        return None
    number = os.getpid() if pid is None else int(pid)
    counter = f"\\GPU Process Memory(pid_{number}*)\\Shared Usage"
    try:
        proc = subprocess.run(
            ["typeperf", counter, "-sc", "1"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return parse_shared_usage(proc.stdout or "")
