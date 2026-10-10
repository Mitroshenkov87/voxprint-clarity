"""Run one video through decode, restore, and encode.

The GPU lock, VRAM budget, thermal rule, and weight fetch go through
``clarity.engine_shim``. Tests pass a ``Control`` so they can run without a
GPU, ffmpeg, or the private engine.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO

import numpy as np

from clarity.engine_shim import (
    choose_tile,
    empty_cuda_cache,
    ensure_weight,
    gpu_job,
    halve_tile,
    measure,
    note_spill,
    thermal_decide,
    video_module,
)
from clarity.errors import Cancelled, ClarityError
from clarity.models import ClarityModel
from clarity.pipeline.encode import parse_encoder_list
from clarity.pipeline.job import Hooks, Settings, commands_for, models_for, process_frames, wait_for_thermal
from clarity.pipeline.probe import parse_probe, probe_command, require_sdr

try:
    import torch
    from spandrel import ModelLoader
except ImportError:
    torch = None
    ModelLoader = None

Proc = Any
Progress = Callable[[float, str], None]


class CancelToken:
    """Set from the UI thread. ``cancel`` also kills ffmpeg so a blocked read returns."""

    def __init__(self) -> None:
        self._flag = threading.Event()
        self._procs: list[Proc] = []
        self._guard = threading.Lock()

    def is_set(self) -> bool:
        return self._flag.is_set()

    def watch(self, proc: Proc) -> None:
        with self._guard:
            self._procs.append(proc)
        if self._flag.is_set():
            _kill(proc)

    def cancel(self) -> None:
        self._flag.set()
        with self._guard:
            procs = list(self._procs)
        for proc in procs:
            _kill(proc)


@dataclass
class JobSpec:
    source: Path
    destination: Path
    settings: Settings
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass
class Control:
    infer: Callable[[ClarityModel, np.ndarray], np.ndarray]
    lock: Callable[[str, float], AbstractContextManager[None]]
    spawn: Callable[[list[str]], Proc]
    run_text: Callable[[list[str]], str]
    token: CancelToken = field(default_factory=CancelToken)
    progress: Progress = lambda _fraction, _text: None
    device: bool = False


def _kill(proc: Proc) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except OSError:
        return


def _spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    kwargs: dict[str, Any] = {"stderr": subprocess.PIPE}
    if "pipe:1" in argv:
        kwargs["stdout"] = subprocess.PIPE
    else:
        kwargs["stdin"] = subprocess.PIPE
        kwargs["stdout"] = subprocess.DEVNULL
    if os.name == "nt":
        kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return subprocess.Popen(argv, **kwargs)


def _run_text(argv: list[str]) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClarityError(f"{argv[0]} failed to start: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "command failed").strip()
        raise ClarityError(detail)
    return proc.stdout


def _prepare_infer(settings: Settings) -> Callable[[ClarityModel, np.ndarray], np.ndarray]:
    """Fetch weights through the engine store, then run them with spandrel on CUDA."""
    paths = {model.id: ensure_weight(model.manifest()) for model in models_for(settings)}
    loaded: dict[str, Any] = {}

    def infer(model: ClarityModel, nchw: np.ndarray) -> np.ndarray:
        if torch is None or ModelLoader is None:
            raise ClarityError("PyTorch and spandrel are required. Install requirements-gpu.txt on this machine.")
        if not torch.cuda.is_available():
            raise ClarityError("CUDA is not available. Clarity has no CPU mode.")
        network = loaded.get(model.id)
        if network is None:
            path = paths.get(model.id)
            if path is None:
                raise ClarityError(f"{model.id} was not fetched before the job")
            network = ModelLoader().load_from_file(str(path)).eval().cuda().half()
            loaded[model.id] = network
        tensor = torch.from_numpy(np.ascontiguousarray(nchw)).to(device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            output = network(tensor)
        return np.asarray(output.detach().float().cpu().numpy(), dtype=np.float32)

    return infer


def live_control(settings: Settings, token: CancelToken) -> Control:
    """The path the Start button uses. The lock is the suite GPU lock."""

    def lock(job: str, eta_s: float) -> AbstractContextManager[None]:
        return gpu_job(job, eta_s=eta_s, owner="clarity", cancel=token.is_set)

    return Control(
        infer=_prepare_infer(settings),
        lock=lock,
        spawn=_spawn,
        run_text=_run_text,
        token=token,
        device=True,
    )


def _read_exact(stream: IO[bytes], size: int, token: CancelToken) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        if token.is_set():
            raise Cancelled("cancelled")
        part = stream.read(size - len(buf))
        if token.is_set():
            raise Cancelled("cancelled")
        if not part:
            break
        buf += part
    if token.is_set():
        raise Cancelled("cancelled")
    return bytes(buf)


def _drain(stream: IO[bytes] | None, sink: list[str]) -> None:
    if stream is None:
        return
    try:
        data = stream.read()
    except OSError:
        return
    if data:
        sink.append(data.decode("utf-8", errors="replace")[-2000:])


def _as_bytes(frames: np.ndarray) -> bytes:
    image = np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image).tobytes()


def _write_all(stream: IO[bytes], payload: bytes) -> None:
    view = payload
    while view:
        written = stream.write(view)
        if written is None:
            return
        if written <= 0:
            raise ClarityError("the encoder stopped reading frames")
        view = view[written:]


def _fail_text(sink: list[str], code: int) -> str:
    detail = " ".join(part.strip() for part in sink if part.strip())
    if detail:
        return f"ffmpeg failed ({code}): {detail}"
    return f"ffmpeg failed ({code})"


def run_job(
    spec: JobSpec,
    control: Control | None = None,
    *,
    token: CancelToken | None = None,
    progress: Progress | None = None,
) -> None:
    """Decode with NVDEC, restore each chunk, and encode. Audio is stream-copied.

    Omitting ``control`` uses the engine shim, the suite GPU lock, and ffmpeg.
    A set cancel token stops the job and deletes the partial file.
    """
    if not spec.source.is_file():
        raise ClarityError("the source video was not found")
    if token is None:
        token = control.token if control is not None else CancelToken()
    text_of = control.run_text if control is not None else _run_text
    probe_payload = json.loads(text_of(probe_command(spec.ffprobe, str(spec.source))))
    if not isinstance(probe_payload, dict):
        raise ClarityError("ffprobe did not return an object")
    probe = parse_probe(probe_payload)
    require_sdr(probe)
    if control is None:
        bound = live_control(spec.settings, token)
    else:
        control.token = token
        bound = control
    if progress is not None:
        bound.progress = progress
    if token.is_set():
        raise Cancelled("cancelled")
    encoders = parse_encoder_list(bound.run_text([spec.ffmpeg, "-hide_banner", "-encoders"]))
    partial = spec.destination.with_suffix(spec.destination.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    commands = commands_for(
        probe,
        spec.settings,
        ffmpeg=spec.ffmpeg,
        source=str(spec.source),
        destination=str(partial),
        encoders=encoders,
    )
    frame_bytes = probe.width * probe.height * 3
    out_h = probe.height * (2 if spec.settings.upscale else 1)
    out_w = probe.width * (2 if spec.settings.upscale else 1)
    total = probe.frames if probe.frames > 0 else 0
    eta = max(120.0, float(total or 300))
    tile = spec.settings.tile if spec.settings.tile > 0 else 512
    batch = max(1, spec.settings.batch)
    baseline: int | None = None
    paused = False
    started = time.monotonic()
    stderr_text: list[str] = []
    done = 0

    def flush(chunk: list[np.ndarray]) -> None:
        nonlocal tile, batch, baseline, paused, done
        if token.is_set():
            raise Cancelled("cancelled")
        if bound.device:
            hooks = Hooks(
                infer=bound.infer,
                decide=thermal_decide,
                elapsed=lambda: time.monotonic() - started,
            )
            paused = _wait(hooks, paused, token)
            budget = float(getattr(measure(), "budget_gb", 0.0) or 0.0)
            if spec.settings.tile <= 0:
                tile = choose_tile(budget, 0, 0)
            usage = video_module().read_shared_usage()
            if video_module().sysmem_spill(usage, baseline):
                empty_cuda_cache()
                batch = note_spill(batch, True)
                smaller = halve_tile(tile)
                if smaller is not None:
                    tile = int(smaller)
                baseline = usage
            elif baseline is None:
                baseline = usage
        restored = process_frames(
            np.stack(chunk),
            spec.settings,
            bound.infer,
            tile=tile,
            batch=batch,
        )
        if restored.shape[1] != out_h or restored.shape[2] != out_w or restored.shape[-1] != 3:
            raise ClarityError(f"restored frames are {restored.shape}, expected {(len(chunk), out_h, out_w, 3)}")
        current = encoder
        if current is None or current.stdin is None:
            raise ClarityError("the encoder has no stdin")
        _write_all(current.stdin, _as_bytes(restored))
        done += len(chunk)
        denom = total if total > 0 else done
        bound.progress(done / denom, f"{done}/{denom}")

    decoder: Proc | None = None
    encoder: Proc | None = None
    try:
        with bound.lock("clarity:restore", eta):
            if token.is_set():
                raise Cancelled("cancelled")
            decoder = bound.spawn(list(commands["decode"]))
            encoder = bound.spawn(list(commands["encode"]))
            token.watch(decoder)
            token.watch(encoder)
            threading.Thread(target=_drain, args=(getattr(decoder, "stderr", None), stderr_text), daemon=True).start()
            threading.Thread(target=_drain, args=(getattr(encoder, "stderr", None), stderr_text), daemon=True).start()
            source_out = decoder.stdout
            if source_out is None:
                raise ClarityError("the decoder has no stdout")
            pending: list[np.ndarray] = []
            while True:
                if token.is_set():
                    raise Cancelled("cancelled")
                raw = _read_exact(source_out, frame_bytes, token)
                if token.is_set():
                    raise Cancelled("cancelled")
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise ClarityError("the decoder ended in the middle of a frame")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(probe.height, probe.width, 3).astype(np.float32)
                pending.append(frame / 255.0)
                if len(pending) >= spec.settings.max_frames:
                    flush(pending)
                    pending = []
            if pending:
                flush(pending)
            if token.is_set():
                raise Cancelled("cancelled")
            if done < 1:
                raise ClarityError("the decoder produced no frames")
            encoder_in = encoder.stdin
            if encoder_in is not None:
                encoder_in.close()
            code_decode = int(decoder.wait(timeout=120))
            code_encode = int(encoder.wait(timeout=120))
            if code_decode != 0 or code_encode != 0:
                raise ClarityError(_fail_text(stderr_text, code_decode or code_encode))
        if token.is_set():
            raise Cancelled("cancelled")
        os.replace(partial, spec.destination)
        bound.progress(1.0, f"{done}/{done}")
    except Cancelled:
        _discard(decoder, encoder, partial)
        raise
    except Exception:
        _discard(decoder, encoder, partial)
        if token.is_set():
            raise Cancelled("cancelled") from None
        raise


def _wait(hooks: Hooks, paused: bool, token: CancelToken) -> bool:
    """Thermal pause, but leave as soon as the user cancels."""

    def sleep(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if token.is_set():
                raise Cancelled("cancelled")
            time.sleep(0.1)

    hooks.sleep = sleep
    return wait_for_thermal(hooks, paused)


def _discard(decoder: Proc | None, encoder: Proc | None, partial: Path) -> None:
    for proc in (decoder, encoder):
        if proc is None:
            continue
        _kill(proc)
        try:
            proc.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
    try:
        partial.unlink(missing_ok=True)
    except OSError:
        return
