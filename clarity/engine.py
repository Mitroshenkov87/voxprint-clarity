"""Calls into ``voxprint_engine``. Clarity does not keep a second copy of it.

Video names come from ``voxprint_engine.api``. The parent GPU lock and the
restore worker both use ``OWNER``, because the engine shares a parent lock
only when that string matches. A ``restore_batch`` lock timeout comes back as
an error line and the worker stays up; ``submit_restore`` waits and retries.
"""
from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from clarity.errors import Cancelled, ClarityError
from clarity.i18n import tr

try:
    import torch
except ImportError:
    torch = None

OWNER = "clarity"
ENGINE_REQUIREMENT = "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0"
_GPU_BUSY = "the GPU is busy"
_GPU_CANCELLED = "cancelled while waiting for the GPU"

class _MissingLock(TimeoutError):
    """Stand-in used only when the engine package is not installed."""


def _load_module(name: str) -> Any:
    """Import ``name`` through importlib. A missing module returns None.

    The result stays ``Any``, so mypy accepts the missing-package branch after
    ``voxprint_engine`` is installed and its modules are typed.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _load_attr(module_name: str, attr: str) -> Any:
    module = _load_module(module_name)
    if module is None:
        return None
    return getattr(module, attr)


_engine_api = _load_module("voxprint_engine.api")
_hardware = _load_module("voxprint_engine.hardware")
_vram = _load_module("voxprint_engine.vram")
_Manifest = _load_attr("voxprint_engine.registry.schema", "ModelManifest")
_LockTimeout = _load_attr("voxprint_engine.gpu_lock", "GpuLockTimeout")
if _LockTimeout is None:
    _LockTimeout = _MissingLock

GpuLockTimeout = _LockTimeout
ModelManifest = _Manifest


class EngineUnavailable(RuntimeError):
    """The shared engine package is not installed."""


class _LineWorker(Protocol):
    """The slice of ``WorkerClient`` a restore retry needs. It must not kill the process."""

    def _send(self, message: dict[str, Any]) -> None: ...

    def _read(self, timeout: float) -> dict[str, Any]: ...


def engine_ready() -> bool:
    """Return whether ``voxprint_engine.api`` imported.

    Returns:
        True when the shared engine package is installed.
    """
    return _engine_api is not None


def _require() -> Any:
    if _engine_api is None:
        raise EngineUnavailable(
            "Clarity needs the shared voxprint-engine package from Voxprint AI Enlarge. "
            f"Install {ENGINE_REQUIREMENT}."
        )
    return _engine_api


_LANCZOS_REFUSAL = (
    "Clarity has no CPU stand-in and does not honour VOXPRINT_CI_LANCZOS. "
    "An NVIDIA RTX 40 series GPU (or newer) and a 600-series driver are required."
)


class Gate:
    """Launch decision. ``message`` is empty when ``ok`` is true."""

    __slots__ = ("ok", "message", "warning")

    def __init__(self, ok: bool, message: str, warning: str = "") -> None:
        """Store a launch decision.

        Args:
            ok: Whether Clarity may start a GPU job.
            message: Refusal text. Empty when ``ok`` is true.
            warning: Optional note that does not block launch.
        """
        self.ok = ok
        self.message = message
        self.warning = warning


def assess_launch() -> Gate:
    """RTX 40 series (compute capability 8.9) and a 600-series driver."""
    if os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1":
        return Gate(False, _LANCZOS_REFUSAL)
    _require()
    if _hardware is None:
        raise EngineUnavailable("voxprint_engine.hardware is not installed")
    gate = _hardware.assess_gpu(refuse_below=(8, 9))
    return Gate(bool(gate.ok), str(getattr(gate, "message", "") or ""), str(getattr(gate, "warning", "") or ""))


def hold_gpu(
    job: str,
    eta_s: float = 600.0,
    *,
    cancel: Callable[[], bool] | None = None,
    on_wait: Callable[[dict[str, Any], float], None] | None = None,
    timeout: float | None = None,
) -> Any:
    """Hold the suite lock as ``OWNER``. The worker must use that same owner."""
    api = _require()
    return api.gpu_job(job, eta_s=eta_s, timeout=timeout, cancel=cancel, on_wait=on_wait, owner=OWNER)


def choose_tile(budget_gb: float, model_tile: int = 0, user_tile: int = 0) -> int:
    """Ask the engine for a tile size that fits the VRAM budget.

    Args:
        budget_gb: Free VRAM the engine may spend, in gibibytes.
        model_tile: Tile the model prefers. ``0`` leaves the choice to the engine.
        user_tile: Tile from settings. ``0`` leaves the choice to the engine.

    Returns:
        Tile side length in pixels.
    """
    return int(_require().choose_tile(budget_gb, model_tile, user_tile))


def note_spill(batch: int, spilled: bool) -> int:
    """Ask the engine for the next batch size after a shared-memory spill.

    Args:
        batch: Frames per forward pass before the spill.
        spilled: Whether shared GPU memory crossed the spill line.

    Returns:
        The batch size to use next.
    """
    return int(_require().note_spill(batch, spilled))


def halve_tile(tile: int) -> int | None:
    """Ask the engine for a smaller tile after a spill.

    Args:
        tile: Current tile side length.

    Returns:
        The smaller tile, or ``None`` when the engine will not shrink it further.
    """
    value = _require().halve_tile(tile)
    return None if value is None else int(value)


def sysmem_spill(shared_now: int | None, shared_baseline: int | None, threshold: int = 256 * 1024 * 1024) -> bool:
    """Ask the engine whether shared GPU memory has spilled into system RAM.

    Args:
        shared_now: Current shared-memory use, in bytes.
        shared_baseline: Use recorded before this chunk.
        threshold: Byte rise that counts as a spill. The engine default is 256 MiB.

    Returns:
        True when the engine wants a smaller batch and tile.
    """
    return bool(_require().sysmem_spill(shared_now, shared_baseline, threshold))


def read_shared_usage() -> int | None:
    """Read shared GPU memory use from the engine.

    Returns:
        Bytes in use, or ``None`` when the engine cannot read the counter.
    """
    value = _require().read_shared_usage()
    return None if value is None else int(value)


def restore_frames(*args: Any, **kwargs: Any) -> Any:
    """Run one tiled restore through ``voxprint_engine.api.restore_frames``.

    Args:
        *args: Positional arguments forwarded unchanged.
        **kwargs: Keyword arguments forwarded unchanged.

    Returns:
        Whatever the engine returns for that call.
    """
    return _require().restore_frames(*args, **kwargs)


def thermal_decide(elapsed_s: float, median_c: float | None, throttling_s: float, paused: bool) -> Any:
    """Ask the engine whether to pause between chunks.

    Args:
        elapsed_s: Seconds since the job started.
        median_c: Median GPU temperature, or ``None`` when sensors are missing.
        throttling_s: Seconds the GPU has spent throttling.
        paused: Whether the job is already paused.

    Returns:
        The engine decision object. Callers read its ``action``.
    """
    return _require().decide(elapsed_s, median_c, throttling_s, paused)


def measure(device: str = "auto") -> Any:
    """Measure VRAM through ``voxprint_engine.vram``.

    Args:
        device: Engine device name. ``auto`` picks the installed NVIDIA GPU.

    Returns:
        The engine measurement. Callers read ``budget_gb``.
    """
    _require()
    if _vram is None:
        raise EngineUnavailable("voxprint_engine.vram is not installed")
    return _vram.measure(device)


def manifest_for(payload: dict[str, Any]) -> Any:
    """Build an engine model manifest.

    Args:
        payload: Fields ``voxprint_engine`` accepts for one weight file.

    Returns:
        The engine manifest object.
    """
    _require()
    if _Manifest is None:
        raise EngineUnavailable("voxprint_engine is not installed")
    return _Manifest(**payload)


def ensure_weight(payload: dict[str, Any], root: Path | None = None) -> Path:
    """Download or reuse one catalogue weight in the shared models folder.

    Args:
        payload: Manifest fields for the weight.
        root: Models directory. ``None`` uses the engine default.

    Returns:
        Path of the checked weight file.
    """
    return Path(_require().ensure_weight(manifest_for(payload), root))


def register_user(key: str = OWNER, root: Path | None = None) -> dict[str, bool]:
    """Register Clarity as a user of the shared models folder.

    Args:
        key: Suite user key. The default is ``OWNER``.
        root: Models directory. ``None`` uses the engine default.

    Returns:
        The engine's registration map.
    """
    return dict(_require().register(key, root))


def unregister_user(key: str = OWNER, root: Path | None = None) -> list[str]:
    """Drop Clarity's registration. Weights stay until no user remains.

    Args:
        key: Suite user key. The default is ``OWNER``.
        root: Models directory. ``None`` uses the engine default.

    Returns:
        User keys that still hold the folder.
    """
    return list(_require().unregister(key, root))


def models_directory() -> Path:
    """Return the shared models directory from the engine.

    Returns:
        Path of the folder that holds catalogue weights.
    """
    return Path(_require().models_dir())


def restore_message(**fields: Any) -> dict[str, Any]:
    """A ``restore_batch`` line. ``owner`` is always ``OWNER``."""
    api = _require()
    fields["owner"] = OWNER
    return dict(api.RestoreBatchRequest(**fields).to_message())


def gpu_busy_text(detail: str) -> str:
    """Format the translated "GPU is busy" status.

    Args:
        detail: Engine text that names the current lock holder.

    Returns:
        The localized status line.
    """
    return tr("status.gpu_busy").format(detail=detail)


def _is_gpu_wait(message: str) -> bool:
    return message.startswith(_GPU_BUSY)


def submit_restore(
    client: _LineWorker,
    message: dict[str, Any],
    *,
    on_status: Callable[[str], None],
    cancel: Callable[[], bool],
    pause: Callable[[], None] | None = None,
    timeout: float = 60.0,
    limit: int = 120,
) -> dict[str, Any]:
    """Send ``restore_batch``. A GPU-lock error line waits and retries.

    The engine returns that timeout as ``op=error`` and leaves the worker
    running. This function does not kill the worker. Cancel stops the retries.
    """
    if message.get("owner") != OWNER:
        raise ClarityError("the restore worker must use the same GPU owner as Clarity")
    wait = pause or (lambda: time.sleep(0.25))
    last = ""
    for _ in range(limit):
        if cancel():
            raise Cancelled("cancelled")
        client._send(message)
        while True:
            reply = client._read(timeout)
            op = str(reply.get("op") or "")
            if op == "progress":
                on_status(str(reply.get("message") or ""))
                continue
            if op == "result" and reply.get("ok") is True:
                return reply
            if op == "error":
                last = str(reply.get("message") or "")
                if last == _GPU_CANCELLED or cancel():
                    raise Cancelled("cancelled")
                if _is_gpu_wait(last):
                    on_status(gpu_busy_text(last))
                    wait()
                    break
                raise ClarityError(last or "the restore worker failed")
            raise ClarityError(f"unexpected worker message: {reply}")
    raise ClarityError(gpu_busy_text(last or "the wait expired"))


def empty_cuda_cache() -> None:
    """Drop cached GPU blocks after a system-memory spill. No-op without torch."""
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    cuda.empty_cache()
