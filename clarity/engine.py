"""Calls into ``voxprint_engine`` and ``voxprint_common``.

Video names come from ``voxprint_engine.api``. The parent GPU lock records
the app id ``OWNER``, and the restore worker sends that same string as
``owner``. A ``restore_batch`` lock timeout comes back as an error line and
the worker stays up; ``submit_restore`` waits and retries.
"""
from __future__ import annotations

import importlib
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from voxprint_common.gpu_lock import GpuLockTimeout as GpuLockTimeout
from voxprint_common.gpu_lock import Holder, gpu_job
from voxprint_common.gpu_probe import assess as assess_gpu
from voxprint_common.paths import models_dir, shared_dir
from voxprint_common.thermal import decide as decide_thermal

from clarity.appinfo import APP_NAME
from clarity.errors import Cancelled, ClarityError
from clarity.i18n import tr
from clarity.progress import progress_stage

try:
    import torch
except ImportError:
    torch = None

try:
    import spandrel
except ImportError:
    spandrel = None

OWNER = "clarity"
NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
# Tag engine-v1.4.0. pyproject.toml is at the repository root, so this
# archive URL has no subdirectory. The gpu extra needs the NVIDIA package index.
ENGINE_TAG = "engine-v1.4.0"
ENGINE_REQUIREMENT = (
    "voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge"
    "/archive/refs/tags/engine-v1.4.0.tar.gz"
)
# The wheel ships voxprint_engine only. ensure_runtime lives in the Enlarge
# package at the same tag.
SUITE_RUNTIME = "enlarge.suite_runtime"
_ENGINE_MISSING = "VOXPrint inference engine missing, run repair."
_DRIVER_VERSION = re.compile(r"driver ([0-9][0-9.]*)")
_GPU_BUSY = "the GPU is busy"
_GPU_CANCELLED = "cancelled while waiting for the GPU"

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
_vram = _load_module("voxprint_engine.vram")
_Manifest = _load_attr("voxprint_engine.registry.schema", "ModelManifest")
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
        raise EngineUnavailable(_ENGINE_MISSING)
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


def driver_refusal(version: str) -> str:
    """Explain that the NVIDIA driver is older than the R600 series.

    Args:
        version: The driver version nvidia-smi reported.

    Returns:
        The refusal, including NVIDIA's driver download page.
    """
    shown = version.strip() or "unknown"
    return (
        f"This NVIDIA driver ({shown}) is older than the R600 series. "
        "Clarity needs a 600-series driver or newer. "
        f"Download a current driver from {NVIDIA_DRIVER_URL}"
    )


def annotate_driver_message(message: str) -> str:
    """Add the driver download page when the engine refused an old driver.

    Args:
        message: Text from ``voxprint_common.gpu_probe.assess``.

    Returns:
        ``message`` unchanged, or :func:`driver_refusal` when it names an old driver.
    """
    if "driver" not in message.lower() or "600" not in message:
        return message
    found = _DRIVER_VERSION.search(message)
    if found:
        return driver_refusal(found.group(1))
    if NVIDIA_DRIVER_URL in message:
        return message
    return f"{message} Download a current driver from {NVIDIA_DRIVER_URL}"


def gpu_stack_error() -> str:
    """Return why a GPU restore cannot start, or an empty string when torch and spandrel import.

    The shared venv can exist without Clarity's CUDA torch. A restore then
    fails at the first Clean or Strong frame. ``--selftest`` without
    ``--no-gpu`` reports that here. ``--selftest --no-gpu`` does not call this.

    Returns:
        An empty string when both packages import.
    """
    if torch is None or spandrel is None:
        return "PyTorch and spandrel are required. Install requirements-gpu.txt."
    return ""


def assess_launch() -> Gate:
    """RTX 40 series (compute capability 8.9) and a 600-series driver.

    The probe is ``voxprint_common.gpu_probe.assess``. ``VOXPRINT_CI_LANCZOS``
    is refused before that call. The engine package is still required, because
    a launch that passes the probe then uses it.
    """
    if os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1":
        return Gate(False, _LANCZOS_REFUSAL)
    _require()
    gate = assess_gpu(APP_NAME, refuse_below=(8, 9))
    message = annotate_driver_message(gate.message)
    return Gate(gate.ok, message, gate.warning)


def shared_directory() -> Path:
    """Return the shared suite directory.

    Returns:
        ``voxprint_common.paths.shared_dir()``.
    """
    return Path(shared_dir())


def _public_text(text: str) -> str:
    """Stand-alone wording for an engine message that names another app."""
    if "Enlarge" not in text:
        return text
    if "3.14" in text:
        return "Python 3.14 was not found. Clarity installs Python 3.14 with uv during setup."
    return _ENGINE_MISSING


def _guard_progress(progress: Callable[[str], None] | None) -> Callable[[str], None] | None:
    if progress is None:
        return None

    def emit(message: str) -> None:
        progress(_public_text(message))

    return emit


def _present_report(report: Any) -> Any:
    """Rewrite a runtime report so a status line stands alone for Clarity."""
    components = list(getattr(report, "components", ()) or ())
    rewritten: list[Any] = []
    changed = False
    for item in components:
        detail = _public_text(str(getattr(item, "detail", "") or ""))
        original_detail = str(getattr(item, "detail", "") or "")
        if detail != original_detail:
            changed = True
            holders = tuple(getattr(item, "holders", ()) or ())
            try:
                item = type(item)(item.id, item.version, item.state, holders, detail)
            except (TypeError, AttributeError):
                return report
        rewritten.append(item)
    if not changed:
        return report
    try:
        return type(report)(bool(report.ok), bool(getattr(report, "skipped", False)), tuple(rewritten))
    except TypeError:
        return report


def _suite_runtime() -> Any:
    """Import ``enlarge.suite_runtime``.

    Raises:
        EngineUnavailable: When that module is not installed.
    """
    module = _load_module(SUITE_RUNTIME)
    if module is None:
        raise EngineUnavailable(_ENGINE_MISSING)
    return module


def ensure_shared_runtime(progress: Callable[[str], None] | None = None) -> Any:
    """Hold ``clarity`` on the shared runtime and repair anything that is missing.

    ``enlarge.suite_runtime.ensure_runtime`` writes the app id into ``refs``
    on the Python and wheel components. It does not download Enlarge's default
    model. ``ensure_default_model`` installs that weight only for ``enlarge``.
    Catalogue weights are fetched separately, under ``clarity`` in the shared
    models folder. API stays 1.

    Args:
        progress: Receives one status line per check or install step.

    Returns:
        The runtime report.

    Raises:
        EngineUnavailable: When ``enlarge.suite_runtime`` is not installed.
        ClarityError: When the installer refuses the install.
    """
    module = _suite_runtime()
    try:
        report = module.ensure_runtime(OWNER, progress=_guard_progress(progress))
    except module.SuiteInstallError as exc:
        raise ClarityError(_public_text(str(exc))) from exc
    return _present_report(report)


def release_shared_runtime(progress: Callable[[str], None] | None = None) -> Any:
    """Drop ``clarity`` from the shared runtime.

    A payload is deleted only when its ref list is empty.

    Args:
        progress: Receives one status line per release step.

    Returns:
        The runtime report.

    Raises:
        EngineUnavailable: When ``enlarge.suite_runtime`` is not installed.
        ClarityError: When the installer refuses the release.
    """
    module = _suite_runtime()
    try:
        report = module.release_runtime(OWNER, progress=_guard_progress(progress))
    except module.SuiteInstallError as exc:
        raise ClarityError(_public_text(str(exc))) from exc
    return _present_report(report)


def hold_gpu(
    job: str,
    eta_s: float = 600.0,
    *,
    cancel: Callable[[], bool] | None = None,
    on_wait: Callable[[Holder, float], None] | None = None,
    timeout: float | None = None,
) -> Any:
    """Hold the suite lock for app ``OWNER`` during ``job``.

    The restore worker must send the same app id as ``owner``.
    """
    return gpu_job(OWNER, job, eta=eta_s, timeout=timeout, cancel=cancel, on_wait=on_wait)


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


def iter_tile_frames(*args: Any, **kwargs: Any) -> Any:
    """Yield one restored frame from ``voxprint_engine.api.iter_tile_frames``.

    The accumulator stays one frame. Stacking the yields allocates the chunk again.

    Args:
        *args: Positional arguments forwarded unchanged.
        **kwargs: Keyword arguments forwarded unchanged.

    Returns:
        An iterator of HWC float frames.
    """
    return _require().iter_tile_frames(*args, **kwargs)


def ensure_model_onnx(
    payload: dict[str, Any],
    weight: Path,
    on_progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Convert a pinned weight with the engine and return the cached ONNX file.

    The engine stores ``<sha256>.onnx``. A second call returns that file.

    Args:
        payload: Manifest fields for the weight.
        weight: The checked ``.pth`` file.
        on_progress: Receives ``(fraction, message)`` while the engine converts.

    Returns:
        Path of the checked ONNX file.

    Raises:
        ClarityError: When the engine cannot convert the weight.
        EngineUnavailable: When the engine package is not installed.
    """
    api = _require()
    error = _load_attr("voxprint_engine.onnx_export", "OnnxExportError")
    store_error = _load_attr("voxprint_engine.store", "StoreError")
    try:
        return Path(api.ensure_onnx(manifest_for(payload), None, on_progress, weight=weight))
    except ClarityError:
        raise
    except Exception as exc:
        if error is not None and isinstance(exc, error):
            raise ClarityError(str(exc)) from exc
        if store_error is not None and isinstance(exc, store_error):
            raise ClarityError(str(exc)) from exc
        raise


def _engine_stage_text(message: str) -> str:
    """Turn an engine TensorRT sentence into the status line the window shows."""
    stage = progress_stage(message)
    if stage == "building_engine":
        return tr("status.building_engine")
    if stage == "loading_cached_engine":
        return tr("status.loading_cached_engine")
    return message


def open_restore_session(onnx_path: Path, on_progress: Callable[[float, str], None]) -> Any:
    """Open the ONNX file, trying TensorRT and then CUDA.

    The progress callback hears the engine's build or cache message before
    TensorRT loads, because the first engine build can take minutes.

    Args:
        onnx_path: A checked ONNX file from :func:`ensure_model_onnx`.
        on_progress: Receives ``(fraction, message)`` for the build and the cache.

    Returns:
        The open ONNX Runtime session.

    Raises:
        ClarityError: When no provider accepts the file.
        EngineUnavailable: When the engine session helpers are not installed.
    """
    session_mod = _load_module("voxprint_engine.session")
    providers_mod = _load_module("voxprint_engine.providers")
    if session_mod is None or providers_mod is None:
        raise EngineUnavailable(_ENGINE_MISSING)
    installed = session_mod.available_providers()
    labels = ("clarity", "driver", "trt")
    attempts = [providers_mod.choices_for(installed, "auto", gpu=labels[0], driver=labels[1], trt=labels[2])]
    fallback = providers_mod.cuda_fallback_choices(installed, "auto", gpu=labels[0], driver=labels[1], trt=labels[2])
    if fallback:
        attempts.append(fallback)
    errors: list[str] = []
    for choices in attempts:
        if not choices:
            continue
        if choices[0].name == "TensorrtExecutionProvider":
            cache = Path(str(choices[0].options.get("trt_engine_cache_path") or ""))
            on_progress(0.0, _engine_stage_text(str(session_mod.trt_build_message(cache))))
        try:
            session, provider = session_mod.open_session(onnx_path, choices)
        except session_mod.EngineBuildError as exc:
            errors.append(str(exc))
            continue
        if provider == "TensorrtExecutionProvider":
            on_progress(0.0, tr("status.loading_cached_engine"))
        return session
    raise ClarityError("; ".join(errors) or "the ONNX session did not load")


def graph_batch_limit(session: Any) -> int | None:
    """Return the session's fixed batch size, or None when that axis is dynamic.

    The engine export marks height and width dynamic and traces a batch of 1.
    A string or missing dimension means the graph accepts the batch it is given.

    Args:
        session: An ONNX Runtime session, or a test double with ``get_inputs``.

    Returns:
        A positive integer, or None.
    """
    getter = getattr(session, "get_inputs", None)
    if not callable(getter):
        return None
    try:
        specs = getter()
    except (TypeError, RuntimeError, OSError, ValueError):
        return None
    if not specs:
        return None
    shape = getattr(specs[0], "shape", None)
    if not shape:
        return None
    dim = shape[0]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        return None
    return int(dim)


def run_onnx_batch(session: Any, nchw: Any) -> Any:
    """Run one NCHW batch on an ONNX Runtime session from the engine.

    A tile batch larger than the graph's fixed batch is run in slices and
    concatenated. A dynamic batch axis is one call.

    Args:
        session: A session from :func:`open_restore_session`.
        nchw: Float NCHW input.

    Returns:
        Float32 NCHW output.
    """
    video = _load_module("voxprint_engine.video")
    if video is None:
        raise EngineUnavailable(_ENGINE_MISSING)
    array = np.asarray(nchw)
    limit = graph_batch_limit(session)
    count = int(array.shape[0]) if array.ndim == 4 else 0
    if limit is None or count <= limit:
        return video.run_batch(session, nchw)
    parts = [
        np.asarray(video.run_batch(session, array[start : start + limit]), dtype=np.float32)
        for start in range(0, count, limit)
    ]
    return np.concatenate(parts, axis=0)


def thermal_decide(elapsed_s: float, median_c: float | None, throttling_s: float, paused: bool) -> Any:
    """Ask ``voxprint_common.thermal`` whether to pause between chunks.

    Args:
        elapsed_s: Seconds since the job started.
        median_c: Median GPU temperature, or ``None`` when sensors are missing.
        throttling_s: Seconds the GPU has spent throttling.
        paused: Whether the job is already paused.

    Returns:
        The decision object. Callers read its ``action``.
    """
    return decide_thermal(elapsed_s, median_c, throttling_s, paused)


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
    """Return the shared models directory.

    Returns:
        ``voxprint_common.paths.models_dir()``.
    """
    return Path(models_dir())


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
