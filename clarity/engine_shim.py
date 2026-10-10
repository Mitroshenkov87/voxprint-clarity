"""The only door from Clarity into ``voxprint_engine``.

Video helpers that are not in enlarge 6d00d44 are loaded from
``docs/engine-patches/files`` until that patch lands. Delete this fallback
when ``voxprint_engine.video`` provides the same names. Do not grow a second
GPU lock, VRAM policy, model store, or ONNX session here.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

# Torch is an optional GPU extra. Importing it here keeps the import at module
# scope and lets unit tests run without the CUDA wheel.
try:
    import torch as _torch
except ImportError:
    _torch = None

# TODO: depend on voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge.git@engine-v1.0.0
# once that tag and the engine pyproject exist. Until then the pin is 6d00d44.
ENGINE_COMMIT = "6d00d44"
ENGINE_TAG_TODO = "engine-v1.0.0"
ADA = (8, 9)

_VIDEO_NAMES = ("temporal_windows", "tile_frames", "run_batch", "restore_frames", "sysmem_spill")


class EngineUnavailable(RuntimeError):
    """The shared engine is not installed, or it is older than the video patch and the patch file is missing."""


@dataclass(frozen=True)
class Gate:
    ok: bool
    message: str
    warning: str = ""


def _patch_video_path() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "docs" / "engine-patches" / "files" / "voxprint_engine" / "video.py"
        if candidate.is_file():
            return candidate
    return None


def _load_path(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EngineUnavailable(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses on 3.12 looks the class up in sys.modules while the body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def video_module() -> ModuleType:
    """Upstream ``voxprint_engine.video`` when it has the video API, else the patch file."""
    try:
        module = importlib.import_module("voxprint_engine.video")
    except ImportError:
        module = None
    if module is not None and all(hasattr(module, name) for name in _VIDEO_NAMES):
        return module
    path = _patch_video_path()
    if path is None:
        raise EngineUnavailable(
            "Clarity needs voxprint_engine.video (the patch in docs/engine-patches). "
            f"The engine pin is {ENGINE_COMMIT} until tag {ENGINE_TAG_TODO}."
        )
    return _load_path(path, "voxprint_engine_video_pending")


def require_engine() -> ModuleType:
    try:
        return importlib.import_module("voxprint_engine")
    except ImportError as exc:
        raise EngineUnavailable(
            "Clarity needs the shared voxprint-engine package from Voxprint AI Enlarge. "
            f"Pin: git+https://github.com/Mitroshenkov87/voxprint-enlarge.git@{ENGINE_COMMIT} "
            f"(TODO: switch to tag {ENGINE_TAG_TODO} when it exists). "
            "A private install needs the VOXPRINT_ENGINE_TOKEN read secret."
        ) from exc


def call_assess(assess: Callable[..., Any]) -> Any:
    """Call ``assess_gpu(refuse_below=8.9)``. Older engines only warn; Clarity turns that warning into a refusal."""
    try:
        return assess(refuse_below=ADA)
    except TypeError:
        gate = assess()
        warning = str(getattr(gate, "warning", "") or "")
        if bool(getattr(gate, "ok", False)) and warning:
            return type(gate)(False, warning, "")
        return gate


def assess_launch() -> Gate:
    """RTX 40 series (compute capability 8.9) and a 600-series driver, via the engine gate."""
    if os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1":
        return Gate(
            False,
            "Clarity has no CPU stand-in and does not honour VOXPRINT_CI_LANCZOS. "
            "An NVIDIA RTX 40 series GPU (or newer) and a 600-series driver are required.",
        )
    try:
        hardware = importlib.import_module("voxprint_engine.hardware")
    except ImportError as exc:
        raise EngineUnavailable(
            "Clarity cannot check the GPU because voxprint_engine is not installed. "
            f"Install the engine pinned to {ENGINE_COMMIT}."
        ) from exc
    gate = call_assess(hardware.assess_gpu)
    return Gate(bool(gate.ok), str(getattr(gate, "message", "") or ""), str(getattr(gate, "warning", "") or ""))


def _vram() -> ModuleType:
    require_engine()
    return importlib.import_module("voxprint_engine.vram")


def _thermal() -> ModuleType:
    require_engine()
    return importlib.import_module("voxprint_engine.thermal")


def _lock() -> ModuleType:
    require_engine()
    return importlib.import_module("voxprint_engine.gpu_lock")


def _store() -> ModuleType:
    require_engine()
    return importlib.import_module("voxprint_engine.store")


def measure(device: str = "auto") -> Any:
    return _vram().measure(device)


def choose_tile(budget_gb: float, model_tile: int = 0, user_tile: int = 0) -> int:
    return int(_vram().choose_tile(budget_gb, model_tile, user_tile))


def note_spill(batch: int, spilled: bool) -> int:
    return int(_vram().note_spill(batch, spilled))


def halve_tile(tile: int) -> int | None:
    value = _vram().halve_tile(tile)
    return None if value is None else int(value)


def thermal_decide(elapsed_s: float, median_c: float | None, throttling_s: float, paused: bool) -> Any:
    return _thermal().decide(elapsed_s, median_c, throttling_s, paused)


def gpu_job(
    job: str,
    eta_s: float = 600.0,
    owner: str = "clarity",
    cancel: Callable[[], bool] | None = None,
) -> Any:
    """Hold the suite lock. ``owner`` is passed when the engine accepts it (the video patch).

    ``cancel`` is polled while waiting. An engine that predates those arguments
    is called again without them.
    """
    lock = _lock()
    kwargs: dict[str, Any] = {"eta_s": eta_s}
    if cancel is not None:
        kwargs["cancel"] = cancel
    try:
        return lock.gpu_job(job, owner=owner, **kwargs)
    except TypeError:
        kwargs.pop("cancel", None)
        try:
            return lock.gpu_job(job, owner=owner, **kwargs)
        except TypeError:
            return lock.gpu_job(job, **kwargs)


def register_user(key: str = "clarity", root: Path | None = None) -> dict[str, bool]:
    store = _store()
    return dict(store.register(key, root))


def unregister_user(key: str = "clarity", root: Path | None = None) -> list[str]:
    store = _store()
    return list(store.unregister(key, root))


def models_directory() -> Path:
    require_engine()
    paths = importlib.import_module("voxprint_engine.paths")
    return Path(paths.models_dir())


def ensure_weight(model: Any, root: Path | None = None) -> Path:
    store = _store()
    return Path(store.ensure_weight(model, root))


def empty_cuda_cache() -> None:
    """Drop cached GPU blocks after a system-memory spill. No-op when torch is absent."""
    if _torch is None:
        return
    cuda = getattr(_torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    cuda.empty_cache()
