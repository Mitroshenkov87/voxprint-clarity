"""CUDA 13 and TensorRT on the process search path before ONNX Runtime.

``voxprint_engine.trt_libs.prepare_native_runtime`` registers wheel
directories and keeps the Windows ``os.add_dll_directory`` handles. The
pinned tag ``engine-v1.4.0`` (commit ``f0ecb87``) does not define that
function. This module always registers its own directories as well.

CUDA 13 wheels nest ``cudart64_13.dll`` under ``nvidia/<any>/bin/x86_64``.
On the RTX 4090 laptop, ``cublas64_13.dll`` and ``cublasLt64_13.dll`` exist
only in ``site-packages/torch/lib``. A one-level ``nvidia/*/bin`` walk misses
both a deeper nest and that torch directory, and ONNX Runtime then stays on
the CPU.

Nothing here imports ``onnxruntime`` at import time. A session import loads
that module, so the directory registration has to finish first.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path

from clarity.errors import ClarityError
from clarity.native_gap import (
    build_identity_model,
    call_capturing_stderr,
    format_missing_runtime,
    format_probe_failure,
)
from clarity.native_gap import native_gap_directories as shared_gap_directories
from clarity.native_gap import site_package_roots as shared_site_package_roots

_HANDLES: list[object] = []
_PREPARED = False
_GPU = ("TensorrtExecutionProvider", "CUDAExecutionProvider")


class GpuRuntimeError(ClarityError):
    """TensorRT or CUDA did not stay active on a real session."""


def reset_gpu_runtime() -> None:
    """Drop the one-shot guard so a test can register directories again."""
    global _PREPARED
    _PREPARED = False


def site_package_roots() -> list[Path]:
    """Directories that can hold the CUDA and TensorRT wheels."""
    return shared_site_package_roots()


def native_gap_directories(roots: Sequence[Path]) -> list[Path]:
    """CUDA 13, cuDNN, and TensorRT directories, plus ``torch/lib``.

    The installer probe loads the same function from :mod:`clarity.native_gap`.

    Args:
        roots: Site-packages directories to scan.

    Returns:
        Existing directories, without duplicates.
    """
    return shared_gap_directories(roots)


def register_native_gap(
    roots: Sequence[Path],
    *,
    platform: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> list[Path]:
    """Keep DLL-directory handles and prepend the same folders to ``PATH``.

    Args:
        roots: Site-packages directories to scan.
        platform: ``sys.platform`` when omitted. Tests pass ``win32``.
        env: Environment to prepend. Defaults to ``os.environ``.

    Returns:
        The directories that were found.
    """
    folders = native_gap_directories(roots)
    system = sys.platform if platform is None else platform
    target = os.environ if env is None else env
    if system == "win32":
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            for folder in folders:
                try:
                    _HANDLES.append(add(str(folder)))
                except OSError:
                    continue
    _prepend(target, folders, "PATH" if system == "win32" else "LD_LIBRARY_PATH")
    return folders


def prepare_gpu_runtime() -> None:
    """Register CUDA 13 and TensorRT before any ``onnxruntime`` import.

    Calls ``prepare_native_runtime`` when the installed engine provides it,
    then always registers the directories this module finds. The engine
    helper does not add ``torch/lib``. The pinned tag has no helper at all.
    A second call does nothing. A missing engine package is not an error:
    docs and unit tests import Clarity without it.
    """
    global _PREPARED
    if _PREPARED:
        return
    try:
        prepare = _engine_prepare()
    except ImportError:
        register_native_gap(site_package_roots())
        _PREPARED = True
        return
    prepare()
    # The engine helper can miss torch/lib and a nest deeper than nvidia/*/bin.
    register_native_gap(site_package_roots())
    _PREPARED = True


def require_gpu_session(session: object) -> list[str]:
    """Refuse a session that kept neither TensorRT nor CUDA.

    A test double with no ``get_providers`` is left alone. A real session
    that lists only ``CPUExecutionProvider`` is an error, including when the
    name list also omits the GPU providers.

    Args:
        session: An ONNX Runtime session, or a test double.

    Returns:
        The provider names, or an empty list when the object has none.

    Raises:
        GpuRuntimeError: The session stayed on CPU. The message lists which
            of ``cudart64_13``, ``cublas64_13``, ``cublasLt64_13``, ``cudnn*``,
            and ``nvinfer_10`` the search path does not contain.
    """
    getter = getattr(session, "get_providers", None)
    if not callable(getter):
        return []
    try:
        active = [str(name) for name in getter()]
    except (TypeError, RuntimeError, OSError, ValueError):
        return []
    if any(name in active for name in _GPU):
        return active
    raise GpuRuntimeError(f"ONNX Runtime kept {active}. {missing_runtime_message()}.")


def require_tensorrt_and_cuda() -> None:
    """Open a real TensorRT session and a real CUDA session.

    The engine's ``session_providers`` opens a tiny graph and reads
    ``get_providers()``. The pinned tag has no such helper, so this builds
    a one-operation Identity model and opens that instead.

    Raises:
        GpuRuntimeError: Either provider failed to stay active.
    """
    prepare_gpu_runtime()
    # The pinned tag's session module has no session_providers. getattr keeps
    # mypy quiet when that name is absent and the package is installed.
    try:
        import voxprint_engine.session as session_mod
    except ImportError:
        _require_with_identity_model()
        return
    session_providers = getattr(session_mod, "session_providers", None)
    if not callable(session_providers):
        _require_with_identity_model()
        return
    missing: list[str] = []
    logs: list[str] = []
    for name in _GPU:

        def open_provider(provider: str = name):
            return session_providers(provider)

        (active, reason), log = call_capturing_stderr(open_provider)
        if log:
            logs.append(log)
        if name not in [str(item) for item in active]:
            missing.append(str(reason or active))
    if missing:
        folders = native_gap_directories(site_package_roots())
        raise GpuRuntimeError(format_probe_failure(folders, missing, "\n".join(logs)))


def main() -> int:
    """Return 0 when both providers stay active on a real session.

    Returns:
        1 when TensorRT or CUDA falls back. The message lists the missing libraries.
    """
    try:
        require_tensorrt_and_cuda()
    except GpuRuntimeError as exc:
        print(str(exc), flush=True)
        return 1
    return 0


def _engine_prepare() -> Callable[[], object]:
    """Return the engine helper. ImportError when this pin does not have it.

    ``getattr`` is required: engine-v1.4.0 installs, and mypy then sees that
    ``prepare_native_runtime`` is not on that tag.
    """
    import voxprint_engine.trt_libs as trt_libs

    prepare = getattr(trt_libs, "prepare_native_runtime", None)
    if not callable(prepare):
        raise ImportError("prepare_native_runtime")
    return prepare


def missing_runtime_message(roots: Sequence[Path] | None = None) -> str:
    """Name the CUDA 13 and TensorRT libraries the search path does not have.

    Args:
        roots: Site-packages directories. Defaults to this process.

    Returns:
        A single sentence. It lists whichever of ``cudart64_13``,
        ``cublas64_13``, ``cublasLt64_13``, ``cudnn*``, and ``nvinfer_10``
        is absent. When every one of those is present, it says so.
    """
    folders = native_gap_directories(site_package_roots() if roots is None else roots)
    return format_missing_runtime(folders)


def _prepend(env: MutableMapping[str, str], folders: Sequence[Path], key: str) -> None:
    if not folders:
        return
    current = env.get(key, "")
    parts = [part for part in current.split(os.pathsep) if part]
    prefix = [str(folder) for folder in folders if str(folder) not in parts]
    if not prefix:
        return
    env[key] = os.pathsep.join([*prefix, *parts])


def _require_with_identity_model() -> None:
    """One Identity op, then TensorRT and CUDA. Used when the engine pin is old.

    ``onnxruntime`` is imported here, after :func:`prepare_gpu_runtime`, because
    importing it earlier makes Windows load the provider DLLs before
    ``torch/lib`` and the CUDA 13 bins are on the search path.
    """
    import numpy as np

    import onnxruntime as ort

    payload = build_identity_model()
    sample = np.zeros((1, 1), dtype=np.float32)
    errors: list[str] = []
    logs: list[str] = []
    for name in _GPU:

        def open_session(provider: str = name) -> list[str] | str:
            try:
                session = ort.InferenceSession(payload, providers=[provider])
                active = [str(item) for item in session.get_providers()]
                if provider not in active:
                    return f"{provider} fell back to {active}"
                session.run(None, {"X": sample})
            except (OSError, RuntimeError, ValueError) as exc:
                return f"{provider}: {exc}"
            return active

        outcome, log = call_capturing_stderr(open_session)
        if log:
            logs.append(log)
        if isinstance(outcome, str):
            errors.append(outcome)
    if errors:
        folders = native_gap_directories(site_package_roots())
        raise GpuRuntimeError(format_probe_failure(folders, errors, "\n".join(logs)))
