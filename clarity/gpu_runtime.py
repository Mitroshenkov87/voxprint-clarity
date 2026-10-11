"""CUDA 13 and TensorRT on the process search path before ONNX Runtime.

``voxprint_engine.trt_libs.prepare_native_runtime`` registers those
directories and keeps the Windows ``os.add_dll_directory`` handles. The
pinned tag ``engine-v1.4.0`` (commit ``f0ecb87``) does not define that
function. When it is missing, this module registers the same directories
and also ``nvidia/*/bin/x86_64``, which is where the CUDA 13 wheels keep
``cudart64_13.dll``.

Nothing here imports ``onnxruntime`` at import time. A session import loads
that module, so the directory registration has to finish first.
"""
from __future__ import annotations

import os
import site
import sys
import sysconfig
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path

from clarity.errors import ClarityError

_HANDLES: list[object] = []
_PREPARED = False
_GPU = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
_MISSING_DLL = (
    "nvinfer_10.dll is missing from the DLL search path "
    "(site-packages/tensorrt_libs and site-packages/nvidia/*/bin)"
)


class GpuRuntimeError(ClarityError):
    """TensorRT or CUDA did not stay active on a real session."""


def reset_gpu_runtime() -> None:
    """Drop the one-shot guard so a test can register directories again."""
    global _PREPARED
    _PREPARED = False


def site_package_roots() -> list[Path]:
    """Directories that can hold the CUDA and TensorRT wheels."""
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            roots.append(Path(value))
    try:
        packages = site.getsitepackages()
    except (AttributeError, OSError):
        return roots
    for item in packages:
        roots.append(Path(item))
    return roots


def native_gap_directories(roots: Sequence[Path]) -> list[Path]:
    """TensorRT library directory and every NVIDIA ``bin`` the wheels ship.

    Args:
        roots: Site-packages directories to scan.

    Returns:
        Existing directories, without duplicates.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        _add_dir(found, seen, root / "tensorrt_libs")
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        try:
            children = list(nvidia.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            bin_dir = child / "bin"
            _add_dir(found, seen, bin_dir)
            _add_dir(found, seen, bin_dir / "x86_64")
            lib_dir = child / "lib"
            if _has_cu13(lib_dir):
                _add_dir(found, seen, lib_dir)
            if _has_cu13(lib_dir / "x64"):
                _add_dir(found, seen, lib_dir / "x64")
    return found


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

    Calls ``prepare_native_runtime`` when the installed engine provides it.
    That helper does not walk ``nvidia/*/bin/x86_64``, so the gap registration
    still runs and adds those directories. The pinned tag has no helper at
    all, and then the gap registration is the whole search path. A second
    call does nothing. A missing engine package is not an error: docs and
    unit tests import Clarity without it.
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
    # The engine helper scans nvidia/*/bin, not nvidia/*/bin/x86_64.
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
        GpuRuntimeError: The session stayed on CPU. The message names
            ``nvinfer_10.dll``.
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
    raise GpuRuntimeError(f"ONNX Runtime kept {active}. {_MISSING_DLL}.")


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
    for name in _GPU:
        active, reason = session_providers(name)
        if name not in [str(item) for item in active]:
            missing.append(str(reason or active))
    if missing:
        raise GpuRuntimeError(f"{_MISSING_DLL}. {'; '.join(missing)}")


def main() -> int:
    """Return 0 when both providers stay active on a real session.

    Returns:
        1 when TensorRT or CUDA falls back. The message names ``nvinfer_10.dll``.
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


def _add_dir(found: list[Path], seen: set[Path], folder: Path) -> None:
    if not folder.is_dir():
        return
    try:
        resolved = folder.resolve()
    except OSError:
        return
    if resolved in seen:
        return
    seen.add(resolved)
    found.append(folder)


def _has_cu13(folder: Path) -> bool:
    try:
        names = [path.name.lower() for path in folder.iterdir() if path.is_file()]
    except OSError:
        return False
    return any("_13" in name or ".so.13" in name or "cu13" in name for name in names)


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
    ``nvinfer_10.dll`` and the CUDA 13 bins are on the search path.
    """
    import numpy as np
    import onnx
    from onnx import helper

    import onnxruntime as ort

    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph(
        [node],
        "identity",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    payload = model.SerializeToString()
    sample = np.zeros((1, 1), dtype=np.float32)
    errors: list[str] = []
    for name in _GPU:
        try:
            session = ort.InferenceSession(payload, providers=[name])
            active = [str(item) for item in session.get_providers()]
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if name not in active:
            errors.append(f"{name} fell back to {active}")
            continue
        session.run(None, {"X": sample})
    if errors:
        raise GpuRuntimeError(f"{_MISSING_DLL}. {'; '.join(errors)}")
