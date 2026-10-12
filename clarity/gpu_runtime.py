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

import fnmatch
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
# Directory scan. ``*.so.13`` and ``libnvinfer.so*`` are the Linux names of the
# same libraries the Windows globs select.
_RUNTIME_GLOBS = (
    "*64_13.dll",
    "nvinfer*.dll",
    "cudnn*.dll",
    "libnvinfer.so*",
    "libcudnn.so*",
    "*.so.13",
)
# Names a failed session reports. The label is what the user reads.
_REQUIRED_LIBS = (
    ("cudart64_13.dll", "cudart64_13"),
    ("cublas64_13.dll", "cublas64_13"),
    ("cublaslt64_13.dll", "cublasLt64_13"),
    ("cudnn*.dll", "cudnn*"),
    ("nvinfer_10.dll", "nvinfer_10"),
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
    """CUDA 13, cuDNN, and TensorRT directories, plus ``torch/lib``.

    ``nvidia`` and ``tensorrt_libs`` are scanned at any depth. A directory is
    kept when it contains ``*64_13.dll``, ``nvinfer*.dll``, or ``cudnn*.dll``
    (or the Linux ``.so`` names of those libraries). ``torch/lib`` is added
    whenever the directory exists, because cuBLAS 13 can live only there.

    Args:
        roots: Site-packages directories to scan.

    Returns:
        Existing directories, without duplicates.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        _scan_runtime_tree(found, seen, root / "nvidia")
        _scan_runtime_tree(found, seen, root / "tensorrt_libs")
        _add_dir(found, seen, root / "torch" / "lib")
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
    for name in _GPU:
        active, reason = session_providers(name)
        if name not in [str(item) for item in active]:
            missing.append(str(reason or active))
    if missing:
        raise GpuRuntimeError(f"{missing_runtime_message()}. {'; '.join(missing)}")


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
    missing = _missing_runtime_labels(folders)
    if not missing:
        return "CUDA and TensorRT libraries were on the DLL search path, but the session did not keep a GPU provider"
    return "Missing from the DLL search path: " + ", ".join(missing)


def _scan_runtime_tree(found: list[Path], seen: set[Path], root: Path) -> None:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        folder = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (folder / name).is_symlink()]
        if _has_runtime_library(filenames):
            _add_dir(found, seen, folder)


def _has_runtime_library(names: Sequence[str]) -> bool:
    for name in names:
        lowered = name.lower()
        if any(fnmatch.fnmatch(lowered, pattern) for pattern in _RUNTIME_GLOBS):
            return True
    return False


def _missing_runtime_labels(folders: Sequence[Path]) -> list[str]:
    present: list[str] = []
    for folder in folders:
        try:
            present.extend(path.name.lower() for path in folder.iterdir() if path.is_file())
        except OSError:
            continue
    missing: list[str] = []
    for pattern, label in _REQUIRED_LIBS:
        if not any(fnmatch.fnmatch(name, pattern) for name in present):
            missing.append(label)
    return missing


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
        raise GpuRuntimeError(f"{missing_runtime_message()}. {'; '.join(errors)}")
