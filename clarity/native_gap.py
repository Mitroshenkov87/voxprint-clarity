"""CUDA 13 and TensorRT directories shared by the app and the installer probe.

``installer/pin_runtime.py`` is extracted before the Clarity package exists,
so it loads this file by path. The scan here is the only directory list.
"""
from __future__ import annotations

import fnmatch
import os
import site
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import onnx
from onnx import helper

# ``*.so.13`` and ``libnvinfer.so*`` are the Linux names of the same libraries.
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
# ONNX Runtime 1.31 rejects an IR version above 13. The onnx helper default
# is IR 14, so the probe sets 10. Opset 13 defines Identity and is valid
# under IR 10 (the checker only requires the IR to be at least the opset's
# minimum, which for opset 13 is 7).
IDENTITY_IR_VERSION = 10
IDENTITY_OPSET = 13

_T = TypeVar("_T")


def site_package_roots() -> list[Path]:
    """Directories that can hold the CUDA and TensorRT wheels."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            _add_root(roots, seen, Path(value))
    try:
        packages = site.getsitepackages()
    except (AttributeError, OSError):
        return roots
    for item in packages:
        _add_root(roots, seen, Path(item))
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


def format_missing_runtime(folders: Sequence[Path]) -> str:
    """Name the required libraries ``folders`` does not contain.

    Args:
        folders: Directories already chosen by :func:`native_gap_directories`.

    Returns:
        One sentence. It lists whichever of ``cudart64_13``, ``cublas64_13``,
        ``cublasLt64_13``, ``cudnn*``, and ``nvinfer_10`` is absent.
    """
    missing = missing_runtime_labels(folders)
    if not missing:
        return (
            "CUDA and TensorRT libraries were on the DLL search path, "
            "but the session did not keep a GPU provider"
        )
    return "Missing from the DLL search path: " + ", ".join(missing)


def format_probe_failure(folders: Sequence[Path], details: Sequence[str], ort_log: str = "") -> str:
    """Failure text for a session that did not stay on TensorRT or CUDA.

    Args:
        folders: Directories the probe registered.
        details: Loader or provider errors from this attempt.
        ort_log: Stderr captured around the session call.

    Returns:
        The missing-library sentence, the loader errors, and one ONNX Runtime
        log line when ``ort_log`` contains one.
    """
    parts = [format_missing_runtime(folders)]
    body = "; ".join(item.strip() for item in details if item.strip())
    if body:
        parts.append(body)
    line = ort_log_line(ort_log)
    if line:
        parts.append(line)
    return ". ".join(parts)


def ort_log_line(captured: str) -> str:
    """Return one ONNX Runtime loader line from captured stderr.

    Args:
        captured: Text written to the process stderr during the session call.

    Returns:
        The last line that names ONNX Runtime or a DLL load, otherwise the
        last non-empty line. Empty when nothing was captured.
    """
    lines = [line.strip() for line in captured.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        lowered = line.lower()
        if "onnxruntime" in lowered or lowered.startswith("[e:") or "loadlibrary" in lowered:
            return line
    return lines[-1]


def missing_runtime_labels(folders: Sequence[Path]) -> list[str]:
    """Labels in :data:`_REQUIRED_LIBS` that ``folders`` does not contain.

    Args:
        folders: Directories to read.

    Returns:
        The missing labels, in list order.
    """
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


def build_identity_model() -> bytes:
    """Serialize a one-node Identity model at IR 10 and opset 13.

    The onnx helper writes its current default IR version, which ONNX
    Runtime 1.31 rejects when that default is above 13. The bytes here set
    the IR version after the helper returns.

    Returns:
        A ``ModelProto`` payload.
    """
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph(
        [node],
        "identity",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", IDENTITY_OPSET)])
    model.ir_version = IDENTITY_IR_VERSION
    return bytes(model.SerializeToString())


def call_capturing_stderr(action: Callable[[], _T]) -> tuple[_T, str]:
    """Run ``action`` and return its result plus text written to stderr.

    ONNX Runtime writes loader errors to the C stderr stream. Python's
    ``sys.stderr`` does not see those writes, so this duplicates file
    descriptor 2 for the duration of the call and restores it afterwards.

    Args:
        action: Work that may load an ONNX Runtime session.

    Returns:
        The action's return value and the captured stderr text.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    sink = tempfile.TemporaryFile()
    try:
        os.dup2(sink.fileno(), 2)
        try:
            result = action()
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
    finally:
        os.close(saved)
        sink.seek(0)
        captured = sink.read().decode("utf-8", errors="replace")
        sink.close()
    return result, captured


def _add_root(roots: list[Path], seen: set[Path], folder: Path) -> None:
    try:
        resolved = folder.resolve()
    except OSError:
        return
    if resolved in seen:
        return
    seen.add(resolved)
    roots.append(folder)


def _add_dir(found: list[Path], seen: set[Path], folder: Path) -> None:
    if not folder.is_dir():
        return
    _add_root(found, seen, folder)


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
