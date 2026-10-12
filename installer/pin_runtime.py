"""Re-pin a venv after the engine package is installed.

The engine ``gpu`` extra installs ``onnxruntime-gpu`` and
``tensorrt-cu13-libs``. The CPU distribution named ``onnxruntime`` is a
fallback only: when that distribution is present it is uninstalled, and the
install fails if CUDA and TensorRT providers are still missing. Torch and
spandrel still come from ``requirements-gpu.txt``.

The script is stdlib-only so the Windows installer can run it before the
Clarity package is importable.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

SPANDREL_SPEC = "spandrel>=0.4,<0.5"


class PinError(RuntimeError):
    """The runtime pin could not finish."""


_CPU_DIST = "onnxruntime"
_DIST_PROBE = (
    "import importlib.metadata as meta\n"
    "names = {\n"
    "    dist.metadata['Name'].lower()\n"
    "    for dist in meta.distributions()\n"
    "    if dist.metadata and dist.metadata.get('Name')\n"
    "}\n"
    "raise SystemExit(0 if 'onnxruntime' in names else 1)\n"
)
_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def cpu_uninstall_args() -> list[str]:
    """Pip arguments that remove only the CPU ONNX Runtime distribution."""
    return ["uninstall", "-y", _CPU_DIST]


def torch_args(requirements: Path) -> list[str]:
    """Pip arguments that install Clarity's CUDA torch file and spandrel.

    Args:
        requirements: ``requirements-gpu.txt``.

    Returns:
        Arguments after ``pip``.
    """
    return ["install", "-r", str(requirements), SPANDREL_SPEC]


def load_lock(path: Path) -> dict[str, Any]:
    """Read the runtime lock.

    Args:
        path: ``runtime_lock.json``.

    Returns:
        The JSON object.

    Raises:
        PinError: When the file is missing or not an object.
    """
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinError(f"could not read {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise PinError("runtime lock must be an object")
    return body


def pip_command(python: str, args: list[str], *, uv: str = "") -> list[str]:
    """Build one pip invocation.

    ``uv pip`` does not prompt, so the pip ``-y`` flag is left off.
    ``--python`` selects the venv. A ``uv venv`` without ``--seed`` has no
    pip module, and that is the virtualenv ``vm_test`` creates.

    Args:
        python: The interpreter whose site-packages change.
        args: Arguments after ``pip``.
        uv: The uv executable. Empty means ``python -m pip``.

    Returns:
        The argv list.
    """
    if uv:
        cleaned = [part for part in args if part != "-y"]
        return [uv, "pip", *cleaned, "--python", python]
    return [python, "-m", "pip", *args]


def _run(cmd: list[str], *, required: bool) -> None:
    completed = subprocess.run(cmd, check=False)
    if required and completed.returncode != 0:
        shown = " ".join(cmd[:6])
        raise PinError(f"{shown} exited {completed.returncode}")


def cpu_onnxruntime_installed(python: str) -> bool:
    """Return whether the CPU ``onnxruntime`` distribution is installed.

    ``onnxruntime-gpu`` is a different distribution name and does not count.

    Args:
        python: The venv interpreter.

    Returns:
        True when that distribution is present.

    Raises:
        PinError: When the probe cannot run.
    """
    completed = subprocess.run([python, "-c", _DIST_PROBE], check=False, capture_output=True, **_TEXT)
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = (completed.stderr or completed.stdout or "").strip()
    raise PinError(detail or "could not list installed distributions")


def require_gpu_providers(python: str) -> None:
    """Fail unless a real TensorRT session and a real CUDA session both stay active.

    The child runs ``--probe-sessions`` so DLL registration happens before
    ONNX Runtime is imported. Provider names from ``get_available_providers``
    are not enough.

    Args:
        python: The venv interpreter.

    Raises:
        PinError: When either session fails or the probe cannot run. The
            probe lists which of ``cudart64_13``, ``cublas64_13``,
            ``cublasLt64_13``, ``cudnn*``, and ``nvinfer_10`` is missing.
    """
    script = str(Path(__file__).resolve())
    completed = subprocess.run(
        [python, script, "--probe-sessions"],
        check=False,
        capture_output=True,
        **_TEXT,
    )
    if completed.returncode == 0:
        return
    detail = (completed.stdout or completed.stderr or "").strip()
    raise PinError(
        "CPU onnxruntime was uninstalled, but CUDAExecutionProvider or "
        f"TensorrtExecutionProvider is still missing. {detail}".strip()
    )


def apply_pin(python: str, lock_path: Path, requirements: Path, *, uv: str = "") -> None:
    """Install torch and spandrel, then drop a CPU ONNX Runtime wheel if one is present.

    Args:
        python: The venv interpreter.
        lock_path: ``runtime_lock.json``. Read so a damaged lock fails setup.
        requirements: ``requirements-gpu.txt``.
        uv: When set, run ``uv pip --python`` instead of ``python -m pip``.

    Raises:
        PinError: When pip fails, the CPU wheel remains, or the providers are missing.
    """
    load_lock(lock_path)
    _run(pip_command(python, torch_args(requirements), uv=uv), required=True)
    if cpu_onnxruntime_installed(python):
        _run(pip_command(python, cpu_uninstall_args(), uv=uv), required=True)
        if cpu_onnxruntime_installed(python):
            raise PinError("CPU onnxruntime is still installed after uninstall")
    require_gpu_providers(python)


def probe_sessions() -> int:
    """Open TensorRT and CUDA in this process. 1 names the missing DLL.

    The repo checkout imports ``clarity.gpu_runtime``. The Windows installer
    extracts this file to a temp directory before the app exists, so that
    path uses the engine helper directly.

    Returns:
        0 when both providers stay active, otherwise 1.
    """
    root = Path(__file__).resolve().parent.parent
    if (root / "clarity" / "gpu_runtime.py").is_file():
        sys.path.insert(0, str(root))
        from clarity.gpu_runtime import main as probe_main

        return int(probe_main())
    return _probe_without_app()


def _probe_without_app() -> int:
    """Installer temp copy: the engine helper, then a real session."""
    try:
        from voxprint_engine.trt_libs import prepare_native_runtime
    except ImportError:
        prepare_native_runtime = None
    if prepare_native_runtime is not None:
        prepare_native_runtime()
    # The helper can miss torch/lib. The pinned tag has no helper at all.
    folders = _register_gap()
    try:
        from voxprint_engine.session import session_providers
    except ImportError:
        return _identity_probe(folders)
    missing: list[str] = []
    for name in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
        active, reason = session_providers(name)
        if name not in [str(item) for item in active]:
            missing.append(str(reason or active))
    if missing:
        print(f"{_missing_runtime_message(folders)}. " + "; ".join(missing), flush=True)
        return 1
    return 0


# Same globs as clarity.gpu_runtime. This file is copied alone into the installer temp dir.
_RUNTIME_GLOBS = (
    "*64_13.dll",
    "nvinfer*.dll",
    "cudnn*.dll",
    "libnvinfer.so*",
    "libcudnn.so*",
    "*.so.13",
)
_REQUIRED_LIBS = (
    ("cudart64_13.dll", "cudart64_13"),
    ("cublas64_13.dll", "cublas64_13"),
    ("cublaslt64_13.dll", "cublasLt64_13"),
    ("cudnn*.dll", "cudnn*"),
    ("nvinfer_10.dll", "nvinfer_10"),
)


def _gap_directories(roots: list[Path]) -> list[Path]:
    """``torch/lib`` plus CUDA 13 and TensorRT directories at any nest depth.

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


def _register_gap() -> list[Path]:
    """Register the gap directories. Keeps the Windows DLL-directory handles.

    Returns:
        The directories that were found.
    """
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            roots.append(Path(value))
    folders = _gap_directories(roots)
    handles: list[object] = []
    if sys.platform == "win32":
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            for folder in folders:
                try:
                    handles.append(add(str(folder)))
                except OSError:
                    continue
    # The list has to stay referenced or Windows drops the search path.
    _register_gap.handles = handles  # type: ignore[attr-defined]
    if folders:
        key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
        current = os.environ.get(key, "")
        prefix = os.pathsep.join(str(folder) for folder in folders)
        os.environ[key] = prefix + (os.pathsep + current if current else "")
    return folders


def _scan_runtime_tree(found: list[Path], seen: set[Path], root: Path) -> None:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        folder = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (folder / name).is_symlink()]
        if _has_runtime_library(filenames):
            _add_dir(found, seen, folder)


def _has_runtime_library(names: list[str]) -> bool:
    for name in names:
        lowered = name.lower()
        if any(fnmatch.fnmatch(lowered, pattern) for pattern in _RUNTIME_GLOBS):
            return True
    return False


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


def _missing_runtime_message(folders: list[Path]) -> str:
    """List the CUDA 13 and TensorRT libraries ``folders`` does not contain."""
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
    if not missing:
        return "CUDA and TensorRT libraries were on the DLL search path, but the session did not keep a GPU provider"
    return "Missing from the DLL search path: " + ", ".join(missing)


def _identity_probe(folders: list[Path]) -> int:
    """One Identity op on TensorRT and on CUDA. Imports ORT after DLL setup."""
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
    for name in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
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
        print(f"{_missing_runtime_message(folders)}. " + "; ".join(errors), flush=True)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Pin the interpreter named on the command line.

    Args:
        argv: Flags, without the program name.

    Returns:
        0 on success, 2 when the lock or pip step fails.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw == ["--probe-sessions"]:
        return probe_sessions()
    parser = argparse.ArgumentParser(prog="pin_runtime")
    parser.add_argument("--python", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--uv", default="", help="uv executable. Uses uv pip when set.")
    args = parser.parse_args(argv)
    try:
        apply_pin(args.python, args.lock, args.requirements, uv=args.uv)
    except PinError as exc:
        print(f"FAILED: step install: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
