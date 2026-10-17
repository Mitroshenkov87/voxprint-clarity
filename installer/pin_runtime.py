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
import importlib.util
import json
import os
import subprocess
import sys
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
    detail = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip())
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
    """Open TensorRT and CUDA in this process.

    The message lists which required libraries are actually missing, and the
    ONNX Runtime log line when the loader wrote one. It does not assume
    ``nvinfer_10.dll`` is the library that failed. The repo checkout imports
    ``clarity.gpu_runtime``. The Windows installer extracts this file to a
    temp directory before the app exists, so that path uses the engine helper
    directly.

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
    gap = _load_gap()
    missing: list[str] = []
    logs: list[str] = []
    for name in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):

        def open_provider(provider: str = name) -> tuple[object, object]:
            return session_providers(provider)

        (active, reason), log = gap.call_capturing_stderr(open_provider)
        if log:
            logs.append(log)
        if name not in [str(item) for item in active]:
            missing.append(str(reason or active))
    if missing:
        print(_failure_text(folders, missing, "\n".join(logs)), flush=True)
        return 1
    return 0


_GAP: Any = None


def _load_gap() -> Any:
    """Return :mod:`clarity.native_gap`, the directory list the worker uses.

    The repo and an installed tree find ``clarity/native_gap.py`` next to
    this file's parent. The Windows installer extracts both files to a temp
    directory before the package exists, so the copy beside this file is
    loaded by path. The import of the package stays inside this function
    because that package is not always on ``sys.path``.

    Returns:
        The loaded module.

    Raises:
        PinError: When neither copy can be loaded.
    """
    global _GAP
    if _GAP is not None:
        return _GAP
    here = Path(__file__).resolve().parent
    candidates = (
        here / "native_gap.py",
        here / "clarity" / "native_gap.py",
        here.parent / "clarity" / "native_gap.py",
    )
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("clarity_native_gap_pin", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GAP = module
        return module
    try:
        import clarity.native_gap as module
    except ImportError as exc:
        raise PinError("native_gap.py was not found beside pin_runtime.py or in clarity/") from exc
    _GAP = module
    return module


def _gap_directories(roots: list[Path]) -> list[Path]:
    """``torch/lib`` plus CUDA 13 and TensorRT directories at any nest depth.

    Args:
        roots: Site-packages directories to scan.

    Returns:
        The directories :func:`clarity.native_gap.native_gap_directories` returns.
    """
    found: list[Path] = _load_gap().native_gap_directories(roots)
    return list(found)


def _register_gap() -> list[Path]:
    """Register the same directories the worker registers.

    Returns:
        The directories that were found.
    """
    gap = _load_gap()
    folders = list(gap.native_gap_directories(gap.site_package_roots()))
    handles: list[object] = []
    if sys.platform == "win32":
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            for folder in folders:
                try:
                    handles.append(add(os.fspath(folder)))
                except OSError:
                    continue
    # The list has to stay referenced or Windows drops the search path.
    _register_gap.handles = handles  # type: ignore[attr-defined]
    if folders:
        key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
        current = os.environ.get(key, "")
        prefix = os.pathsep.join(os.fspath(folder) for folder in folders)
        os.environ[key] = prefix + (os.pathsep + current if current else "")
    return folders


def _missing_runtime_message(folders: list[Path]) -> str:
    """List the CUDA 13 and TensorRT libraries ``folders`` does not contain."""
    return str(_load_gap().format_missing_runtime(folders))


def _failure_text(folders: list[Path], details: list[str], ort_log: str) -> str:
    """Missing libraries, the loader error, and the ORT log line when captured."""
    return str(_load_gap().format_probe_failure(folders, details, ort_log))


def _identity_probe(folders: list[Path]) -> int:
    """One Identity op on TensorRT and on CUDA. Imports ORT after DLL setup.

    ``onnxruntime`` is imported here, after :func:`_register_gap`, because
    importing it earlier makes Windows load the provider DLLs before
    ``torch/lib`` and the CUDA 13 bins are on the search path.
    """
    import numpy as np

    import onnxruntime as ort

    gap = _load_gap()
    payload = gap.build_identity_model()
    sample = np.zeros((1, 1), dtype=np.float32)
    errors: list[str] = []
    logs: list[str] = []
    for name in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):

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

        outcome, log = gap.call_capturing_stderr(open_session)
        if log:
            logs.append(log)
        if isinstance(outcome, str):
            errors.append(outcome)
    if errors:
        print(_failure_text(folders, errors, "\n".join(logs)), flush=True)
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
