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
import json
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
_PROVIDER_PROBE = (
    "import onnxruntime as ort\n"
    "needed = ('CUDAExecutionProvider', 'TensorrtExecutionProvider')\n"
    "found = list(ort.get_available_providers())\n"
    "missing = [name for name in needed if name not in found]\n"
    "if missing:\n"
    "    raise SystemExit('missing ' + ', '.join(missing) + '; found ' + ', '.join(found))\n"
)


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
    completed = subprocess.run([python, "-c", _DIST_PROBE], check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = (completed.stderr or completed.stdout or "").strip()
    raise PinError(detail or "could not list installed distributions")


def require_gpu_providers(python: str) -> None:
    """Fail when CUDA or TensorRT is not an available ONNX Runtime provider.

    Args:
        python: The venv interpreter.

    Raises:
        PinError: When either provider is missing or the probe cannot run.
    """
    completed = subprocess.run([python, "-c", _PROVIDER_PROBE], check=False, capture_output=True, text=True)
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
    if not cpu_onnxruntime_installed(python):
        return
    _run(pip_command(python, cpu_uninstall_args(), uv=uv), required=True)
    if cpu_onnxruntime_installed(python):
        raise PinError("CPU onnxruntime is still installed after uninstall")
    require_gpu_providers(python)


def main(argv: list[str] | None = None) -> int:
    """Pin the interpreter named on the command line.

    Args:
        argv: Flags, without the program name.

    Returns:
        0 on success, 2 when the lock or pip step fails.
    """
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
