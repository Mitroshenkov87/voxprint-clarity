"""Re-pin a venv after the engine package is installed.

The engine distribution depends on the CPU ``onnxruntime`` wheel. That wheel
shadows ``onnxruntime-gpu`` and leaves only the CPU and Azure providers.
TensorRT 10 is a runtime component: ``tensorrt-cu13-libs`` and the
``tensorrt-cu13`` meta package at 10.14.1.48. The versions live in
``installer/runtime_lock.json``. The meta package is installed with
``--no-deps`` because TensorRT 10.14 has no cp314 bindings wheel. An
unpinned ``tensorrt-cu13`` is TensorRT 11.

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
    """The runtime lock cannot build the ONNX Runtime and TensorRT commands."""


def _need(lock: dict[str, Any], key: str) -> str:
    value = lock.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PinError(f"runtime lock is missing {key}")
    text = value.strip()
    if any(char.isspace() for char in text):
        raise PinError(f"runtime lock {key} must be one token")
    return text


def gpu_pin_steps(lock: dict[str, Any]) -> list[tuple[list[str], bool]]:
    """Pip arguments that replace the CPU ONNX Runtime wheel with the lock pair.

    Each item is ``(arguments after "pip", fail when the status is non-zero)``.
    Uninstall steps are not required: a package that is already absent is fine.

    Args:
        lock: The parsed ``runtime_lock.json`` object.

    Returns:
        The uninstall, GPU install, CPU uninstall, and TensorRT meta steps.

    Raises:
        PinError: When a pin is missing or the NVIDIA index is not https.
    """
    ort = _need(lock, "onnxruntime-gpu")
    libs = _need(lock, "tensorrt-cu13-libs")
    meta = _need(lock, "tensorrt-cu13")
    index = _need(lock, "nvidia_index")
    if not index.startswith("https://"):
        raise PinError("nvidia_index must be https")
    return [
        (
            [
                "uninstall",
                "-y",
                "onnxruntime",
                "onnxruntime-gpu",
                "tensorrt",
                "tensorrt-cu13",
                "tensorrt-cu13-bindings",
                "tensorrt-cu13-libs",
            ],
            False,
        ),
        (
            [
                "install",
                "--extra-index-url",
                index,
                f"onnxruntime-gpu=={ort}",
                f"tensorrt-cu13-libs=={libs}",
            ],
            True,
        ),
        (["uninstall", "-y", "onnxruntime"], False),
        (["install", "--no-deps", f"tensorrt-cu13=={meta}"], True),
        (["uninstall", "-y", "onnxruntime"], False),
    ]


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


def apply_pin(python: str, lock_path: Path, requirements: Path, *, uv: str = "") -> None:
    """Install torch and spandrel, then the lock's ONNX Runtime GPU and TensorRT 10.

    Args:
        python: The venv interpreter.
        lock_path: ``runtime_lock.json``.
        requirements: ``requirements-gpu.txt``.
        uv: When set, run ``uv pip --python`` instead of ``python -m pip``.

    Raises:
        PinError: When a required pip command fails or the lock is incomplete.
    """
    lock = load_lock(lock_path)
    _run(pip_command(python, torch_args(requirements), uv=uv), required=True)
    for args, required in gpu_pin_steps(lock):
        _run(pip_command(python, args, uv=uv), required=required)


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
