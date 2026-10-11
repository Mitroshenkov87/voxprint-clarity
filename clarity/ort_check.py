"""ONNX Runtime provider check.

The engine distribution depends on the CPU ``onnxruntime`` wheel, and
``ensure_runtime`` then installs ``onnxruntime-gpu`` plus an unpinned
``tensorrt-cu13``. When both Runtime wheels are installed, the CPU build
wins and only the Azure and CPU providers remain. ONNX Runtime 1.31's
TensorRT EP loads ``nvinfer_10``, so TensorRT 11 does not satisfy it.
"""
from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence

try:
    import onnxruntime as ort
except ImportError:
    ort = None

ORT_GPU_VERSION = "1.31.0"
TRT_LIBS_VERSION = "10.14.1.48"
TRT_META_VERSION = "10.14.1.48"
CPU_DIST = "onnxruntime"
GPU_DIST = "onnxruntime-gpu"
REQUIRED_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")


def describe_provider_failure(providers: Sequence[str], distributions: set[str]) -> str | None:
    """Explain a Runtime install that cannot run TensorRT.

    Args:
        providers: ``onnxruntime.get_available_providers()``.
        distributions: Lower-case package names from ``importlib.metadata``.

    Returns:
        A failure message, or None when CUDA and TensorRT providers are present
        and the CPU wheel is absent.
    """
    folded = {name.lower() for name in distributions}
    if CPU_DIST in folded:
        return (
            "The CPU onnxruntime wheel is installed beside onnxruntime-gpu, so import "
            "onnxruntime loads the CPU build (Azure and CPU providers only). Uninstall "
            "onnxruntime and keep onnxruntime-gpu."
        )
    missing = [name for name in REQUIRED_PROVIDERS if name not in providers]
    if missing:
        joined = ", ".join(missing)
        return (
            f"ONNX Runtime is missing {joined}. "
            f"onnxruntime-gpu {ORT_GPU_VERSION} needs TensorRT {TRT_LIBS_VERSION} "
            "(nvinfer_10). TensorRT 11 does not provide that DLL."
        )
    return None


def installed_distributions() -> set[str]:
    """Lower-case names of the installed distributions."""
    names: set[str] = set()
    for dist in importlib.metadata.distributions():
        raw = dist.metadata["Name"] if dist.metadata is not None else ""
        if raw:
            names.add(raw.lower())
    return names


def provider_failure() -> str | None:
    """Load ONNX Runtime and return a failure message, or None when it is usable.

    Returns:
        The message from :func:`describe_provider_failure`, or a message when
        the GPU wheel cannot be imported.
    """
    if ort is None:
        return "onnxruntime-gpu is not installed."
    providers = [str(name) for name in ort.get_available_providers()]
    return describe_provider_failure(providers, installed_distributions())
