"""VOXPrint AI Clarity: lightweight non-generative video restoration.

``prepare_gpu_runtime`` runs before any other Clarity import so a later
``voxprint_engine.session`` import does not load ONNX Runtime first.
"""
from __future__ import annotations

from clarity.appinfo import APP_VERSION
from clarity.gpu_runtime import prepare_gpu_runtime

# CUDA 13 and TensorRT have to be registered before onnxruntime is imported.
prepare_gpu_runtime()

__version__ = APP_VERSION

__all__ = ["__version__"]
