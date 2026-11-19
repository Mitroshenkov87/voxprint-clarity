"""Exit 0 when a real TensorRT session and a real CUDA session both stay active.

The rented GPU VM runs this after ``python -m clarity --selftest``. CI uses
``--selftest --no-gpu`` and does not run this file. Importing ``clarity``
registers the CUDA 13 and TensorRT directories before ONNX Runtime loads.
"""
from __future__ import annotations

import clarity  # noqa: F401
from clarity.gpu_runtime import main as probe_sessions


def main() -> int:
    """Open both providers and return 1 when one of them falls back to CPU.

    Returns:
        0 when both sessions keep the requested provider, otherwise 1.
    """
    return probe_sessions()


if __name__ == "__main__":
    raise SystemExit(main())
