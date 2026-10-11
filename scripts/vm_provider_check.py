"""Exit 0 when the CUDA and TensorRT execution providers both load.

The rented GPU VM runs this after ``python -m clarity --selftest``. CI uses
``--selftest --no-gpu`` and does not run this file.
"""
from __future__ import annotations

import json

from voxprint_engine.session import available_providers

_REQUIRED = ("CUDAExecutionProvider", "TensorrtExecutionProvider")


def main() -> int:
    """Print the provider report and return 1 when one of them is missing.

    Returns:
        0 when both providers loaded, otherwise 1.
    """
    found = [str(name) for name in available_providers()]
    missing = [name for name in _REQUIRED if name not in found]
    print(json.dumps({"type": "providers", "providers": found, "missing": missing}), flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
