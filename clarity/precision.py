"""Declared compute precision for a catalogue model.

DRUNet's spandrel descriptor rejects fp16. SCUNet and the Real-ESRGAN nets
run in fp16. The loader must follow the declaration instead of calling
``.half()`` on every network.
"""
from __future__ import annotations

from typing import Any

from clarity.errors import ClarityError

PRECISIONS = frozenset({"fp16", "bf16", "fp32"})


def apply_precision(network: Any, precision: str) -> Any:
    """Move ``network`` onto the declared precision. It is already on CUDA.

    Args:
        network: A spandrel descriptor or a module with ``half``, ``bfloat16``, and ``float``.
        precision: ``fp16``, ``bf16``, or ``fp32``.

    Returns:
        The same network after the dtype cast.

    Raises:
        ClarityError: When ``precision`` is not one of those three.
    """
    if precision == "fp16":
        return network.half()
    if precision == "bf16":
        return network.bfloat16()
    if precision == "fp32":
        return network.float()
    raise ClarityError(f"unknown precision {precision}")


def torch_dtype_name(precision: str) -> str:
    """The ``torch`` dtype attribute for ``precision``.

    Args:
        precision: ``fp16``, ``bf16``, or ``fp32``.

    Returns:
        ``float16``, ``bfloat16``, or ``float32``.

    Raises:
        ClarityError: When ``precision`` is not one of those three.
    """
    if precision == "fp16":
        return "float16"
    if precision == "bf16":
        return "bfloat16"
    if precision == "fp32":
        return "float32"
    raise ClarityError(f"unknown precision {precision}")
