"""Worker handler for ``restore_batch``. Additive. Enlarge's image ops stay as they are.

This file is part of the patch against voxprint-enlarge 6d00d44. It imports
the engine; Clarity does not copy it into the application package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from voxprint_engine.gpu_lock import gpu_job
from voxprint_engine.providers import choices_for
from voxprint_engine.session import available_providers, open_session
from voxprint_engine.video import restore_frames, run_batch


def handle_restore(message: dict[str, Any]) -> dict[str, Any]:
    """Load a NHWC float ``.npy``, restore it, write the output ``.npy``."""
    job_id = str(message.get("id") or "")
    frames_path = Path(str(message.get("frames_npy") or ""))
    out_path = Path(str(message.get("out_npy") or ""))
    onnx = Path(str(message.get("onnx") or ""))
    if not frames_path.is_file():
        raise FileNotFoundError("frames_npy was not found")
    if not onnx.is_file():
        raise FileNotFoundError("the ONNX file was not found")
    frames = np.load(frames_path, allow_pickle=False)
    if frames.ndim != 4:
        raise ValueError("frames_npy must be NHWC")
    tile = int(message.get("tile") or 512)
    overlap = int(message.get("overlap") or 16)
    batch = int(message.get("batch") or 1)
    scale = int(message.get("scale") or 1)
    length = int(message.get("length") or 1)
    window_overlap = int(message.get("window_overlap") or 0)
    device = str(message.get("device") or "auto")
    owner = str(message.get("owner") or "clarity")
    installed = available_providers()
    choices = choices_for(installed, device)
    session, provider = open_session(onnx, choices)

    def infer(batch_nchw: np.ndarray) -> np.ndarray:
        return run_batch(session, batch_nchw)

    try:
        with gpu_job(f"{owner}:restore", eta_s=600.0, timeout=30.0, owner=owner):
            restored = restore_frames(
                frames.astype(np.float32, copy=False),
                infer,
                tile=tile,
                overlap=overlap,
                scale=scale,
                batch=batch,
                length=length,
                window_overlap=window_overlap,
            )
    finally:
        closer = getattr(session, "close", None)
        if callable(closer):
            closer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, restored)
    return {
        "op": "result",
        "id": job_id,
        "ok": True,
        "message": f"onnx {provider}",
        "dst": str(out_path),
        "engine": "onnx",
        "frames": int(restored.shape[0]),
    }
