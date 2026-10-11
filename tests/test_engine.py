"""Tests that need the installed ``voxprint-engine`` package.

CI installs ``voxprint_engine`` at tag ``engine-v1.4.0`` before this module runs.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("voxprint_engine")

from voxprint_common.gpu_probe import GpuDevice
from voxprint_engine.api import sysmem_spill, temporal_windows, tile_frames
from voxprint_engine.video import parse_shared_usage, run_batch

from clarity.engine import OWNER, assess_launch, restore_message


def test_temporal_windows_cover_every_frame_once() -> None:
    for count, length, overlap in ((1, 1, 0), (10, 1, 0), (10, 4, 1), (2, 8, 2), (20, 8, 3)):
        emitted: list[int] = []
        for window in temporal_windows(count, length, overlap):
            for slot in range(window.keep_start, window.keep_stop):
                emitted.append(window.input_start + slot)
        assert emitted == list(range(count))


def test_run_batch_keeps_the_batch_axis() -> None:
    class _Input:
        name = "images"
        type = "tensor(float16)"

    class _Session:
        def get_inputs(self) -> list[_Input]:
            return [_Input()]

        def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert feed["images"].dtype == np.float16
            return [feed["images"].astype(np.float32) + 1]

    batch = np.zeros((3, 3, 4, 5), dtype=np.float32)
    out = run_batch(_Session(), batch)
    assert out.shape == (3, 3, 4, 5)
    assert out.dtype == np.float32
    assert float(out.mean()) == pytest.approx(1.0)


def test_tile_frames_identity_roundtrip() -> None:
    rng = np.random.default_rng(0)
    frames = rng.random((2, 30, 40, 3), dtype=np.float32)

    def infer(batch: np.ndarray) -> np.ndarray:
        return batch

    restored = tile_frames(frames, infer, tile=16, overlap=4, scale=1, batch=3)
    assert restored.shape == frames.shape
    assert np.allclose(restored, frames, atol=1e-5)


def test_sysmem_spill_threshold() -> None:
    assert sysmem_spill(None, 0) is False
    assert sysmem_spill(300, 0, threshold=256) is True
    assert sysmem_spill(100, 0, threshold=256) is False
    assert parse_shared_usage('"counter","4096"\n') == 4096


def test_clarity_refuses_a_card_below_ada(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launch gate is ``voxprint_common.gpu_probe``, and 8.6 still refuses."""
    low = [
        GpuDevice(
            index=0,
            name="NVIDIA GeForce RTX 3060",
            driver="610.00",
            compute=(8, 6),
            memory_mib=12288,
        )
    ]
    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", lambda runner=None: low)
    gate = assess_launch()
    assert gate.ok is False
    assert "8.6" in gate.message

    ada = [
        GpuDevice(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            driver="610.00",
            compute=(8, 9),
            memory_mib=24564,
        )
    ]
    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", lambda runner=None: ada)
    allowed = assess_launch()
    assert allowed.ok is True


def test_restore_message_uses_the_clarity_owner() -> None:
    message = restore_message(id="1", frames_npy="in.npy", out_npy="out.npy", onnx="model.onnx", owner="enlarge")
    assert message["op"] == "restore_batch"
    assert message["owner"] == OWNER == "clarity"
