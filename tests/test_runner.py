"""The Start button's job, with fake ffmpeg pipes. No GPU and no engine package."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from clarity.errors import Cancelled
from clarity.i18n import tr
from clarity.models import ClarityModel
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings
from clarity.pipeline.runner import CancelToken, Control, JobSpec, run_job
from clarity.ui.main_window import MainWindow

_PROBE = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 4,
            "height": 4,
            "avg_frame_rate": "1/1",
            "nb_frames": "2",
            "color_transfer": "bt709",
            "field_order": "progressive",
        }
    ]
}
_ENCODERS = " V..... av1_nvenc             NVIDIA NVENC AV1 encoder\n"


def _identity(model: ClarityModel, nchw: np.ndarray) -> np.ndarray:
    rgb = nchw[:, :3]
    scale = model.scale
    if scale == 1:
        return rgb
    return np.repeat(np.repeat(rgb, scale, axis=2), scale, axis=3)


class _Bytes:
    def __init__(self, payload: bytes = b"") -> None:
        self._payload = payload
        self._offset = 0
        self.written = bytearray()

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        part = self._payload[self._offset : self._offset + size]
        self._offset += len(part)
        return part

    def write(self, payload: bytes) -> int:
        self.written += payload
        return len(payload)

    def close(self) -> None:
        return None


class _Proc:
    def __init__(self, stdout: _Bytes | None = None, stdin: _Bytes | None = None) -> None:
        self.stdout = stdout
        self.stdin = stdin
        self.stderr = None
        self.killed = False

    def poll(self) -> int | None:
        if self.killed:
            return -9
        return None

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return -9 if self.killed else 0


class _Lock:
    def __init__(self) -> None:
        self.name = ""
        self.held = False
        self.spawned_while_held = 0

    def __call__(self, job: str, eta_s: float) -> _Lock:
        del eta_s
        self.name = job
        return self

    def __enter__(self) -> None:
        self.held = True
        return None

    def __exit__(self, *_exc: object) -> bool:
        self.held = False
        return False


def _control(
    lock: _Lock,
    spawned: list[list[str]],
    frames: bytes,
    partial: Path,
) -> tuple[Control, _Bytes, list[_Proc]]:
    encoded = _Bytes()
    procs: list[_Proc] = []

    def run_text(argv: list[str]) -> str:
        if argv[0] == "ffprobe":
            return json.dumps(_PROBE)
        return _ENCODERS

    def spawn(argv: list[str]) -> _Proc:
        assert lock.held
        lock.spawned_while_held += 1
        spawned.append(argv)
        if "pipe:1" in argv:
            proc = _Proc(stdout=_Bytes(frames))
        else:
            partial.write_bytes(b"mkv-partial")
            proc = _Proc(stdin=encoded)
        procs.append(proc)
        return proc

    return (
        Control(infer=_identity, lock=lock, spawn=spawn, run_text=run_text, device=False),
        encoded,
        procs,
    )


def _settings(**overrides: object) -> Settings:
    settings = Settings(
        mode="strong",
        sharpen=0,
        white_balance=False,
        levels=False,
        tile=4,
        overlap=0,
        batch=2,
        copy_audio=True,
        container=Container.MKV,
        codec=VideoCodec.AV1,
    )
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def test_run_job_decodes_restores_and_encodes_av1(tmp_path: Path) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    destination = tmp_path / "out.mkv"
    partial = destination.with_suffix(destination.suffix + ".partial")
    frames = bytes(4 * 4 * 3 * 2)
    lock = _Lock()
    spawned: list[list[str]] = []
    control, encoded, procs = _control(lock, spawned, frames, partial)
    run_job(JobSpec(source, destination, _settings()), control)
    assert procs[0].killed is False
    assert procs[1].killed is False
    decode, encode = spawned
    assert "cuda" in decode
    assert "pipe:1" in decode
    assert "av1_nvenc" in encode
    assert encode[encode.index("-c:a") + 1] == "copy"
    assert not any("264" in part for part in encode)
    assert lock.name == "clarity:restore"
    assert lock.spawned_while_held == 2
    assert lock.held is False
    assert bytes(encoded.written) == frames
    assert destination.read_bytes() == b"mkv-partial"
    assert not partial.exists()


def test_run_job_cancel_stops_before_the_next_chunk(tmp_path: Path) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    destination = tmp_path / "out.mkv"
    partial = destination.with_suffix(destination.suffix + ".partial")
    frames = bytes(4 * 4 * 3 * 2)
    lock = _Lock()
    spawned: list[list[str]] = []
    control, encoded, procs = _control(lock, spawned, frames, partial)
    token = CancelToken()
    seen = {"n": 0}

    def progress(fraction: float, text: str) -> None:
        del fraction, text
        seen["n"] += 1
        token.cancel()

    with pytest.raises(Cancelled):
        run_job(
            JobSpec(source, destination, _settings(max_frames=1)),
            control,
            token=token,
            progress=progress,
        )
    assert seen["n"] == 1
    assert len(encoded.written) == 4 * 4 * 3
    assert not partial.exists()
    assert not destination.exists()
    assert all(proc.killed for proc in procs)


def _app() -> QApplication:
    current = QApplication.instance()
    if isinstance(current, QApplication):
        return current
    return QApplication([])


def _pump(window: MainWindow, until: str) -> None:
    app = _app()
    deadline = time.monotonic() + 5
    while window.status.text() != until and time.monotonic() < deadline:
        app.processEvents()
        thread = window._thread
        if thread is not None:
            thread.wait(30)
    app.processEvents()


def test_start_button_runs_the_injected_job(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    seen: list[JobSpec] = []

    def runner(spec: JobSpec, control: object = None, *, token: CancelToken | None = None, progress: object = None) -> None:
        del control, token
        seen.append(spec)
        assert callable(progress)
        progress(0.5, "1/2")
        progress(1.0, "2/2")

    app = _app()
    window = MainWindow(runner=runner)
    window.source.setText(str(source))
    window.show()
    app.processEvents()
    window.start.click()
    _pump(window, tr("status.done"))
    assert window.progress.value() == 100
    assert window.status.text() == tr("status.done")
    assert window.start.text() == tr("start")
    spec = seen[0]
    assert spec.settings.codec is VideoCodec.AV1
    assert spec.settings.container is Container.MKV
    assert spec.settings.copy_audio is True
    assert spec.settings.mode == "clean"
    assert spec.settings.sharpen == pytest.approx(0.35)
    assert spec.destination == source.with_suffix(".clarity.mkv")
    window.close()


def test_start_button_cancel(tmp_path: Path) -> None:
    source = tmp_path / "clip.mkv"
    source.write_bytes(b"video")
    started = threading.Event()

    def runner(spec: JobSpec, control: object = None, *, token: CancelToken | None = None, progress: object = None) -> None:
        del spec, control, progress
        assert token is not None
        started.set()
        while not token.is_set():
            time.sleep(0.01)
        raise Cancelled("cancelled")

    _app()
    window = MainWindow(runner=runner)
    window.output.setText(str(tmp_path / "out.mkv"))
    window.source.setText(str(source))
    window.show()
    window.start.click()
    assert started.wait(2)
    window.start.click()
    assert window.status.text() == tr("status.cancelling")
    _pump(window, tr("status.cancelled"))
    assert window.status.text() == tr("status.cancelled")
    assert window.start.text() == tr("start")
    assert window.start.isEnabled()
    window.close()
