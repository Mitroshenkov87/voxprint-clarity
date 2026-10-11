"""TensorRT build progress on the window and on ``--json`` lines."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from clarity.cli import Console
from clarity.engine import open_restore_session
from clarity.i18n import tr
from clarity.progress import progress_stage
from clarity.ui.main_window import MainWindow


def test_progress_stage_names_the_tensorrt_build() -> None:
    assert progress_stage("building the TensorRT engine") == "building_engine"
    assert progress_stage("loading the cached TensorRT engine") == "loading_cached_engine"
    assert progress_stage("TensorRT engine is cached") == "loading_cached_engine"
    assert progress_stage(tr("status.building_engine")) == "building_engine"
    assert progress_stage(tr("status.loading_cached_engine", "ru")) == "loading_cached_engine"
    assert progress_stage("12/48") is None
    assert "minutes" in tr("status.building_engine")


def test_json_progress_includes_the_engine_stage(capsys: pytest.CaptureFixture[str]) -> None:
    console = Console(True)
    console.progress(0.0, tr("status.building_engine"))
    body = json.loads(capsys.readouterr().out)
    assert body["type"] == "progress"
    assert body["fraction"] == 0.0
    assert body["stage"] == "building_engine"
    assert "minutes" in body["message"]
    console.progress(0.0, tr("status.loading_cached_engine"))
    cached = json.loads(capsys.readouterr().out)
    assert cached["stage"] == "loading_cached_engine"
    console.progress(0.25, "12/48")
    frame = json.loads(capsys.readouterr().out)
    assert frame["message"] == "12/48"
    assert "stage" not in frame


def test_window_status_shows_why_the_engine_build_waits() -> None:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window._on_progress(0.0, tr("status.building_engine"))
    assert window.status.text() == tr("status.building_engine")
    window._on_progress(0.0, tr("status.loading_cached_engine"))
    assert window.status.text() == tr("status.loading_cached_engine")
    window._on_progress(0.5, "3/10")
    assert window.status.text() == tr("status.running").format(done="3", total="10")
    window.close()


def test_open_restore_session_reports_the_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Choice:
        def __init__(self, name: str) -> None:
            self.name = name
            self.options = {"trt_engine_cache_path": str(tmp_path)}

    class _Session:
        @staticmethod
        def available_providers() -> list[str]:
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider"]

        @staticmethod
        def trt_build_message(_cache: Path) -> str:
            return "building the TensorRT engine"

        @staticmethod
        def open_session(_path: Path, choices: list[_Choice]) -> tuple[object, str]:
            return object(), choices[0].name

        class EngineBuildError(Exception):
            """Stand-in for the engine build error."""

    class _Providers:
        @staticmethod
        def choices_for(*_args: object, **_kwargs: object) -> list[_Choice]:
            return [_Choice("TensorrtExecutionProvider")]

        @staticmethod
        def cuda_fallback_choices(*_args: object, **_kwargs: object) -> list[_Choice]:
            return [_Choice("CUDAExecutionProvider")]

    def load(name: str) -> type | None:
        if name == "voxprint_engine.session":
            return _Session
        if name == "voxprint_engine.providers":
            return _Providers
        return None

    monkeypatch.setattr("clarity.engine._load_module", load)
    messages: list[str] = []
    session = open_restore_session(tmp_path / "model.onnx", lambda _fraction, message: messages.append(message))
    assert session is not None
    assert messages == [tr("status.building_engine"), tr("status.loading_cached_engine")]
