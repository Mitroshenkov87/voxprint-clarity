"""The product window keeps the old project actions and the Restore button."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from clarity.i18n import tr
from clarity.pipeline.encode import VideoCodec
from clarity.pipeline.job import Settings
from clarity.project import load_project, load_recent
from clarity.ui.shell import MainWindow, reference_shell

_NAMES = (
    "openProject",
    "saveProject",
    "saveAs",
    "saveCopy",
    "exportOutput",
    "downloadModels",
    "browseSource",
    "language",
    "codec",
    "sharpen",
    "anime",
    "repairRuntime",
    "manageModels",
    "installSuite",
    "planCheck",
    "ffmpegPath",
    "ffprobePath",
    "reportOut",
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _isolate(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(folder / "home"))
    monkeypatch.setenv("VOXPRINT_CLARITY_STATE", str(folder / "state"))
    monkeypatch.setenv("VOXPRINT_LANG", "en")


def test_restore_button_and_the_old_controls_are_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    app = _app()
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window.show()
    app.processEvents()
    assert window.start.text().strip() != ""
    for name in _NAMES:
        assert window.findChild(QWidget, name) is not None
    for name in ("saveProject", "saveAs", "saveCopy", "exportOutput", "downloadModels"):
        button = window.findChild(QPushButton, name)
        assert button is not None
        assert not button.icon().isNull()
    window.close()


def test_window_saves_and_opens_a_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"frame-bytes")
    app = _app()
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window.use_source_text(str(source))
    window.set_codec(VideoCodec.HEVC.value)
    window.show()
    app.processEvents()
    project = tmp_path / "clip.vxclarity"
    window.save_project(project)
    assert load_recent()[0] == str(project.resolve())
    loaded = load_project(project)
    assert loaded.settings.codec is VideoCodec.HEVC
    assert Path(window.output_text()).name == "clip.clarity.mkv"
    window.use_source_text("")
    window.set_codec(VideoCodec.AV1.value)
    window.open_project(project)
    assert window.source_text() == str(source.resolve())
    assert window.codec_value() == VideoCodec.HEVC.value
    assert Path(window.output_text()).name == "clip.clarity.mkv"
    recent = window.findChild(QWidget, "recent")
    assert recent is not None
    recent.activated.emit(1)  # type: ignore[attr-defined]
    app.processEvents()
    assert window.source_text() == str(source.resolve())
    assert tr("status.saved")
    window.close()


def test_restore_click_uses_the_default_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    source = tmp_path / "clip.mkv"
    source.write_bytes(b"frame-bytes")
    captured: dict[str, Settings] = {}

    def runner(spec: object, token: object = None, progress: object = None, on_chunk: object = None) -> None:
        del token, on_chunk
        captured["settings"] = spec.settings  # type: ignore[attr-defined]
        if progress is not None:
            progress(0.5, "12/48")  # type: ignore[operator]

    app = _app()
    window = MainWindow(runner=runner)
    window.use_source_text(str(source))
    window.show()
    app.processEvents()
    window.start.click()
    thread = window._thread
    assert thread is not None
    assert thread.wait(5000)
    app.processEvents()
    assert captured["settings"] == Settings()
    window.close()


def test_reference_shell_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    app = _app()
    shell = reference_shell(state_dir=tmp_path / "reference")
    shell.show()
    app.processEvents()
    shell.close()
