"""The product window keeps the old project actions and the Restore button."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from clarity.i18n import tr
from clarity.pipeline.encode import VideoCodec
from clarity.pipeline.job import Settings
from clarity.project import load_project, load_recent
from clarity.ui.shell import MainWindow, reference_shell
from clarity.ui.shell_dress import PANEL_WIDTH, gpu_selftest_command, gpu_selftest_passed

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
    "checkGpu",
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


def test_graphics_card_check_uses_the_gpu_selftest() -> None:
    command = gpu_selftest_command()
    assert command[1:] == ["-m", "clarity", "--selftest"]
    assert "--no-gpu" not in command
    assert gpu_selftest_passed(0, "ffmpeg skipped\nselftest ok\n")
    assert not gpu_selftest_passed(0, "still running")
    assert not gpu_selftest_passed(1, "selftest ok")


def test_screen_uses_the_width_and_shows_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    source = tmp_path / "clip.mkv"
    source.write_bytes(b"frame-bytes")
    app = _app()
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window.resize(1280, 800)
    window.use_source_text(str(source))
    window.show()
    app.processEvents()
    drop = window.findChild(QLabel, "fileDrop")
    assert drop is not None
    assert "clip.mkv" in drop.text()
    assert str(source) not in drop.text()
    assert str(source) in drop.toolTip()
    subtitle = window.findChild(QLabel, "subtitle")
    footer = window.findChild(QLabel, "versionLabel")
    assert subtitle is not None and "16px" in subtitle.styleSheet()
    assert footer is not None and "16px" in footer.styleSheet()
    assert "#2dd4bf" in window.styleSheet()
    assert "QPushButton#generate" in window.styleSheet()
    tools = _menu(window, "Tools")
    assert any(action.text() == "Check the graphics card" for action in tools.actions())
    assert window._layout.generate()
    window._layout.progress(1, 2, "Restoring")
    window._layout.finish("item-1")
    screen = window._screen
    assert screen is not None
    screen.set_expert(True)
    screen.refresh()
    window.resize(1280, 800)
    app.processEvents()
    app.processEvents()
    panel = window.findChild(QFrame, "expertPanel")
    assert panel is not None
    assert panel.width() == PANEL_WIDTH
    groups = [label.text() for label in window.findChildren(QLabel, "settingGroup")]
    assert groups == ["Restoration", "Upscale", "Encoding", "Advanced"]
    models = window.findChild(QPlainTextEdit, "job.restore_models")
    assert models is not None
    assert "Block cleanup" in models.toolTip()
    assert "drunet" not in models.toPlainText()
    compare = window.findChild(QLabel, "compareView")
    assert compare is not None
    assert compare.minimumHeight() >= 200
    assert compare.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert compare.height() > 200
    assert panel.height() > 400
    thumb = window.findChild(QPushButton, "thumb-item-1")
    assert thumb is not None
    assert "clip.mkv" in thumb.text()
    assert thumb.text() != "1"
    stamp = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d")
    assert stamp in thumb.text()
    window.set_language_code("ru")
    app.processEvents()
    app.processEvents()
    chrome = window.centralWidget()
    assert chrome is not None
    toggle = chrome.findChild(QPushButton, "expertToggle")
    assert toggle is not None
    assert toggle.text() == "Режим эксперта"
    font = QFont(toggle.font())
    font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(toggle.text())
    assert toggle.minimumWidth() >= advance + 32
    window.close()


def _menu(window: MainWindow, title: str) -> QMenu:
    for action in window.menuBar().actions():
        menu = action.menu()
        if menu is not None and action.text() == title:
            return menu
    raise AssertionError(title)


def test_reference_shell_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    app = _app()
    shell = reference_shell(state_dir=tmp_path / "reference")
    shell.show()
    app.processEvents()
    shell.close()
