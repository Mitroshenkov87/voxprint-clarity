"""The product window keeps the old project actions and the Restore button."""
from __future__ import annotations

import os
import re
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
    QSpinBox,
    QWidget,
)

from clarity.i18n import tr
from clarity.pipeline.encode import VideoCodec
from clarity.pipeline.job import Settings, models_for
from clarity.project import load_project, load_recent
from clarity.ui.job_presets import job_from_state
from clarity.ui.shell import MainWindow, reference_shell
from clarity.ui.shell_dress import PANEL_WIDTH, gpu_selftest_command, gpu_selftest_passed
from voxprint_common.ui.presets import PresetMatch

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
    # mdl.json lists real files now; this test is about settings, not downloads.
    monkeypatch.setattr("clarity.ui.shell.models_ready", lambda: True)
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
    assert drop.text() == "clip.mkv · Choose another video"
    assert "Drop a video" not in drop.text()
    assert str(source) not in drop.text()
    assert str(source) in drop.toolTip()
    note = window.findChild(QLabel, "simpleNote")
    assert note is not None
    assert note.text() == "Output: clip.clarity.mkv"
    assert note.toolTip() == str(source.with_suffix(".clarity.mkv"))
    assert "/" not in note.text()
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
    assert re.fullmatch(r"clip\.mkv\n\d{2}:\d{2}\nLight", thumb.text())
    assert str(source) in thumb.toolTip()
    assert "Light" in thumb.toolTip()
    _assert_expert_labels(
        window,
        {"Softer 2× mix", "AV1 quality (CRF)", "Block removal", "Denoise strength (σ)"},
        "Auto",
    )
    assert _expert_groups(window)["Block removal"] == "Restoration"
    assert _expert_groups(window)["Softer 2× mix"] == "Upscale"
    assert _expert_groups(window)["AV1 quality (CRF)"] == "Encoding"
    _assert_simple_chips(window, "en")
    window.set_language_code("ru")
    app.processEvents()
    app.processEvents()
    chrome = window.centralWidget()
    assert chrome is not None
    russian_drop = chrome.findChild(QLabel, "fileDrop")
    assert russian_drop is not None
    assert russian_drop.text() == "clip.mkv · Выбрать другое видео"
    assert "Перетащите видео" not in russian_drop.text()
    russian_note = chrome.findChild(QLabel, "simpleNote")
    assert russian_note is not None
    assert russian_note.text() == "Результат: clip.clarity.mkv"
    assert russian_note.toolTip() == str(source.with_suffix(".clarity.mkv"))
    russian_thumb = chrome.findChild(QPushButton, "thumb-item-1")
    assert russian_thumb is not None
    assert re.fullmatch(r"clip\.mkv\n\d{2}:\d{2}\nЛёгкая", russian_thumb.text())
    rebuilt = window._screen
    assert rebuilt is not None
    rebuilt.set_expert(True)
    app.processEvents()
    _assert_expert_labels(
        chrome,
        {"Мягкая смесь ×2", "Качество AV1 (CRF)", "Удаление блоков", "Сила шумоподавления (σ)"},
        "Авто",
    )
    assert _expert_groups(chrome)["Удаление блоков"] == "Восстановление"
    assert _expert_groups(chrome)["Мягкая смесь ×2"] == "Увеличение"
    assert _expert_groups(chrome)["Качество AV1 (CRF)"] == "Кодирование"
    _assert_simple_chips(chrome, "ru")
    toggle = chrome.findChild(QPushButton, "expertToggle")
    assert toggle is not None
    assert toggle.text() == "Режим эксперта"
    font = QFont(toggle.font())
    font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(toggle.text())
    assert toggle.minimumWidth() >= advance + 32
    window.close()


def _expert_groups(root: QWidget) -> dict[str, str]:
    panel = root.findChild(QFrame, "expertPanel")
    assert panel is not None
    form = panel.findChild(QWidget, "expertForm")
    assert form is not None
    layout = form.layout()
    assert layout is not None
    current = ""
    found: dict[str, str] = {}
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QLabel) and widget.objectName() == "settingGroup":
            current = widget.text()
            continue
        nested = item.layout()
        if nested is None:
            continue
        first = nested.itemAt(0)
        label = None if first is None else first.widget()
        if isinstance(label, QLabel):
            found[label.text()] = current
    return found


def _assert_expert_labels(root: QWidget, names: set[str], auto: str) -> None:
    texts = {label.text() for label in root.findChildren(QLabel, "panel")}
    assert names <= texts
    for label in root.findChildren(QLabel, "panel"):
        assert label.wordWrap() is False
        advance = QFontMetrics(label.font()).horizontalAdvance(label.text())
        assert label.minimumWidth() >= advance
    tile = root.findChild(QSpinBox, "job.tile")
    assert tile is not None
    assert tile.value() == 0
    assert tile.specialValueText() == auto
    assert tile.text() == auto


def _assert_simple_chips(root: QWidget, language: str) -> None:
    expected = {
        "en": {
            "chip-cleanup-light": "Light",
            "chip-cleanup-medium": "Medium",
            "chip-cleanup-strong": "Strong",
            "chip-fix-noise": "Noise",
            "chip-fix-blocks": "Compression blocks",
            "chip-fix-colours": "Colours",
            "chip-sharpness-off": "Off",
            "chip-sharpness-normal": "Normal",
            "chip-sharpness-sharper": "Sharper",
            "chip-size-same": "Same size",
            "chip-size-2x": "2x larger",
            "chip-quality-smaller": "Smaller file",
            "chip-quality-balanced": "Balanced",
            "chip-quality-best": "Best quality",
            "chip-file-mkv": "MKV",
            "chip-file-mp4": "MP4",
        },
        "ru": {
            "chip-cleanup-light": "Лёгкая",
            "chip-cleanup-medium": "Средняя",
            "chip-cleanup-strong": "Сильная",
            "chip-fix-noise": "Шум",
            "chip-fix-blocks": "Блоки сжатия",
            "chip-fix-colours": "Цвета",
            "chip-sharpness-off": "Не трогать",
            "chip-sharpness-normal": "Обычная",
            "chip-sharpness-sharper": "Чётче",
            "chip-size-same": "Тот же размер",
            "chip-size-2x": "В два раза больше",
            "chip-quality-smaller": "Меньше файл",
            "chip-quality-balanced": "Баланс",
            "chip-quality-best": "Лучше качество",
            "chip-file-mkv": "MKV",
            "chip-file-mp4": "MP4",
        },
    }[language]
    host = root.findChild(QWidget, "mainColumn")
    assert host is not None
    for name, text in expected.items():
        button = host.findChild(QPushButton, name)
        assert button is not None, name
        assert button.text() == text
        assert button.geometry().right() <= host.width() + 1
    noise = host.findChild(QPushButton, "chip-fix-noise")
    blocks = host.findChild(QPushButton, "chip-fix-blocks")
    colours = host.findChild(QPushButton, "chip-fix-colours")
    plain = host.findChild(QPushButton, "chip-fix-plain")
    quality = host.findChild(QLabel, "preset-quality")
    assert quality is not None
    assert quality.text() == ("File quality" if language == "en" else "Качество файла")
    assert noise is not None and blocks is not None and colours is not None and plain is not None
    assert noise.isChecked() and blocks.isChecked() and colours.isChecked()
    assert plain.isHidden()


def test_simple_chips_write_the_shared_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    app = _app()
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    host = window.findChild(QWidget, "mainColumn")
    assert host is not None
    colours = host.findChild(QPushButton, "chip-fix-colours")
    noise = host.findChild(QPushButton, "chip-fix-noise")
    blocks = host.findChild(QPushButton, "chip-fix-blocks")
    medium = host.findChild(QPushButton, "chip-cleanup-medium")
    sharper = host.findChild(QPushButton, "chip-sharpness-sharper")
    smaller = host.findChild(QPushButton, "chip-quality-smaller")
    assert None not in {colours, noise, blocks, medium, sharper, smaller}
    assert colours is not None and noise is not None and blocks is not None
    assert medium is not None and sharper is not None and smaller is not None
    colours.click()
    app.processEvents()
    assert window.settings_state.get("job.white_balance") is False
    assert window.settings_state.get("job.levels") is False
    assert noise.isChecked() and blocks.isChecked() and not colours.isChecked()
    noise.click()
    blocks.click()
    assert window.settings_state.get("job.white_balance") is False
    medium.click()
    sharper.click()
    smaller.click()
    app.processEvents()
    assert window.settings_state.get("job.mode") == "clean"
    assert window.settings_state.get("job.sigma") == 25
    assert window.settings_state.get("job.sharpen") == 0.6
    assert window.settings_state.get("job.crf") == 32
    assert [model.id for model in models_for(job_from_state(window.settings_state.values))] == [
        model.id for model in models_for(Settings(mode="clean"))
    ]
    window.settings_state.set_value("job.sharpen", 0.5)
    screen = window._screen
    assert screen is not None
    screen.refresh()
    app.processEvents()
    assert window.settings_state.status("sharpness").kind is PresetMatch.CUSTOM
    custom = host.findChild(QPushButton, "chip-sharpness-custom")
    assert custom is not None and custom.isVisible()
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
