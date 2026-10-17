"""Project and Setup pages around the shared video screen.

The video tab is the shell. These pages keep every other control from the
old window, and the CLI flags that window did not show.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import tr

from clarity.i18n import languages
from clarity.models import catalogue_rows
from clarity.ui.icons import suite_icon

_ICON = 22


class ProjectPage(QWidget):
    """Source, output, project files, and the recent list."""

    def __init__(self, window: Any) -> None:
        """Build the project form.

        Args:
            window: The shell window. It supplies the actions and the live paths.
        """
        super().__init__()
        self.setObjectName("projectPage")
        self._window = window
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.source = QLineEdit()
        self.source.setObjectName("sourcePath")
        self.source.setPlaceholderText(tr("source.placeholder"))
        self.source.editingFinished.connect(self._commit_source)
        browse = QPushButton(tr("source.browse"))
        browse.setObjectName("browseSource")
        browse.clicked.connect(window.browse_source)
        root.addLayout(self._row(self.source, browse))

        self.output = QLineEdit()
        self.output.setObjectName("outputPath")
        self.output.setPlaceholderText(tr("output.placeholder"))
        self.output.editingFinished.connect(self._commit_output)
        self.export_button = self._button("exportOutput", "project.export", window.browse_output, "export")
        root.addLayout(self._row(self.output, self.export_button))

        buttons = QHBoxLayout()
        self.open_project_button = self._button("openProject", "project.open", window.browse_project, None)
        self.save_project_button = self._button("saveProject", "project.save", window.save_current, "save")
        self.save_as_button = self._button("saveAs", "project.save_as", window.browse_save, "save-as")
        self.save_copy_button = self._button("saveCopy", "project.save_copy", window.browse_copy, "save-copy")
        self.download_button = self._button("downloadModels", "project.download", window.download_models, "download")
        for button in (
            self.open_project_button,
            self.save_project_button,
            self.save_as_button,
            self.save_copy_button,
            self.download_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.recent = QComboBox()
        self.recent.setObjectName("recent")
        self.recent.activated.connect(self._recent_chosen)
        root.addWidget(self.recent)
        root.addStretch(1)
        self._inputs = (
            self.source,
            self.output,
            browse,
            self.export_button,
            self.open_project_button,
            self.save_project_button,
            self.save_as_button,
            self.save_copy_button,
            self.download_button,
            self.recent,
        )

    def sync(self, *, busy: bool) -> None:
        """Copy the window state into the form.

        Args:
            busy: True while a restore is running. The fields lock.
        """
        self._set_line(self.source, self._window.source_text())
        self._set_line(self.output, self._window.output_text())
        for widget in self._inputs:
            if not widget.hasFocus():
                widget.setEnabled(not busy)
        self._fill_recent()

    def _commit_source(self) -> None:
        self._window.use_source_text(self.source.text())

    def _commit_output(self) -> None:
        self._window.set_output_text(self.output.text())

    def _recent_chosen(self, index: int) -> None:
        value = self.recent.itemData(index)
        if isinstance(value, str) and value:
            self._window.open_project_path(value)

    def _fill_recent(self) -> None:
        if self.recent.view().isVisible():
            return
        current = self.recent.currentData()
        self.recent.blockSignals(True)
        self.recent.clear()
        self.recent.addItem(tr("project.recent"), "")
        for item in self._window.recent_paths():
            self.recent.addItem(Path(item).name, item)
        if isinstance(current, str) and current:
            index = self.recent.findData(current)
            if index >= 0:
                self.recent.setCurrentIndex(index)
        self.recent.blockSignals(False)

    def _row(self, field: QWidget, button: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(field, 1)
        row.addWidget(button)
        return row

    def _button(self, name: str, key: str, handler: Any, icon: str | None) -> QPushButton:
        button = QPushButton(tr(key))
        button.setObjectName(name)
        if icon is not None:
            button.setIcon(suite_icon(icon))
            button.setIconSize(QSize(_ICON, _ICON))
        button.clicked.connect(handler)
        return button

    @staticmethod
    def _set_line(box: QLineEdit, text: str) -> None:
        if box.hasFocus() or box.text() == text:
            return
        box.setText(text)


class SetupPage(QWidget):
    """Codec, sharpen, language, and the CLI actions that had no old button."""

    def __init__(self, window: Any) -> None:
        """Build the setup form.

        Args:
            window: The shell window. It runs each action.
        """
        super().__init__()
        self.setObjectName("setupPage")
        self._window = window
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)

        form = QFormLayout()
        self.codec = QComboBox()
        self.codec.setObjectName("codec")
        self.codec.blockSignals(True)
        for value, label in window.codec_choices():
            self.codec.addItem(label, value)
        self.codec.blockSignals(False)
        self.codec.currentIndexChanged.connect(self._codec_changed)
        form.addRow(tr("codec.label"), self.codec)

        self.sharpen = QSlider(Qt.Orientation.Horizontal)
        self.sharpen.setObjectName("sharpen")
        self.sharpen.setRange(0, 100)
        self.sharpen.valueChanged.connect(window.set_sharpen_percent)
        form.addRow(tr("sharpen.label"), self.sharpen)

        self.language = QComboBox()
        self.language.setObjectName("language")
        self.language.blockSignals(True)
        for code in languages():
            self.language.addItem(tr(f"language.{code}"), code)
        self.language.blockSignals(False)
        self.language.currentIndexChanged.connect(self._language_changed)
        form.addRow(tr("language.label"), self.language)

        self.anime = QCheckBox(tr("anime.label"))
        self.anime.setObjectName("anime")
        self.anime.toggled.connect(window.set_anime)
        form.addRow("", self.anime)
        left.addLayout(form)

        hint = QLabel(tr("upscale.hint"))
        hint.setObjectName("upscaleHint")
        hint.setWordWrap(True)
        left.addWidget(hint)

        paths = QFormLayout()
        self.ffmpeg = QLineEdit()
        self.ffmpeg.setObjectName("ffmpegPath")
        self.ffmpeg.editingFinished.connect(self._commit_paths)
        self.ffprobe = QLineEdit()
        self.ffprobe.setObjectName("ffprobePath")
        self.ffprobe.editingFinished.connect(self._commit_paths)
        self.report = QLineEdit()
        self.report.setObjectName("reportOut")
        self.report.setPlaceholderText(tr("clarity.report.label"))
        self.report.editingFinished.connect(self._commit_paths)
        paths.addRow(tr("clarity.ffmpeg.label"), self.ffmpeg)
        paths.addRow(tr("clarity.ffprobe.label"), self.ffprobe)
        paths.addRow(tr("clarity.report.label"), self.report)
        left.addLayout(paths)
        left.addStretch(1)

        pick = QLabel(tr("clarity.models.pick"))
        pick.setObjectName("modelPick")
        right.addWidget(pick)
        self.models = QListWidget()
        self.models.setObjectName("modelPickList")
        for row in catalogue_rows():
            model_id = str(row["id"])
            key = f"model.{model_id}"
            label = tr(key)
            item = QListWidgetItem(label if label != key else str(row["name"]))
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.models.addItem(item)
        self.models.itemChanged.connect(self._checks_changed)
        right.addWidget(self.models, 1)

        self.repair_button = self._action("repairRuntime", "clarity.action.repair", window.repair_runtime)
        self.install_button = self._action("installSuite", "clarity.action.install_suite", window.install_suite)
        self.manage_button = self._action("manageModels", "clarity.action.manage_models", window.manage_models)
        self.list_button = self._action("listModels", "clarity.action.list_models", window.list_models)
        self.dry_button = self._action("dryRunRestore", "clarity.action.dry_run", window.dry_run_restore)
        self.fetch_button = self._action("fetchModels", "clarity.action.fetch", window.fetch_selected)
        self.fetch_dry_button = self._action("fetchDryRun", "clarity.action.fetch_dry_run", window.fetch_selected_dry_run)
        self.register_button = self._action("registerModels", "clarity.action.register", window.register_models)
        self.unregister_button = self._action("unregisterModels", "clarity.action.unregister", window.unregister_models)
        self.plan_button = self._action("planCheck", "clarity.action.plan_check", window.plan_check)
        self.gpu_button = self._action("checkGpu", "clarity.action.check_gpu", window.check_graphics_card)
        for button in (
            self.repair_button,
            self.install_button,
            self.manage_button,
            self.list_button,
            self.dry_button,
            self.fetch_button,
            self.fetch_dry_button,
            self.register_button,
            self.unregister_button,
            self.plan_button,
            self.gpu_button,
        ):
            right.addWidget(button)

        self._inputs: tuple[QWidget, ...] = (
            self.codec,
            self.sharpen,
            self.language,
            self.anime,
            self.ffmpeg,
            self.ffprobe,
            self.report,
            self.models,
            self.repair_button,
            self.install_button,
            self.manage_button,
            self.list_button,
            self.dry_button,
            self.fetch_button,
            self.fetch_dry_button,
            self.register_button,
            self.unregister_button,
            self.plan_button,
            self.gpu_button,
        )

    def sync(self, *, busy: bool) -> None:
        """Copy codec, sharpen, anime, and the tool paths from the window.

        Args:
            busy: True while a restore is running. The fields lock.
        """
        self._set_combo(self.codec, self._window.codec_value())
        percent = self._window.sharpen_percent()
        if not self.sharpen.hasFocus() and self.sharpen.value() != percent:
            self.sharpen.blockSignals(True)
            self.sharpen.setValue(percent)
            self.sharpen.blockSignals(False)
        anime = self._window.anime_checked()
        if not self.anime.hasFocus() and self.anime.isChecked() != anime:
            self.anime.blockSignals(True)
            self.anime.setChecked(anime)
            self.anime.blockSignals(False)
        self.anime.setEnabled((not busy) and self._window.upscale_enabled())
        self._set_line(self.ffmpeg, self._window.ffmpeg_text())
        self._set_line(self.ffprobe, self._window.ffprobe_text())
        self._set_line(self.report, self._window.report_text())
        selected = set(self._window.selected_model_ids())
        self.models.blockSignals(True)
        for index in range(self.models.count()):
            item = self.models.item(index)
            model_id = item.data(Qt.ItemDataRole.UserRole)
            want = Qt.CheckState.Checked if model_id in selected else Qt.CheckState.Unchecked
            if item.checkState() != want:
                item.setCheckState(want)
        self.models.blockSignals(False)
        for widget in self._inputs:
            if widget is self.anime or widget.hasFocus():
                continue
            widget.setEnabled(not busy)

    def _codec_changed(self) -> None:
        value = self.codec.currentData()
        if isinstance(value, str):
            self._window.set_codec(value)

    def _language_changed(self) -> None:
        if self.language.signalsBlocked():
            return
        value = self.language.currentData()
        if isinstance(value, str):
            self._window.set_language_code(value)

    def _commit_paths(self) -> None:
        self._window.set_tool_paths(self.ffmpeg.text(), self.ffprobe.text(), self.report.text())

    def _checks_changed(self) -> None:
        chosen: list[str] = []
        for index in range(self.models.count()):
            item = self.models.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(model_id, str):
                    chosen.append(model_id)
        self._window.set_selected_model_ids(chosen)

    def _action(self, name: str, key: str, handler: Any) -> QPushButton:
        button = QPushButton(tr(key))
        button.setObjectName(name)
        button.clicked.connect(handler)
        return button

    def select_language(self, code: str) -> None:
        """Show ``code`` without asking the window to switch again.

        Args:
            code: ``en`` or ``ru``.
        """
        self.language.blockSignals(True)
        index = self.language.findData(code)
        if index >= 0:
            self.language.setCurrentIndex(index)
        self.language.blockSignals(False)

    @staticmethod
    def _set_line(box: QLineEdit, text: str) -> None:
        if box.hasFocus() or box.text() == text:
            return
        box.setText(text)

    @staticmethod
    def _set_combo(box: QComboBox, value: str) -> None:
        if box.hasFocus() or box.currentData() == value:
            return
        index = box.findData(value)
        if index < 0:
            return
        box.blockSignals(True)
        box.setCurrentIndex(index)
        box.blockSignals(False)
