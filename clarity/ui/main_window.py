"""The Clarity window. Strings come from the English and Russian catalogues."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Never

from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from clarity.appinfo import APP_NAME, version_label
from clarity.engine import EngineUnavailable
from clarity.errors import Cancelled, ClarityError
from clarity.i18n import current_language, languages, tr
from clarity.models import fetch_weights
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings, models_for
from clarity.pipeline.runner import CancelToken, JobSpec, run_job
from clarity.project import (
    Project,
    ProjectFormatError,
    load_project,
    load_recent,
    remember_recent,
    save_project,
    sync_resume,
    video_fingerprint,
    work_dir_for as project_work_dir,
)
from clarity.ui.acrylic import enable_acrylic
from clarity.ui.icons import suite_icon
from clarity.ui.theme import build_style

Runner = Callable[..., None]


class _JobSignals(QObject):
    progressed = Signal(float, str)
    finished = Signal(bool, str)


class _JobThread(QThread):
    """Runs ``run_job`` off the UI thread. Cancel kills ffmpeg through the token."""

    def __init__(
        self,
        spec: JobSpec,
        token: CancelToken,
        runner: Runner,
        on_chunk: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__()
        self._spec = spec
        self._token = token
        self._runner = runner
        self._on_chunk = on_chunk
        self.signals = _JobSignals()

    def run(self) -> None:
        def report(fraction: float, text: str) -> None:
            self.signals.progressed.emit(fraction, text)

        try:
            self._runner(self._spec, token=self._token, progress=report, on_chunk=self._on_chunk)
        except Cancelled:
            self.signals.finished.emit(False, "cancelled")
        except Exception as exc:
            self.signals.finished.emit(False, str(exc))
        else:
            self.signals.finished.emit(True, "")


class MainWindow(QMainWindow):
    def __init__(self, runner: Runner | None = None) -> None:
        super().__init__()
        self._runner = run_job if runner is None else runner
        self._thread: _JobThread | None = None
        self._token: CancelToken | None = None
        self._closing = False
        self._project_path: Path | None = None
        self._project: Project | None = None
        self.setWindowTitle(APP_NAME)
        self.resize(880, 640)
        self.setStyleSheet(build_style())
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        title = QLabel(tr("app.title"))
        title.setStyleSheet("font-size: 22px; font-weight: 600; background: transparent;")
        subtitle = QLabel(tr("app.subtitle"))
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form_box = QGroupBox(tr("group.source"))
        form = QFormLayout(form_box)
        self.source = QLineEdit()
        self.source.setPlaceholderText(tr("source.placeholder"))
        self.browse = QPushButton(tr("source.browse"))
        self.browse.clicked.connect(self._browse_source)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        source_row.addWidget(self.browse)
        form.addRow(tr("source.label"), source_row)

        self.output = QLineEdit()
        self.output.setPlaceholderText(tr("output.placeholder"))
        self.export_button = QPushButton(tr("project.export"))
        self.export_button.setIcon(suite_icon("export"))
        self.export_button.setIconSize(QSize(22, 22))
        self.export_button.clicked.connect(self._browse_export)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        output_row.addWidget(self.export_button)
        form.addRow(tr("output.label"), output_row)

        self.mode = QComboBox()
        self.mode.addItem(tr("mode.clean"), "clean")
        self.mode.addItem(tr("mode.strong"), "strong")
        form.addRow(tr("mode.label"), self.mode)

        self.container = QComboBox()
        self.container.addItem("MKV", Container.MKV.value)
        self.container.addItem("MP4", Container.MP4.value)
        form.addRow(tr("container.label"), self.container)

        self.codec = QComboBox()
        self.codec.addItem(tr("codec.av1"), VideoCodec.AV1.value)
        self.codec.addItem(tr("codec.hevc"), VideoCodec.HEVC.value)
        form.addRow(tr("codec.label"), self.codec)

        self.language = QComboBox()
        for code in languages():
            self.language.addItem(tr(f"language.{code}"), code)
        self.language.setCurrentIndex(0 if current_language() == "en" else 1)
        form.addRow(tr("language.label"), self.language)
        layout.addWidget(form_box)

        options = QGroupBox(tr("group.options"))
        options_layout = QVBoxLayout(options)
        self.upscale = QCheckBox(tr("upscale.label"))
        self.upscale.setChecked(False)
        self.anime = QCheckBox(tr("anime.label"))
        self.anime.setEnabled(False)
        self.upscale.toggled.connect(self.anime.setEnabled)
        self.colour = QCheckBox(tr("colour.label"))
        self.colour.setChecked(True)
        options_layout.addWidget(self.upscale)
        options_layout.addWidget(self.anime)
        options_layout.addWidget(self.colour)
        sharpen_row = QHBoxLayout()
        sharpen_row.addWidget(QLabel(tr("sharpen.label")))
        self.sharpen = QSlider(Qt.Orientation.Horizontal)
        self.sharpen.setRange(0, 100)
        self.sharpen.setValue(35)
        sharpen_row.addWidget(self.sharpen)
        options_layout.addLayout(sharpen_row)
        hint = QLabel(tr("upscale.hint"))
        hint.setObjectName("faint")
        hint.setWordWrap(True)
        options_layout.addWidget(hint)
        layout.addWidget(options)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status = QLabel(tr("status.ready"))
        self.status.setObjectName("subtitle")
        layout.addWidget(self.status)

        project_row = QHBoxLayout()
        self.open_project_button = QPushButton(tr("project.open"))
        self.open_project_button.clicked.connect(self._browse_project)
        self.save_project_button = QPushButton(tr("project.save"))
        self.save_project_button.setIcon(suite_icon("save"))
        self.save_project_button.setIconSize(QSize(22, 22))
        self.save_project_button.clicked.connect(self._save)
        self.save_as_button = QPushButton(tr("project.save_as"))
        self.save_as_button.setIcon(suite_icon("save-as"))
        self.save_as_button.setIconSize(QSize(22, 22))
        self.save_as_button.clicked.connect(self._browse_save)
        self.save_copy_button = QPushButton(tr("project.save_copy"))
        self.save_copy_button.setIcon(suite_icon("save-copy"))
        self.save_copy_button.setIconSize(QSize(22, 22))
        self.save_copy_button.clicked.connect(self._browse_copy)
        self.download_button = QPushButton(tr("project.download"))
        self.download_button.setIcon(suite_icon("download"))
        self.download_button.setIconSize(QSize(22, 22))
        self.download_button.clicked.connect(self._download_models)
        self.recent = QComboBox()
        self.recent.setObjectName("recent")
        self.recent.activated.connect(self._recent_chosen)
        project_row.addWidget(self.open_project_button)
        project_row.addWidget(self.save_project_button)
        project_row.addWidget(self.save_as_button)
        project_row.addWidget(self.save_copy_button)
        project_row.addWidget(self.download_button)
        project_row.addWidget(self.recent, 1)
        layout.addLayout(project_row)
        self._reload_recent()

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.start = QPushButton(tr("start"))
        self.start.setObjectName("start")
        self.start.clicked.connect(self._start)
        button_row.addWidget(self.start)
        layout.addLayout(button_row)

        footer = QLabel(version_label())
        footer.setObjectName("faint")
        layout.addWidget(footer)
        self._inputs = (
            self.source,
            self.output,
            self.browse,
            self.mode,
            self.container,
            self.codec,
            self.language,
            self.upscale,
            self.colour,
            self.sharpen,
            self.open_project_button,
            self.save_project_button,
            self.save_as_button,
            self.save_copy_button,
            self.download_button,
            self.export_button,
            self.recent,
        )

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)  # type: ignore[arg-type]
        enable_acrylic(int(self.winId()))

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        self._closing = True
        token = self._token
        if token is not None:
            token.cancel()
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.wait(3000)
        super().closeEvent(event)  # type: ignore[arg-type]

    def _browse_source(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            tr("source.browse"),
            "",
            tr("source.filter"),
        )
        if path:
            self.source.setText(path)
            if not self.output.text().strip():
                self.output.setText(str(Path(path).with_suffix(self._suffix())))

    def _suffix(self) -> str:
        container = self._container()
        if container is Container.MKV:
            return ".clarity.mkv"
        if container is Container.MP4:
            return ".clarity.mp4"
        unknown: Never = container
        return unknown

    def _container(self) -> Container:
        value = str(self.container.currentData())
        if value == Container.MKV.value:
            return Container.MKV
        if value == Container.MP4.value:
            return Container.MP4
        raise ClarityError(f"unknown container {value}")

    def _codec(self) -> VideoCodec:
        value = str(self.codec.currentData())
        if value == VideoCodec.AV1.value:
            return VideoCodec.AV1
        if value == VideoCodec.HEVC.value:
            return VideoCodec.HEVC
        raise ClarityError(f"unknown codec {value}")

    def _settings(self) -> Settings:
        mode = str(self.mode.currentData())
        if mode not in {"clean", "strong"}:
            raise ClarityError(f"unknown mode {mode}")
        colour = self.colour.isChecked()
        return Settings(
            mode=mode,
            upscale=self.upscale.isChecked(),
            anime=self.anime.isChecked() and self.upscale.isChecked(),
            sharpen=self.sharpen.value() / 100,
            white_balance=colour,
            levels=colour,
            container=self._container(),
            codec=self._codec(),
            copy_audio=True,
        )

    def _destination(self, source: Path) -> Path:
        typed = self.output.text().strip()
        if typed:
            return Path(typed)
        return source.with_suffix(self._suffix())

    def _set_busy(self, busy: bool) -> None:
        for widget in self._inputs:
            widget.setEnabled(not busy)
        if not busy:
            self.anime.setEnabled(self.upscale.isChecked())
        else:
            self.anime.setEnabled(False)
        self.start.setText(tr("cancel") if busy else tr("start"))
        self.start.setEnabled(True)

    def _start(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            token = self._token
            if token is not None:
                token.cancel()
            self.start.setEnabled(False)
            self.status.setText(tr("status.cancelling"))
            return
        source_text = self.source.text().strip()
        if not source_text:
            QMessageBox.warning(self, APP_NAME, tr("error.no_source"))
            return
        source = Path(source_text)
        if not source.is_file():
            QMessageBox.warning(self, APP_NAME, tr("error.missing_source"))
            return
        destination = self._destination(source)
        self.output.setText(str(destination))
        spec = JobSpec(
            source=source,
            destination=destination,
            settings=self._settings(),
            work=self._work_for(destination),
        )
        token = CancelToken()
        job = _JobThread(spec, token, self._runner, on_chunk=self._on_chunk)
        job.signals.progressed.connect(self._on_progress)
        job.signals.finished.connect(self._on_finished)
        self._thread = job
        self._token = token
        self._set_busy(True)
        self.status.setText(tr("status.running").format(done=0, total="…"))
        self.progress.setValue(0)
        job.start()

    def _on_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(max(0, min(100, int(round(fraction * 100)))))
        done, separator, total = text.partition("/")
        if separator:
            self.status.setText(tr("status.running").format(done=done, total=total))
        else:
            self.status.setText(text)

    def _on_finished(self, ok: bool, message: str) -> None:
        self._thread = None
        self._token = None
        self._set_busy(False)
        if self._closing:
            return
        if ok:
            self.progress.setValue(100)
            self._sync_project(self._work_for(Path(self.output.text().strip() or "out.mkv")))
            self.status.setText(tr("status.done"))
            return
        if message == "cancelled":
            self._sync_project(self._work_for(Path(self.output.text().strip() or "out.mkv")))
            self.status.setText(tr("status.cancelled"))
            return
        self.status.setText(message)
        QMessageBox.warning(self, APP_NAME, message)

    def _work_for(self, destination: Path) -> Path:
        if self._project_path is not None:
            resume = self._project.resume if self._project is not None else {}
            return project_work_dir(self._project_path, resume)
        return destination.with_name(destination.name + ".work")

    def _on_chunk(self, work: Path) -> None:
        self._sync_project(work)

    def _sync_project(self, work: Path) -> None:
        path = self._project_path
        if path is None or not path.is_file() or not work.is_dir():
            return
        try:
            self._project = sync_resume(path, work)
        except (OSError, ProjectFormatError, ClarityError):
            return

    def _reload_recent(self) -> None:
        self.recent.blockSignals(True)
        self.recent.clear()
        self.recent.addItem(tr("project.recent"), "")
        for item in load_recent():
            self.recent.addItem(Path(item).name, item)
        self.recent.blockSignals(False)

    def _remember(self, path: Path) -> None:
        remember_recent(path)
        self._reload_recent()

    def _browse_project(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(self, tr("project.open"), "", tr("project.filter"))
        if path:
            self.open_project(Path(path))

    def _save(self) -> None:
        if self._project_path is not None:
            self.save_project(self._project_path)
            return
        self._browse_save()

    def _browse_save(self) -> None:
        suggested = str(self._project_path or "")
        path, _selected = QFileDialog.getSaveFileName(self, tr("project.save_as"), suggested, tr("project.filter"))
        if path:
            self.save_project(Path(path))

    def _browse_copy(self) -> None:
        suggested = str(self._project_path or "")
        path, _selected = QFileDialog.getSaveFileName(self, tr("project.save_copy"), suggested, tr("project.filter"))
        if path:
            self.save_project(Path(path), retarget=False)

    def _browse_export(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            tr("project.export"),
            self.output.text().strip(),
            tr("source.filter"),
        )
        if path:
            self.output.setText(path)

    def _download_models(self) -> None:
        try:
            ids = [model.id for model in models_for(self._settings())]
            fetch_weights(ids, dry_run=False)
        except (EngineUnavailable, ClarityError, KeyError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self.status.setText(tr("status.downloaded"))

    def _recent_chosen(self, index: int) -> None:
        value = self.recent.itemData(index)
        if isinstance(value, str) and value:
            self.open_project(Path(value))

    def open_project(self, path: Path) -> None:
        try:
            project = load_project(path)
        except ProjectFormatError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._project = project
        self._project_path = project.path
        self._apply_project(project)
        self._remember(project.path)
        if project.resume.get("journal"):
            self.status.setText(tr("status.resumed"))

    def save_project(self, path: Path, *, retarget: bool = True) -> None:
        source_text = self.source.text().strip()
        if not source_text or not Path(source_text).is_file():
            QMessageBox.warning(self, APP_NAME, tr("error.missing_source"))
            return
        source = Path(source_text)
        destination = self._destination(source)
        self.output.setText(str(destination))
        try:
            project = save_project(
                path,
                source=source,
                settings=self._settings(),
                destination=destination,
                resume=self._project.resume if self._project is not None else {},
                previous=self._project,
            )
        except ProjectFormatError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if retarget:
            self._project = project
            self._project_path = project.path
        self._remember(project.path)
        self.status.setText(tr("status.saved"))

    def _apply_project(self, project: Project) -> None:
        if project.source is not None:
            self.source.setText(str(project.source))
        else:
            video = video_fingerprint(project.manifest)
            self.source.setText(str(video.get("absolute") or ""))
        if project.destination is not None:
            self.output.setText(str(project.destination))
        settings = project.settings
        self._select(self.mode, settings.mode)
        self._select(self.container, settings.container.value)
        self._select(self.codec, settings.codec.value)
        self.upscale.setChecked(settings.upscale)
        self.anime.setChecked(settings.anime)
        self.colour.setChecked(settings.white_balance or settings.levels)
        self.sharpen.setValue(int(round(settings.sharpen * 100)))

    def _select(self, box: QComboBox, value: str) -> None:
        for index in range(box.count()):
            if box.itemData(index) == value:
                box.setCurrentIndex(index)
                return
