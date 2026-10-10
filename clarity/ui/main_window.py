"""The Clarity window. Strings come from the English and Russian catalogues."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Never

from PySide6.QtCore import QObject, Qt, QThread, Signal
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
from clarity.errors import Cancelled, ClarityError
from clarity.i18n import current_language, languages, tr
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings
from clarity.pipeline.runner import CancelToken, JobSpec, run_job
from clarity.ui.acrylic import enable_acrylic
from clarity.ui.theme import build_style

Runner = Callable[..., None]


class _JobSignals(QObject):
    progressed = Signal(float, str)
    finished = Signal(bool, str)


class _JobThread(QThread):
    """Runs ``run_job`` off the UI thread. Cancel kills ffmpeg through the token."""

    def __init__(self, spec: JobSpec, token: CancelToken, runner: Runner) -> None:
        super().__init__()
        self._spec = spec
        self._token = token
        self._runner = runner
        self.signals = _JobSignals()

    def run(self) -> None:
        def report(fraction: float, text: str) -> None:
            self.signals.progressed.emit(fraction, text)

        try:
            self._runner(self._spec, token=self._token, progress=report)
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
        form.addRow(tr("output.label"), self.output)

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
        spec = JobSpec(source=source, destination=destination, settings=self._settings())
        token = CancelToken()
        job = _JobThread(spec, token, self._runner)
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
            self.status.setText(tr("status.done"))
            return
        if message == "cancelled":
            self.status.setText(tr("status.cancelled"))
            return
        self.status.setText(message)
        QMessageBox.warning(self, APP_NAME, message)
