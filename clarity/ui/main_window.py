"""The Clarity window. Strings come from the English and Russian catalogues."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
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
from clarity.i18n import current_language, languages, tr
from clarity.ui.acrylic import enable_acrylic
from clarity.ui.theme import build_style


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
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
        browse = QPushButton(tr("source.browse"))
        browse.clicked.connect(self._browse_source)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        source_row.addWidget(browse)
        form.addRow(tr("source.label"), source_row)

        self.output = QLineEdit()
        self.output.setPlaceholderText(tr("output.placeholder"))
        form.addRow(tr("output.label"), self.output)

        self.mode = QComboBox()
        self.mode.addItem(tr("mode.clean"), "clean")
        self.mode.addItem(tr("mode.strong"), "strong")
        form.addRow(tr("mode.label"), self.mode)

        self.container = QComboBox()
        self.container.addItem("MKV", "mkv")
        self.container.addItem("MP4", "mp4")
        form.addRow(tr("container.label"), self.container)

        self.codec = QComboBox()
        self.codec.addItem(tr("codec.av1"), "av1")
        self.codec.addItem(tr("codec.hevc"), "hevc")
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

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)  # type: ignore[arg-type]
        enable_acrylic(int(self.winId()))

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
                self.output.setText(str(Path(path).with_suffix(".clarity.mkv")))

    def _start(self) -> None:
        if not self.source.text().strip():
            QMessageBox.warning(self, APP_NAME, tr("error.no_source"))
            return
        self.status.setText(tr("status.need_gpu_job"))
        self.progress.setValue(0)
