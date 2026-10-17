"""Fit the shared video screen to Clarity at 1280 by 800.

The shell owns the widgets. This module only resizes, relabels, and groups
them after each refresh: a fixed expert column, a tall before/after wipe,
file names on the drop area and the history tiles, and the graphics-card check.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QProcess, Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QBoxLayout,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from voxprint_common.i18n import tr
from voxprint_common.ui.simple_image import SimpleLayout, SimpleWidget

PANEL_WIDTH = 340
_COMPARE_MIN = 200
_DROP_KEY = "simple.drop.video"


def gpu_selftest_command() -> list[str]:
    """Return the command that runs the GPU selftest.

    The command is ``python -m clarity --selftest``. It does not pass
    ``--no-gpu``.

    Returns:
        The executable followed by the module arguments.
    """
    return [sys.executable, "-m", "clarity", "--selftest"]


def gpu_selftest_passed(returncode: int, output: str) -> bool:
    """Return whether the selftest process passed.

    Args:
        returncode: The process exit code.
        output: Merged stdout and stderr.

    Returns:
        True when the process exited 0 and reported ``selftest ok``.
    """
    return returncode == 0 and "selftest ok" in output


def install(screen: SimpleWidget, layout: SimpleLayout) -> None:
    """Wrap refresh and expert mode so Clarity's fit runs after the shell.

    Args:
        screen: The shared video screen.
        layout: The layout whose prompt and history the fit reads.
    """
    original_refresh = screen.refresh
    original_expert = screen.set_expert
    original_prompt = screen.set_prompt_text

    def refresh(*, prompt: bool = False) -> None:
        original_refresh(prompt=prompt)
        apply(screen, layout)

    def set_expert(enabled: bool) -> None:
        original_expert(enabled)
        apply(screen, layout)

    def set_prompt_text(text: str) -> None:
        original_prompt(text)
        _dress_drop(screen.root, layout.prompt)

    screen.refresh = refresh  # type: ignore[method-assign]
    screen.set_expert = set_expert  # type: ignore[method-assign]
    screen.set_prompt_text = set_prompt_text  # type: ignore[method-assign]
    watcher = _PanelFit(screen, layout)
    splitter = _splitter(screen)
    if splitter is not None:
        splitter.installEventFilter(watcher)
    screen.root.setProperty("clarityFit", watcher)
    apply(screen, layout)


def apply(screen: SimpleWidget, layout: SimpleLayout) -> None:
    """Relabel and resize the screen from the current layout.

    Args:
        screen: The shared video screen.
        layout: Live prompt, history, and expert values.
    """
    root = screen.root
    _compact_presets(screen)
    _dock_expert_column(screen)
    _dress_compare(root)
    _dress_drop(root, layout.prompt)
    _dress_history(root)
    _widen_expert_rows(root)
    _dress_models(root, layout)
    _dress_toggle(root)
    fit_expert_panel(screen)


def fit_expert_panel(screen: SimpleWidget) -> None:
    """Give the expert column a fixed width and let its list fill the height.

    Args:
        screen: The shared video screen. Hidden panels are left alone.
    """
    panel = _panel(screen)
    splitter = _splitter(screen)
    if panel is None or splitter is None or not panel.isVisible():
        return
    panel.setMinimumWidth(PANEL_WIDTH)
    panel.setMaximumWidth(PANEL_WIDTH)
    panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    scroll = screen.root.findChild(QScrollArea, "expertScroll")
    if scroll is not None:
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setMinimumHeight(0)
    sizes = splitter.sizes()
    if sizes and abs(sizes[-1] - PANEL_WIDTH) <= 2 and panel.minimumWidth() == PANEL_WIDTH:
        return
    total = max(splitter.width(), PANEL_WIDTH + 8)
    handle = splitter.handleWidth()
    splitter.setSizes([max(1, total - handle - PANEL_WIDTH), PANEL_WIDTH])


def run_graphics_check(parent: QWidget) -> None:
    """Run the GPU selftest and show a plain pass or fail.

    The process starts when the dialog opens. Closing the dialog stops it.
    The headline is a sentence. The process output stays behind Details.

    Args:
        parent: The window that owns the dialog.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle(tr("clarity.action.check_gpu"))
    dialog.resize(640, 420)
    column = QVBoxLayout(dialog)
    column.setContentsMargins(16, 16, 16, 16)
    column.setSpacing(12)
    result = QLabel(tr("clarity.gpu.running"))
    result.setObjectName("gpuResult")
    result.setWordWrap(True)
    column.addWidget(result)
    details = QToolButton()
    details.setObjectName("gpuDetails")
    details.setText(tr("clarity.gpu.details"))
    details.setCheckable(True)
    details.setChecked(False)
    details.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    details.setArrowType(Qt.ArrowType.RightArrow)
    column.addWidget(details, alignment=Qt.AlignmentFlag.AlignLeft)
    output = QPlainTextEdit()
    output.setObjectName("gpuOutput")
    output.setReadOnly(True)
    output.hide()
    column.addWidget(output, 1)

    def toggle(checked: bool) -> None:
        details.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        output.setVisible(checked)

    details.toggled.connect(toggle)
    process = QProcess(dialog)
    process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    chunks: list[str] = []

    def append() -> None:
        text = bytes(process.readAllStandardOutput().data()).decode("utf-8", errors="replace")
        if not text:
            return
        chunks.append(text)
        output.setPlainText("".join(chunks))

    def finish(code: int, _status: QProcess.ExitStatus) -> None:
        append()
        passed = gpu_selftest_passed(code, "".join(chunks))
        result.setText(tr("clarity.gpu.pass") if passed else tr("clarity.gpu.fail"))

    def stop() -> None:
        if process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    process.readyReadStandardOutput.connect(append)
    process.finished.connect(finish)
    dialog.finished.connect(stop)
    command = gpu_selftest_command()
    process.start(command[0], command[1:])
    dialog.exec()


class _PanelFit(QObject):
    """Reapply the expert width after the shell's own resize filter."""

    def __init__(self, screen: SimpleWidget, layout: SimpleLayout) -> None:
        """Watch the screen.

        Args:
            screen: The shared video screen.
            layout: Live values used to fit the model names.
        """
        super().__init__(screen.root)
        self._screen = screen
        self._layout = layout

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Schedule a width fit on resize.

        Args:
            watched: The splitter.
            event: The Qt event.

        Returns:
            False so the shell still sees the event.
        """
        del watched
        if event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._refit)
        return False

    def _refit(self) -> None:
        fit_expert_panel(self._screen)
        root = self._screen.root
        _widen_expert_rows(root)
        _dress_models(root, self._layout)


def _compact_presets(screen: SimpleWidget) -> None:
    host = getattr(screen, "_preset_host", None)
    rows = getattr(screen, "_preset_rows", None)
    if host is None or rows is None or host.property("clarityColumns") is True:
        return
    items: list[QLayoutItem] = []
    while rows.count():
        item = rows.takeAt(0)
        if item is not None:
            items.append(item)
    if len(items) < 2:
        for item in items:
            _restore_item(rows, item)
        return
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(4)
    paired = [item for item in items if _row_id(item.layout()) != "size"]
    spanned = [item for item in items if _row_id(item.layout()) == "size"]
    for item in items:
        layout = item.layout()
        if layout is not None:
            layout.setSpacing(6)
            _narrow_labels(layout)
    grid_row = 0
    for index in range(0, len(paired), 2):
        grid.addItem(paired[index], grid_row, 0)
        if index + 1 < len(paired):
            grid.addItem(paired[index + 1], grid_row, 1)
        grid_row += 1
        if grid_row == 1:
            for item in spanned:
                grid.addItem(item, grid_row, 0, 1, 2)
                grid_row += 1
    rows.setSpacing(0)
    rows.addLayout(grid)
    host.setProperty("clarityColumns", True)
    host.setProperty("clarityPresetItems", items)


def _restore_item(rows: QVBoxLayout, item: QLayoutItem) -> None:
    layout = item.layout()
    widget = item.widget()
    if layout is not None:
        rows.addLayout(layout)
        return
    if widget is not None:
        rows.addWidget(widget)


def _narrow_labels(layout: QLayout) -> None:
    for index in range(layout.count()):
        child = layout.itemAt(index)
        if child is None:
            continue
        widget = child.widget()
        if isinstance(widget, QLabel) and widget.property("presetLabel") == "true":
            widget.setMinimumWidth(72)


def _row_id(layout: QLayout | None) -> str:
    if layout is None:
        return ""
    for index in range(layout.count()):
        child = layout.itemAt(index)
        if child is None:
            continue
        widget = child.widget()
        if isinstance(widget, QLabel) and widget.objectName().startswith("preset-"):
            return widget.objectName().removeprefix("preset-")
    return ""


def _dock_expert_column(screen: SimpleWidget) -> None:
    """Put the drop, the options, and the wipe in one column beside Expert.

    The shell places Expert beside the stage only, so the list is as short as
    the picture. Moving the rows above the picture into that column gives the
    list the height of the main column and leaves the wipe the stretch.
    """
    root = screen.root
    if root.property("clarityColumn") is True:
        return
    layout = root.layout()
    splitter = _splitter(screen)
    stage = root.findChild(QFrame, "stage")
    host = getattr(screen, "_preset_host", None)
    drop = root.findChild(QLabel, "fileDrop")
    generate = root.findChild(QPushButton, "generate")
    if layout is None or splitter is None or stage is None or host is None or drop is None or generate is None:
        return
    prompt_index = None
    host_index = None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        if item.widget() is host:
            host_index = index
        if _contains_widget(item.layout(), drop):
            prompt_index = index
    if prompt_index is None or host_index is None:
        return
    held: list[QLayoutItem] = []
    for index in sorted((prompt_index, host_index), reverse=True):
        item = layout.takeAt(index)
        if item is not None:
            held.append(item)
    column = QWidget()
    column.setObjectName("mainColumn")
    column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    column_layout = QVBoxLayout(column)
    column_layout.setContentsMargins(0, 0, 0, 0)
    column_layout.setSpacing(6)
    row = QWidget(column)
    row.setObjectName("sourceRow")
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(12)
    drop.setParent(row)
    generate.setParent(row)
    row_layout.addWidget(drop, 1)
    row_layout.addWidget(generate)
    column_layout.addWidget(row)
    host.setParent(column)
    column_layout.addWidget(host)
    stage.setParent(column)
    column_layout.addWidget(stage, 1)
    splitter.insertWidget(0, column)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    layout.setSpacing(6)
    root.setProperty("clarityColumn", True)
    screen._clarity_held_items = held  # type: ignore[attr-defined]


def _contains_widget(layout: QLayout | None, widget: QWidget) -> bool:
    if layout is None:
        return False
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        if item.widget() is widget:
            return True
        if _contains_widget(item.layout(), widget):
            return True
    return False


def _dress_compare(root: QWidget) -> None:
    frame = root.findChild(QLabel, "compareView")
    if frame is not None:
        frame.setMinimumHeight(_COMPARE_MIN)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        page = frame.parentWidget()
        if page is not None:
            page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = None if page is None else page.layout()
        if isinstance(layout, QGridLayout):
            layout.setRowStretch(0, 1)
            layout.setColumnStretch(0, 1)
    drop = root.findChild(QLabel, "fileDrop")
    if drop is not None:
        drop.setMinimumHeight(64)
        drop.setMaximumHeight(72)
    strip = root.findChild(QScrollArea, "historyStrip")
    if strip is not None and not strip.isHidden():
        strip.setFixedHeight(96)
    stage = root.findChild(QFrame, "stage")
    if stage is not None:
        stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    stack = root.findChild(QStackedWidget, "stageStack")
    if stack is not None:
        stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


def _dress_drop(root: QWidget, prompt: str) -> None:
    drop = root.findChild(QLabel, "fileDrop")
    if drop is None:
        return
    hint = tr(_DROP_KEY)
    cleaned = prompt.strip()
    name = Path(cleaned).name if cleaned else ""
    drop.setText(f"{hint}\n{name}" if name else hint)
    drop.setToolTip(cleaned)


def _dress_history(root: QWidget) -> None:
    for button in root.findChildren(QPushButton):
        if not button.objectName().startswith("thumb-"):
            continue
        path_text = button.toolTip().split("\n", 1)[0].strip()
        path = Path(path_text) if path_text else None
        name = path.name if path is not None and path.name else path_text
        if not name:
            continue
        when = ""
        if path is not None and path.is_file():
            stamp = datetime.fromtimestamp(path.stat().st_mtime)
            when = stamp.strftime("%Y-%m-%d")
        shown = QFontMetrics(button.font()).elidedText(name, Qt.TextElideMode.ElideMiddle, 76)
        button.setText(f"{shown}\n{when}" if when else shown)
        button.setToolTip(f"{path_text}\n{when}" if when else path_text)


def _widen_expert_rows(root: QWidget) -> None:
    panel = root.findChild(QFrame, "expertPanel")
    if panel is None or not panel.isVisible():
        return
    form = panel.findChild(QWidget, "expertForm")
    form_layout = None if form is None else form.layout()
    if form_layout is not None:
        _widen_layout(form_layout)


def _widen_layout(layout: QLayout) -> None:
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        nested = item.layout()
        if not isinstance(nested, QBoxLayout):
            if nested is not None:
                _widen_layout(nested)
            continue
        first = nested.itemAt(0)
        second = nested.itemAt(1)
        label = None if first is None else first.widget()
        control = None if second is None else second.widget()
        if isinstance(label, QLabel) and control is not None:
            label.setMinimumWidth(0)
            label.setMaximumWidth(max(label.sizeHint().width(), 64))
            nested.setStretch(0, 0)
            nested.setStretch(1, 1)
            continue
        _widen_layout(nested)


def _dress_models(root: QWidget, layout: SimpleLayout) -> None:
    panel = root.findChild(QFrame, "expertPanel")
    if panel is None or not panel.isVisible():
        return
    for edit in panel.findChildren(QPlainTextEdit):
        key = edit.objectName()
        if not key or layout.state is None or not layout.state.has(key):
            continue
        value = layout.get(key)
        if not isinstance(value, str):
            continue
        caption = _model_caption(value)
        help_tip = edit.property("helpTip")
        help_text = help_tip if isinstance(help_tip, str) else ""
        width = edit.viewport().width()
        if width < 24:
            width = max(edit.width() - 16, 160)
        shown = QFontMetrics(edit.font()).elidedText(caption, Qt.TextElideMode.ElideRight, width - 8)
        edit.blockSignals(True)
        edit.setReadOnly(True)
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit.setPlainText(shown or caption)
        edit.document().setDocumentMargin(0)
        edit.setFixedHeight(36)
        edit.blockSignals(False)
        if caption and help_text:
            edit.setToolTip(f"{caption}\n{help_text}")
        else:
            edit.setToolTip(caption or help_text)


def _model_caption(raw: str) -> str:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return raw
    names: list[str] = []
    for part in parts:
        key = f"model.{part}"
        label = tr(key)
        names.append(part if label == key else label)
    return ", ".join(names)


def _dress_toggle(root: QWidget) -> None:
    button = root.findChild(QPushButton, "expertToggle")
    if button is None:
        return
    font = QFont(button.font())
    font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(button.text())
    button.setMinimumWidth(max(advance + 64, 36))
    button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    button.updateGeometry()


def _panel(screen: SimpleWidget) -> QFrame | None:
    found = screen.root.findChild(QFrame, "expertPanel")
    return found if isinstance(found, QFrame) else None


def _splitter(screen: SimpleWidget) -> QSplitter | None:
    found = screen.root.findChild(QSplitter, "expertSplit")
    return found if isinstance(found, QSplitter) else None
