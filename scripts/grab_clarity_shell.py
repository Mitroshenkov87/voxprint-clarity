"""Save offscreen shots of simple mode and the Expert panel.

Run with ``QT_QPA_PLATFORM=offscreen``. The Expert shot includes the wipe.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from voxprint_common.i18n import set_language
from voxprint_common.ui.safety import ReviewedOutput

from clarity.ui.shell import MainWindow

_OUT = Path("/opt/cursor/artifacts")
_SIZE = (1280, 800)


def _shot(window: MainWindow, name: str) -> None:
    window.resize(*_SIZE)
    window.show()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    _OUT.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(_OUT / name))


def _show_wipe(window: MainWindow, sample: Path, item_id: str) -> None:
    window.use_source_text(str(sample))
    layout = window._layout
    screen = window._screen
    if screen is None or not layout.generate():
        raise RuntimeError("the sample did not start a restore")
    layout.progress(60, 100, "Restoring")
    layout.finish(item_id, ReviewedOutput(path=sample))
    screen.set_expert(True)
    screen.refresh()
    frame = window.findChild(QLabel, "compareView")
    if frame is not None and frame.height() < 40:
        frame.setMinimumHeight(200)
        screen.refresh()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def main() -> None:
    """Write English and Russian shots of the video tab."""
    home = Path(os.environ.get("VOXPRINT_HOME", "")).expanduser()
    if not str(home):
        raise RuntimeError("set VOXPRINT_HOME to an empty directory")
    sample = home / "sample.mkv"
    sample.write_bytes(b"frame-bytes")
    app = QApplication.instance() or QApplication([])
    set_language("en")
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window.use_source_text(str(sample))
    _shot(window, "clarity-simple-en.png")
    _show_wipe(window, sample, "shot-en")
    _shot(window, "clarity-expert-en.png")
    window.set_language_code("ru")
    app.processEvents()
    window.use_source_text(str(sample))
    _shot(window, "clarity-simple-ru.png")
    _show_wipe(window, sample, "shot-ru")
    _shot(window, "clarity-expert-ru.png")
    tabs = window._tabs
    setup = window._setup_page
    if tabs is None or setup is None:
        raise RuntimeError("the setup tab is missing")
    tabs.setCurrentWidget(setup)
    _shot(window, "clarity-setup-ru.png")
    window.close()


if __name__ == "__main__":
    sys.exit(main())
