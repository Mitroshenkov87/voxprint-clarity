"""Suite icons. The files live in ``assets/icons/suite`` and are not floppy disks."""
from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from clarity.appinfo import repo_root
from clarity.errors import ClarityError
from clarity.ui.theme import TEXT

_NAMES = ("save", "save-as", "save-copy", "export", "download")


def suite_icon(name: str) -> QIcon:
    """Render one suite SVG. ``currentColor`` becomes the theme text colour."""
    if name not in _NAMES:
        raise KeyError(name)
    path = repo_root() / "assets" / "icons" / "suite" / f"{name}.svg"
    raw = path.read_text(encoding="utf-8").replace("currentColor", TEXT)
    renderer = QSvgRenderer(QByteArray(raw.encode("utf-8")))
    if not renderer.isValid():
        raise ClarityError(f"could not read the suite icon {name}")
    pixmap = QPixmap(22, 22)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    return QIcon(pixmap)
