"""Progress dialog for a suite repair. A healthy check does not open it."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget

from clarity.appinfo import APP_NAME

_REPAIR = ("Installing", "Repairing", "Removing")


class RepairProgress:
    """Show a dialog only while the engine or ffmpeg is installing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Remember the parent widget. The dialog is created on the first install line.

        Args:
            parent: Optional owner for the dialog.
        """
        self._parent = parent
        self.dialog: QProgressDialog | None = None

    def __call__(self, message: str) -> None:
        """Update the dialog when ``message`` is an install or removal step.

        Args:
            message: A status line from ``ensure_runtime`` or the ffmpeg repair.
                ``Installing ...`` opens the dialog. A check or ready line does not.
        """
        repairing = message.startswith(_REPAIR)
        if not repairing and self.dialog is None:
            return
        if self.dialog is None:
            box = QProgressDialog(message, "", 0, 0, self._parent)
            box.setWindowTitle(APP_NAME)
            box.setCancelButton(None)
            box.setMinimumDuration(0)
            box.setAutoClose(True)
            box.show()
            self.dialog = box
        self.dialog.setLabelText(message)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
