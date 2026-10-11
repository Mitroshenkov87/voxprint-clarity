"""Progress dialog for a suite repair. A healthy check does not open it."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget

from clarity.appinfo import APP_NAME


class RepairProgress:
    """Show a dialog only when a component is being repaired."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Remember the parent widget. The dialog is created on the first repair.

        Args:
            parent: Optional owner for the dialog.
        """
        self._parent = parent
        self.dialog: QProgressDialog | None = None

    def __call__(self, fraction: float, message: str) -> None:
        """Update the dialog when ``message`` is a repair step.

        Args:
            fraction: Completion from 0 to 1.
            message: ``"repairing <component>"`` opens the dialog. A ready line does not.
        """
        if not message.startswith("repairing") and self.dialog is None:
            return
        if self.dialog is None:
            box = QProgressDialog(message, "", 0, 100, self._parent)
            box.setWindowTitle(APP_NAME)
            box.setCancelButton(None)
            box.setMinimumDuration(0)
            box.setAutoClose(True)
            box.show()
            self.dialog = box
        self.dialog.setLabelText(message)
        self.dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
