"""Which Qt platform a real window uses.

Selftest sets ``QT_QPA_PLATFORM=offscreen``. A normal launch must not. The
RTX 4090 session redirected stdout, ``isatty()`` was false, and Clarity forced
the offscreen plugin. That plugin logs ``propagateSizeHints`` and never maps
a window on the interactive desktop. The missing-font line is the same plugin:
Qt's offscreen integration does not use the Windows font directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import PySide6
from PySide6.QtCore import QLibrary, QLibraryInfo

from clarity.errors import ClarityError

# os.add_dll_directory cookies must stay alive or Windows drops the search path.
_DLL_DIRS: list[object] = []


def gui_qt_platform() -> str | None:
    """The platform a real window should force.

    Returns:
        None. The native plugin (``windows`` on Windows) stays in charge.
        An empty string or ``offscreen`` must not be written into the environment.
    """
    return None


def selftest_qt_platform() -> str:
    """The platform the offscreen selftest uses.

    Returns:
        ``offscreen``.
    """
    return "offscreen"


def windows_platform_plugin() -> Path:
    """Path of ``qwindows.dll`` inside this PySide6 install.

    Windows wheels keep plugins in ``PySide6/plugins``. Linux wheels keep them
    in ``PySide6/Qt/plugins``. ``QLibraryInfo`` reports the directory this
    build uses.

    Returns:
        The platform plugin path. The file is absent on Linux and macOS wheels.
    """
    plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    located = plugins / "platforms" / "qwindows.dll"
    if located.is_file():
        return located
    root = Path(PySide6.__file__).resolve().parent
    for candidate in (
        root / "plugins" / "platforms" / "qwindows.dll",
        root / "Qt" / "plugins" / "platforms" / "qwindows.dll",
    ):
        if candidate.is_file():
            return candidate
    return located


def require_windows_platform_plugin() -> None:
    """Load ``qwindows.dll`` when this process is on Windows.

    The offscreen selftest still runs this check so a Windows install fails
    before a user opens the shortcut against a wheel that shipped no Windows
    plugin.

    Raises:
        ClarityError: When the DLL is missing or ``QLibrary`` cannot load it.
    """
    if sys.platform != "win32":
        return
    path = windows_platform_plugin()
    if not path.is_file():
        raise ClarityError(
            f"Qt's Windows platform plugin is missing ({path}). The window cannot open on this install."
        )
    if hasattr(os, "add_dll_directory"):
        root = Path(PySide6.__file__).resolve().parent
        for folder in (root, root / "Qt" / "bin", root / "Qt6" / "bin"):
            if folder.is_dir():
                _DLL_DIRS.append(os.add_dll_directory(str(folder)))
    library = QLibrary(str(path))
    if not library.load():
        detail = library.errorString() or "QLibrary returned no error string"
        raise ClarityError(f"Qt's Windows platform plugin did not load ({path}): {detail}")
