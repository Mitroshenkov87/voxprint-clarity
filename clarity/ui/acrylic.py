"""Windows 11 Acrylic backdrop. Other platforms keep the plain dark fill."""
from __future__ import annotations

import ctypes
import sys


def enable_acrylic(hwnd: int) -> bool:
    """Ask DWM for an Acrylic system backdrop. False when the call is unavailable."""
    if sys.platform != "win32" or hwnd == 0:
        return False
    backdrop = ctypes.c_int(3)  # DWMSBT_TRANSIENTWINDOW, Acrylic
    dark = ctypes.c_int(1)
    try:
        dwm = ctypes.windll.dwmapi
        hr_back = dwm.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
        dwm.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), 20, ctypes.byref(dark), ctypes.sizeof(dark))
    except (AttributeError, OSError):
        return False
    return hr_back == 0
