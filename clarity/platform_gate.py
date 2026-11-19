"""Windows 11 24H2+ and Linux releases from 2025. Older systems do not start."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

WIN_MIN_BUILD = 26100


@dataclass(frozen=True)
class PlatformGate:
    """Whether this operating system may launch Clarity."""

    ok: bool
    message: str


def windows_supported(build: int) -> bool:
    """Return whether a Windows build is 11 24H2 or newer.

    Args:
        build: ``sys.getwindowsversion().build``.

    Returns:
        True when ``build`` is at least 26100.
    """
    return int(build) >= WIN_MIN_BUILD


def _parse_os_release(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _ubuntu_style(version: str) -> bool | None:
    """``YY.MM`` from 25.04 onward. None when ``version`` is not that shape."""
    major, dot, minor = version.partition(".")
    if not dot or not major.isdigit() or not minor.isdigit():
        return None
    year = int(major)
    month = int(minor)
    if year >= 1000:
        return year >= 2025
    if year < 20:
        return None
    return (year, month) >= (25, 4)


def linux_supported(os_release: str) -> bool:
    """True for a 2025-or-later distro release, or rolling Arch."""
    data = _parse_os_release(os_release)
    ident = data.get("ID", "").lower()
    version = data.get("VERSION_ID", "")
    if ident == "arch":
        return True
    styled = _ubuntu_style(version)
    if ident in {"ubuntu", "pop", "linuxmint", "zorin", "elementary", "neon"} and styled is not None:
        return styled
    if ident == "debian" and version.isdigit():
        return int(version) >= 13
    if ident == "fedora" and version.isdigit():
        return int(version) >= 42
    if version.isdigit() and int(version) >= 2025:
        return True
    if styled is True:
        return True
    return False


def assess_platform(
    *,
    system: str | None = None,
    win_build: int | None = None,
    os_release: str | None = None,
) -> PlatformGate:
    """Refuse an older OS. The message is English; the UI translates the key."""
    host = system if system is not None else sys.platform
    if host == "win32":
        build = win_build
        if build is None:
            reader = getattr(sys, "getwindowsversion", None)
            if not callable(reader):
                return PlatformGate(
                    False,
                    "Clarity needs Windows 11 24H2 or newer, and this Python cannot read the Windows build.",
                )
            build = int(getattr(reader(), "build"))
        if windows_supported(build):
            return PlatformGate(True, "")
        return PlatformGate(
            False,
            f"Clarity needs Windows 11 24H2 or newer (build {WIN_MIN_BUILD}+). This PC is build {build}.",
        )
    if host == "linux":
        text = os_release
        if text is None:
            try:
                text = Path("/etc/os-release").read_text(encoding="utf-8")
            except OSError:
                text = ""
        if linux_supported(text):
            return PlatformGate(True, "")
        return PlatformGate(
            False,
            "Clarity needs a Linux release from 2025 or newer. This system is older or was not recognised.",
        )
    return PlatformGate(False, f"Clarity runs on Windows 11 24H2+ and Linux 2025+. This system is {host}.")
