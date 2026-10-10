"""Version, build number, and the Torah codename for this build.

A codename is one Biblical Hebrew word in ASCII transliteration. It describes
the state of that build. Build 999 is ``Bereshit`` (Genesis 1:1, "in the
beginning"): the first release candidate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "Voxprint AI Clarity"
APP_VERSION = "1.0.0-rc"
APP_BUILD = 999
CODENAME = "Bereshit"
USER_KEY = "clarity"

_ROOT = Path(__file__).resolve().parents[1]


def app_build() -> int:
    """GitHub run number when CI sets it, otherwise the offset in ``BUILD.json``."""
    env = os.environ.get("VOXPRINT_BUILD_NUMBER", "").strip()
    if env.isdigit():
        return int(env)
    if APP_BUILD > 0:
        return APP_BUILD
    try:
        data = json.loads((_ROOT / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    offset = data.get("offset", 0) if isinstance(data, dict) else 0
    return int(offset) if isinstance(offset, int) else 0


def version_label() -> str:
    """Human version, for example ``1.0.0 RC · build 999 "Bereshit"``."""
    pretty = APP_VERSION[:-3] + " RC" if APP_VERSION.endswith("-rc") else APP_VERSION
    label = f"{pretty} · build {app_build()}"
    name = CODENAME.strip()
    if name:
        label += f' "{name}"'
    return label


def release_title() -> str:
    """GitHub pre-release title."""
    return f"{APP_NAME} {version_label()}"
