"""Version, build number, and the Torah codename for this build."""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "Voxprint AI Clarity"
APP_VERSION = "0.1.0"
APP_BUILD = 0
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
    return f"{APP_VERSION} · build {app_build()} \"{CODENAME}\""
