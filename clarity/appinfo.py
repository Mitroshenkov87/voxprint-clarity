"""Version, build number, and the release codename.

``BUILD.json`` is the single source of truth for ``version``, ``build``, and
``codename``. A codename is one Hebrew word from the Torah or Tanakh, written
in ASCII transliteration. It describes the state of that build.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "Voxprint AI Clarity"
USER_KEY = "clarity"

_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Checkout or install root. ``BUILD.json`` and ``assets`` live here."""
    return _ROOT


def _load_build() -> dict[str, object]:
    path = _ROOT / "BUILD.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"BUILD.json could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("BUILD.json must be an object")
    return data


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"BUILD.json {key} must be a non-empty string")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"BUILD.json {key} must be an integer")
    return value


_BUILD = _load_build()
APP_VERSION = _require_str(_BUILD, "version")
APP_BUILD = _require_int(_BUILD, "build")
CODENAME = _require_str(_BUILD, "codename")


def app_build() -> int:
    """GitHub run number when CI sets it, otherwise the build in ``BUILD.json``."""
    env = os.environ.get("VOXPRINT_BUILD_NUMBER", "").strip()
    if env.isdigit():
        return int(env)
    if APP_BUILD > 0:
        return APP_BUILD
    offset = _BUILD.get("offset", 0)
    return int(offset) if isinstance(offset, int) and not isinstance(offset, bool) else 0


def version_label() -> str:
    """Human version, for example ``1.0.0 RC · build 999 "Codename"``."""
    pretty = APP_VERSION[:-3] + " RC" if APP_VERSION.endswith("-rc") else APP_VERSION
    label = f"{pretty} · build {app_build()}"
    name = CODENAME.strip()
    if name:
        label += f' "{name}"'
    return label


def release_title() -> str:
    """GitHub pre-release title."""
    return f"{APP_NAME} {version_label()}"
