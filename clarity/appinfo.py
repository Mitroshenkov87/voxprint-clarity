"""Version, build number, and the release codename.

``voxprint_common.buildinfo`` reads ``BUILD.json``. A codename is one Hebrew
word from the Torah or Tanakh, written in ASCII transliteration. It describes
the state of that build.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from voxprint_common.buildinfo import BuildInfo, label, load

APP_NAME = "Voxprint AI Clarity"
USER_KEY = "clarity"

_ROOT = Path(__file__).resolve().parents[1]
_BUILD_PATH = _ROOT / "BUILD.json"


def repo_root() -> Path:
    """Checkout or install root. ``BUILD.json`` and ``assets`` live here."""
    return _ROOT


def _build_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for ``load``, including Clarity's older build-number name.

    ``VOXPRINT_BUILD`` wins. When it is unset, ``VOXPRINT_BUILD_NUMBER`` is
    copied into that name so an existing CI variable still selects the build.
    """
    source = dict(os.environ if env is None else env)
    legacy = source.get("VOXPRINT_BUILD_NUMBER", "").strip()
    if legacy.isdigit() and not source.get("VOXPRINT_BUILD", "").strip():
        source["VOXPRINT_BUILD"] = legacy
    return source


def build_info(env: Mapping[str, str] | None = None) -> BuildInfo:
    """Return the identity parsed from this checkout's ``BUILD.json``.

    Args:
        env: Environment for the build-number override. Defaults to the
            process environment.

    Returns:
        The shared build identity. ``app`` is ``clarity`` when the file
        omits it.
    """
    return load(_BUILD_PATH, app="clarity", env=_build_env(env))


def _raw_version() -> str:
    """Return the ``version`` string as it is written in ``BUILD.json``."""
    try:
        data = json.loads(_BUILD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"BUILD.json could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("BUILD.json must be an object")
    value = data.get("version")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("BUILD.json version must be a non-empty string")
    return value


_FILE = load(_BUILD_PATH, app="clarity", env={})
APP_VERSION = _raw_version()
APP_BUILD = _FILE.build
CODENAME = _FILE.codename


def app_build() -> int | None:
    """Build number from ``VOXPRINT_BUILD``, otherwise the number in ``BUILD.json``.

    A missing recorded build stays ``None``. :func:`version_label` renders that
    as ``dev``.
    """
    return build_info().build


def version_label() -> str:
    """Human version from ``voxprint_common.buildinfo.label``.

    Returns:
        Text such as ``1.0.0 RC build 1002 "Codename"``.
    """
    return label(build_info())


def release_title() -> str:
    """GitHub pre-release title."""
    return f"{APP_NAME} {version_label()}"
