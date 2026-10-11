"""Versioned suite folder shared by Clarity and the other Voxprint apps.

``voxprint_engine.paths`` already owns the Voxprint home directory and the
models folder. It does not own a versioned runtime folder or an integer
refcount. This module follows that home directory and adds ``shared/``, which
is the layout proposed for the engine in ``docs/UPSTREAM-SHARED.md``.
Weights stay in the engine models folder.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal, Never

_engine_paths: Any
try:
    _engine_paths = importlib.import_module("voxprint_engine.paths")
except ImportError:
    _engine_paths = None

ComponentName = Literal["python", "torch", "ffmpeg", "engine", "models"]
COMPONENTS: tuple[ComponentName, ...] = ("python", "torch", "ffmpeg", "engine", "models")
APP_KEY = "clarity"
_RUNTIME_PARTS: frozenset[str] = frozenset({"python", "torch", "engine"})


def voxprint_home() -> Path:
    """Return the suite home directory.

    ``VOXPRINT_HOME`` wins. Otherwise this matches ``voxprint_engine.paths``
    when that module imports, and the same rules when it does not: Windows uses
    ``%LOCALAPPDATA%\\Voxprint`` and Linux uses ``$XDG_DATA_HOME/voxprint``
    or ``~/.local/share/voxprint``.

    Returns:
        The Voxprint home directory. It is not created here.
    """
    env = os.environ.get("VOXPRINT_HOME", "").strip()
    if env:
        return Path(env)
    if _engine_paths is not None:
        return Path(_engine_paths.voxprint_home())
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "Voxprint"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "voxprint"


def shared_root() -> Path:
    """Return ``<Voxprint home>/shared``.

    Returns:
        The versioned suite directory. It is not created here.
    """
    return voxprint_home() / "shared"


def manifest_path(root: Path | None = None) -> Path:
    """Return the refcount manifest path.

    Args:
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        ``root / "manifest.json"``.
    """
    return (root or shared_root()) / "manifest.json"


def component_version(name: ComponentName) -> str:
    """Return the pinned version string for one shared component.

    Args:
        name: ``python``, ``torch``, ``ffmpeg``, ``engine``, or ``models``.

    Returns:
        The version recorded in the manifest.
    """
    if name == "python":
        return "3.14"
    if name == "torch":
        return "2.14.1+cu130"
    if name == "ffmpeg":
        return "lgpl"
    if name == "engine":
        return "engine-v1.0.0"
    if name == "models":
        return "default"
    unexpected: Never = name
    raise RuntimeError(f"unknown suite component {unexpected}")


def runtime_dirname() -> str:
    """Return the versioned runtime folder name.

    Returns:
        ``py3.14-torch2.14.1-cu130``.
    """
    return "py3.14-torch2.14.1-cu130"


def read_manifest(root: Path | None = None) -> dict[str, Any]:
    """Read the refcount manifest.

    Args:
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        The object, or an empty manifest when the file is missing or unreadable.
    """
    path = manifest_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        return {"format_version": 1, "components": {}}
    components = data.get("components")
    if not isinstance(components, dict):
        data["components"] = {}
    data["format_version"] = 1
    return data


def write_manifest(data: dict[str, Any], root: Path | None = None) -> None:
    """Replace the manifest after the new JSON is fully written.

    Args:
        data: Manifest object.
        root: Suite directory. ``None`` uses :func:`shared_root`.
    """
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def hold_component(name: ComponentName, version: str, app: str = APP_KEY, root: Path | None = None) -> int:
    """Add one install reference from ``app`` to ``name``.

    Args:
        name: Component name.
        version: Version this app is holding.
        app: Suite app key. The default is Clarity.
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        The new reference count for ``app``.
    """
    return _record_user(name, version, app, root, increment=True)


def retain_component(name: ComponentName, version: str, app: str = APP_KEY, root: Path | None = None) -> int:
    """Record that ``app`` uses ``name`` without adding another install reference.

    Launch and ``--selftest`` call this. A second launch leaves an existing
    count unchanged. The first time ``app`` appears, the count becomes 1.

    Args:
        name: Component name.
        version: Version this app is using.
        app: Suite app key. The default is Clarity.
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        The reference count for ``app`` after the call.
    """
    return _record_user(name, version, app, root, increment=False)


def _record_user(name: str, version: str, app: str, root: Path | None, *, increment: bool) -> int:
    folder = root or shared_root()
    data = read_manifest(folder)
    components = data["components"]
    entry = components.get(name)
    if not isinstance(entry, dict):
        entry = {"version": version, "users": {}}
    users = entry.get("users")
    if not isinstance(users, dict):
        users = {}
    current = int(users.get(app) or 0)
    users[app] = current + 1 if increment else max(current, 1)
    entry["version"] = version
    entry["users"] = users
    components[name] = entry
    write_manifest(data, folder)
    return int(users[app])


def release_component(name: ComponentName, app: str = APP_KEY, root: Path | None = None) -> int:
    """Drop one reference. A component directory is removed when nobody holds it.

    The models folder itself is the engine store, so this never deletes weight
    files. The shared runtime directory is removed only when Python, torch, and
    the engine all have no remaining users.

    Args:
        name: Component name.
        app: Suite app key.
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        The remaining reference count for ``app``.
    """
    folder = root or shared_root()
    data = read_manifest(folder)
    components = data["components"]
    entry = components.get(name)
    if not isinstance(entry, dict):
        return 0
    users = entry.get("users")
    if not isinstance(users, dict):
        users = {}
    current = int(users.get(app) or 0)
    if current <= 1:
        users.pop(app, None)
        left = 0
    else:
        left = current - 1
        users[app] = left
    entry["users"] = users
    if users:
        components[name] = entry
    else:
        components.pop(name, None)
    write_manifest(data, folder)
    _delete_unheld(name, data, folder)
    return left


def release_app(app: str = APP_KEY, root: Path | None = None) -> None:
    """Drop one reference on every component for ``app``.

    Args:
        app: Suite app key.
        root: Suite directory. ``None`` uses :func:`shared_root`.
    """
    for name in COMPONENTS:
        release_component(name, app, root)


def _delete_unheld(name: str, data: dict[str, Any], root: Path) -> None:
    components = data["components"]
    if name in components:
        return
    marker = root / name / "READY"
    marker.unlink(missing_ok=True)
    if name == "ffmpeg":
        shutil.rmtree(root / "ffmpeg", ignore_errors=True)
        return
    if name == "models" or name not in _RUNTIME_PARTS:
        return
    if any(part in components for part in _RUNTIME_PARTS):
        return
    shutil.rmtree(root / "runtime", ignore_errors=True)
    for part in _RUNTIME_PARTS:
        shutil.rmtree(root / part, ignore_errors=True)


def marker_path(name: ComponentName, root: Path | None = None) -> Path:
    """Return the ready marker for one component.

    Args:
        name: Component name.
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        ``root / name / "READY"``.
    """
    return (root or shared_root()) / name / "READY"


def write_marker(name: ComponentName, root: Path | None = None) -> None:
    """Write the ready marker for one component.

    Args:
        name: Component name.
        root: Suite directory. ``None`` uses :func:`shared_root`.
    """
    path = marker_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ready\n", encoding="utf-8")
