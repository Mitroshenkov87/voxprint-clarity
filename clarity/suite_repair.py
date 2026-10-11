"""Re-check the shared suite and repair a component that is missing.

The Windows installer and the Linux install script place Python, torch, ffmpeg,
the engine, and the default models into :func:`clarity.suite_share.shared_root`
before the first launch. Launch and ``--selftest`` call :func:`ensure_suite`
again. A component that is already on the machine is adopted. A component that
is missing is installed, and the progress callback reports each step.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Never

from clarity.engine import models_directory
from clarity.errors import ClarityError
from clarity.models import clean_models, fetch_weights, strong_model
from clarity.suite_share import (
    COMPONENTS,
    ComponentName,
    component_version,
    marker_path,
    retain_component,
    shared_root,
    write_marker,
)

Progress = Callable[[float, str], None]
Probe = Callable[[ComponentName, Path], bool]
Install = Callable[[ComponentName, Path, Progress], None]

_FFMPEG_URL = {
    "win32": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip",
    "linux": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl.tar.xz",
}
_MAX_FFMPEG_BYTES = 256 * 1024 * 1024


class SuiteMissing(ClarityError):
    """A shared component is missing and the repair did not restore it."""


def prefer_shared_ffmpeg(root: Path | None = None) -> None:
    """Put the shared ffmpeg directory first on ``PATH`` when it exists.

    Args:
        root: Suite directory. ``None`` uses :func:`shared_root`.
    """
    folder = (root or shared_root()) / "ffmpeg"
    binary = folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not binary.is_file():
        return
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = str(folder) + os.pathsep + current


def default_probe(name: ComponentName, root: Path) -> bool:
    """Return whether one component is already usable.

    A ready marker wins. Otherwise Python 3.14, an importable torch, ffmpeg on
    ``PATH`` or in the shared folder, an importable engine, or the default
    weight files count.

    Args:
        name: Component name.
        root: Suite directory.

    Returns:
        True when the component does not need to be downloaded.
    """
    if marker_path(name, root).is_file():
        return True
    if name == "python":
        return sys.version_info[:2] == (3, 14)
    if name == "torch":
        return _spec_exists("torch")
    if name == "ffmpeg":
        return shutil.which("ffmpeg") is not None or _ffmpeg_binary(root).is_file()
    if name == "engine":
        return _spec_exists("voxprint_engine")
    if name == "models":
        return _default_weights_present()
    unexpected: Never = name
    raise SuiteMissing(f"unknown suite component {unexpected}")


def ensure_suite(
    progress: Progress,
    *,
    root: Path | None = None,
    probe: Probe | None = None,
    install: Install | None = None,
) -> list[ComponentName]:
    """Re-check every shared component and repair the ones that are missing.

    Args:
        progress: Called with a fraction from 0 to 1 and a short message.
        root: Suite directory. ``None`` uses :func:`shared_root`.
        probe: Replacement for :func:`default_probe`.
        install: Replacement for :func:`install_component`.

    Returns:
        The component names that were repaired.

    Raises:
        SuiteMissing: When a repair finishes without a ready marker.
    """
    folder = root or shared_root()
    prefer_shared_ffmpeg(folder)
    look = probe or default_probe
    repair = install or install_component
    repaired: list[ComponentName] = []
    total = len(COMPONENTS)
    for index, name in enumerate(COMPONENTS):
        if look(name, folder):
            retain_component(name, component_version(name), root=folder)
            if not marker_path(name, folder).is_file():
                write_marker(name, folder)
            progress((index + 1) / total, f"{name} ready")
            continue
        progress(index / total, f"repairing {name}")
        repair(name, folder, progress)
        if not marker_path(name, folder).is_file() and not look(name, folder):
            raise SuiteMissing(f"{name} is still missing after repair")
        if not marker_path(name, folder).is_file():
            write_marker(name, folder)
        retain_component(name, component_version(name), root=folder)
        repaired.append(name)
        progress((index + 1) / total, f"{name} repaired")
    prefer_shared_ffmpeg(folder)
    return repaired


def install_component(name: ComponentName, root: Path, progress: Progress) -> None:
    """Install one missing component into the shared directory.

    Args:
        name: Component name.
        root: Suite directory.
        progress: Receives download progress for ffmpeg.

    Raises:
        SuiteMissing: When the component cannot be installed from here.
    """
    if os.environ.get("VOXPRINT_SUITE_INSTALLING") == "1":
        write_marker(name, root)
        return
    if name == "ffmpeg":
        install_ffmpeg(root, progress)
    elif name == "models":
        install_default_models(progress)
    elif name == "python" or name == "torch" or name == "engine":
        _run_installer(progress)
    else:
        unexpected: Never = name
        raise SuiteMissing(f"unknown suite component {unexpected}")
    write_marker(name, root)


def install_default_models(progress: Progress | None = None) -> list[str]:
    """Fetch the Clean and Strong weights into the engine models folder.

    Args:
        progress: Optional status callback. The engine download has its own progress.

    Returns:
        The model ids that were requested.
    """
    if progress is not None:
        progress(0.0, "repairing models")
    ids = [model.id for model in (*clean_models(), strong_model())]
    fetch_weights(ids, dry_run=False)
    return ids


def install_ffmpeg(root: Path, progress: Progress) -> Path:
    """Download the LGPL ffmpeg build into ``root / "ffmpeg"``.

    Args:
        root: Suite directory.
        progress: Called as the archive downloads.

    Returns:
        Path of the ffmpeg binary.

    Raises:
        SuiteMissing: When this platform has no pinned build, or the download fails.
    """
    if sys.platform == "win32":
        url = _FFMPEG_URL["win32"]
    elif sys.platform.startswith("linux"):
        url = _FFMPEG_URL["linux"]
    else:
        url = ""
    if not url.startswith("https://"):
        raise SuiteMissing("ffmpeg repair needs an https LGPL build for this platform")
    folder = root / "ffmpeg"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / ("ffmpeg.zip" if url.endswith(".zip") else "ffmpeg.tar.xz")
    _download(url, archive, progress)
    binary_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    target = folder / binary_name
    _extract_binary(archive, target)
    probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    probe = folder / probe_name
    try:
        _extract_binary(archive, probe)
    except SuiteMissing:
        probe.unlink(missing_ok=True)
    archive.unlink(missing_ok=True)
    if os.name != "nt":
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
        if probe.is_file():
            probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    progress(1.0, "ffmpeg repaired")
    return target


def write_ready_markers(root: Path | None = None) -> Path:
    """Mark every component ready without downloading.

    CI uses this before ``--selftest`` so the re-check runs against a suite the
    installer would already have written. A real install does not call it.

    Args:
        root: Suite directory. ``None`` uses :func:`shared_root`.

    Returns:
        The suite directory.
    """
    folder = root or shared_root()
    for name in COMPONENTS:
        write_marker(name, folder)
        retain_component(name, component_version(name), root=folder)
    return folder


def _default_weights_present() -> bool:
    try:
        folder = models_directory()
    except ClarityError:
        return False
    names = [model.filename for model in (*clean_models(), strong_model())]
    return all((folder / name).is_file() for name in names)


def _spec_exists(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _ffmpeg_binary(root: Path) -> Path:
    return root / "ffmpeg" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


class _HttpsRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves https."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not str(newurl).startswith("https://"):
            raise SuiteMissing("the download was redirected off https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, target: Path, progress: Progress) -> None:
    if not url.startswith("https://"):
        raise SuiteMissing("downloads must use https")
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-clarity"})
    opener = urllib.request.build_opener(_HttpsRedirect)
    try:
        with opener.open(request, timeout=60) as response:
            final = str(response.geturl())
            if not final.startswith("https://"):
                raise SuiteMissing("the download was redirected off https")
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with target.open("wb") as handle:
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > _MAX_FFMPEG_BYTES:
                        raise SuiteMissing("the ffmpeg download is larger than 256 MB")
                    handle.write(block)
                    if total > 0:
                        progress(min(0.9, written / total), "repairing ffmpeg")
    except SuiteMissing:
        target.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        target.unlink(missing_ok=True)
        raise SuiteMissing(f"ffmpeg download failed: {exc}") from exc


def _extract_binary(archive: Path, target: Path) -> None:
    wanted = target.name
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            member = _named_member(bundle.namelist(), wanted)
            if member is None:
                raise SuiteMissing(f"the ffmpeg archive has no {wanted}")
            with bundle.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        return
    with tarfile.open(archive, "r:xz") as bundle:
        chosen: tarfile.TarInfo | None = None
        for info in bundle.getmembers():
            if Path(info.name).name == wanted and info.isfile():
                chosen = info
                break
        if chosen is None:
            raise SuiteMissing(f"the ffmpeg archive has no {wanted}")
        extracted = bundle.extractfile(chosen)
        if extracted is None:
            raise SuiteMissing(f"the ffmpeg archive has no {wanted}")
        with extracted, target.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)


def _named_member(names: list[str], wanted: str) -> str | None:
    for name in names:
        if Path(name).name == wanted:
            return name
    return None


def _run_installer(progress: Progress) -> None:
    script = _installer_script()
    if script is None:
        raise SuiteMissing(
            "Python, torch, or the engine is missing. Re-run the Clarity installer so it can "
            f"place them in {shared_root()}."
        )
    progress(0.0, f"repairing {script.name}")
    if script.suffix == ".ps1":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-AppDir",
            str(script.parent if script.parent.name != "installer" else script.parents[1]),
            "-Requirements",
            str(_requirements_near(script)),
            "-Lock",
            str(_lock_near(script)),
            "-Log",
            str(shared_root() / "repair.log"),
        ]
    else:
        command = [str(script)]
    env = os.environ.copy()
    env["VOXPRINT_SUITE_INSTALLING"] = "1"
    try:
        subprocess.run(command, check=True, env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SuiteMissing(f"the suite installer failed: {exc}") from exc


def _installer_script() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        candidates = [root / "install-runtime.ps1", root / "installer" / "install-runtime.ps1"]
    else:
        candidates = [root / "install-shared.sh", root / "installer" / "linux" / "install-shared.sh"]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _requirements_near(script: Path) -> Path:
    for folder in (script.parent, script.parents[1] if len(script.parents) > 1 else script.parent):
        candidate = folder / "requirements.txt"
        if candidate.is_file():
            return candidate
    return script.parent / "requirements.txt"


def _lock_near(script: Path) -> Path:
    for folder in (script.parent, script.parent / "installer"):
        candidate = folder / "runtime_lock.json"
        if candidate.is_file():
            return candidate
    return script.parent / "runtime_lock.json"

