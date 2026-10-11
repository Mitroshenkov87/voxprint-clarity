"""LGPL ffmpeg in the shared suite folder.

``voxprint_engine.suite_runtime`` installs Python, the CUDA wheels, and the
engine's default model. It does not install ffmpeg. Clarity keeps this one
piece: an https download of the BtbN LGPL build into ``shared/ffmpeg``.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

from clarity.engine import shared_directory
from clarity.errors import ClarityError
from clarity.net import install_system_trust

Progress = Callable[[str], None]
_FFMPEG_URL = {
    "win32": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip",
    "linux": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl.tar.xz",
}
_MAX_FFMPEG_BYTES = 256 * 1024 * 1024


class FfmpegMissing(ClarityError):
    """The LGPL ffmpeg build could not be installed."""


def prefer_shared_ffmpeg() -> None:
    """Put ``shared/ffmpeg`` first on ``PATH`` when the binary is there."""
    binary = _binary(shared_directory())
    if not binary.is_file():
        return
    folder = str(binary.parent)
    current = os.environ.get("PATH", "")
    if current.split(os.pathsep)[0] == folder:
        return
    os.environ["PATH"] = folder + os.pathsep + current


def ffmpeg_ready() -> bool:
    """Return whether ffmpeg is already on ``PATH`` or in the shared folder."""
    if os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1":
        return True
    if shutil.which("ffmpeg"):
        return True
    return _binary(shared_directory()).is_file()


def ensure_ffmpeg(progress: Progress) -> str:
    """Download the LGPL ffmpeg build when the machine does not already have one.

    ``VOXPRINT_CI_LANCZOS=1`` skips the download, the same way the engine skips
    its own suite install. That flag still does not start a restore.

    Args:
        progress: Receives ``Installing ffmpeg`` while the archive downloads.

    Returns:
        ``skipped``, ``ready``, or ``repaired``.

    Raises:
        FfmpegMissing: When this platform has no https build, or the download fails.
    """
    if os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1":
        progress("ffmpeg skipped")
        return "skipped"
    if shutil.which("ffmpeg") or _binary(shared_directory()).is_file():
        prefer_shared_ffmpeg()
        progress("ffmpeg is ready")
        return "ready"
    progress("Installing ffmpeg")
    install_ffmpeg(progress)
    prefer_shared_ffmpeg()
    progress("ffmpeg is ready")
    return "repaired"


def install_ffmpeg(progress: Progress) -> Path:
    """Download the LGPL ffmpeg build into the shared folder.

    Args:
        progress: Called with a short status while the archive downloads.

    Returns:
        Path of the ffmpeg binary.

    Raises:
        FfmpegMissing: When this platform has no https build, or the download fails.
    """
    if sys.platform == "win32":
        url = _FFMPEG_URL["win32"]
    elif sys.platform.startswith("linux"):
        url = _FFMPEG_URL["linux"]
    else:
        url = ""
    if not url.startswith("https://"):
        raise FfmpegMissing("ffmpeg repair needs an https LGPL build for this platform")
    folder = shared_directory() / "ffmpeg"
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
    except FfmpegMissing:
        probe.unlink(missing_ok=True)
    archive.unlink(missing_ok=True)
    if os.name != "nt":
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
        if probe.is_file():
            probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    return target


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
            raise FfmpegMissing("the download was redirected off https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, target: Path, progress: Progress) -> None:
    if not url.startswith("https://"):
        raise FfmpegMissing("downloads must use https")
    install_system_trust()
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-clarity"})
    opener = urllib.request.build_opener(_HttpsRedirect)
    try:
        with opener.open(request, timeout=60) as response:
            final = str(response.geturl())
            if not final.startswith("https://"):
                raise FfmpegMissing("the download was redirected off https")
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with target.open("wb") as handle:
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > _MAX_FFMPEG_BYTES:
                        raise FfmpegMissing("the ffmpeg download is larger than 256 MB")
                    handle.write(block)
                    if total > 0:
                        progress("Installing ffmpeg")
    except FfmpegMissing:
        target.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        target.unlink(missing_ok=True)
        raise FfmpegMissing(f"ffmpeg download failed: {exc}") from exc


def _binary(root: Path) -> Path:
    return root / "ffmpeg" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def _extract_binary(archive: Path, target: Path) -> None:
    wanted = target.name
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            member = _named_member(bundle.namelist(), wanted)
            if member is None:
                raise FfmpegMissing(f"the ffmpeg archive has no {wanted}")
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
            raise FfmpegMissing(f"the ffmpeg archive has no {wanted}")
        extracted = bundle.extractfile(chosen)
        if extracted is None:
            raise FfmpegMissing(f"the ffmpeg archive has no {wanted}")
        with extracted, target.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)


def _named_member(names: list[str], wanted: str) -> str | None:
    for name in names:
        if Path(name).name == wanted:
            return name
    return None
