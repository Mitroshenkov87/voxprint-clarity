"""Copy ``enlarge.suite_runtime`` from the engine tag into site-packages.

The ``voxprint-engine`` wheel contains ``voxprint_engine`` only.
``ensure_runtime`` lives in ``enlarge.suite_runtime`` in the same tag archive.
Clarity calls that module and does not install the Enlarge application.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


class StageError(RuntimeError):
    """The Enlarge runtime package could not be copied into site-packages."""


def direct_url(requirement: str) -> str:
    """Return the https URL from a pip direct requirement."""
    text = requirement.strip()
    marker = " @ "
    if marker not in text:
        raise StageError(f"requirement has no URL: {requirement}")
    url = text.split(marker, 1)[1].strip()
    if not url.startswith("https://"):
        raise StageError(f"requirement URL must be https: {url}")
    return url


def engine_archive_url(lock: Path) -> str:
    """Return the GitHub archive URL from the engine pin in ``lock``."""
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageError(f"runtime lock could not be read: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("engine"), str):
        raise StageError("runtime lock is missing the engine pin")
    return direct_url(payload["engine"])


def site_packages(python: str) -> Path:
    """Return the purelib directory for ``python``."""
    completed = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=False,
        capture_output=True,
        text=True,
    )
    folder = (completed.stdout or "").strip()
    if completed.returncode != 0 or not folder:
        detail = (completed.stderr or "").strip() or "sysconfig failed"
        raise StageError(detail)
    return Path(folder)


def _enlarge_prefix(names: list[str]) -> str:
    suffix = "/enlarge/suite_runtime.py"
    for name in names:
        if name.endswith(suffix):
            return name[: -len("suite_runtime.py")]
    raise StageError("the engine archive has no enlarge/suite_runtime.py")


def extract_enlarge(archive: Path, destination: Path) -> None:
    """Extract the ``enlarge`` package from ``archive`` onto ``destination``.

    ``destination`` is the ``enlarge`` directory itself. An existing directory
    is replaced.

    Raises:
        StageError: The archive has no suite runtime, or a member escapes it.
    """
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        prefix = _enlarge_prefix(names)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for member in tar.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix) :]
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise StageError(f"refusing archive member {member.name}")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise StageError(f"could not read {member.name}")
            with extracted, target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
    if not (destination / "suite_runtime.py").is_file():
        raise StageError("enlarge.suite_runtime was not extracted")


def stage(lock: Path, python: str) -> Path:
    """Download the engine tag and install the ``enlarge`` package for ``python``.

    Returns:
        The ``enlarge`` directory under that interpreter's site-packages.
    """
    url = engine_archive_url(lock)
    folder = site_packages(python) / "enlarge"
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-clarity"})
    try:
        with tempfile.TemporaryDirectory() as temporary, urllib.request.urlopen(request, timeout=180) as response:
            archive = Path(temporary) / "engine.tar.gz"
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            extract_enlarge(archive, folder)
    except OSError as exc:
        raise StageError(f"engine archive download failed: {exc}") from exc
    return folder


def main(argv: list[str] | None = None) -> int:
    """Stage ``enlarge.suite_runtime`` and print the package directory."""
    parser = argparse.ArgumentParser(description="Install enlarge.suite_runtime from the engine tag.")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("runtime_lock.json"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        print(stage(args.lock, args.python))
    except StageError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
