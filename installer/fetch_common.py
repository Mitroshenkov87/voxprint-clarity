"""Download the pinned voxprint-common wheel and check its release digest.

The installer downloads the wheel and the ``.sha256`` file published next to
it. The digest of the wheel must match that sidecar and the digest recorded
in ``runtime_lock.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

WHEEL_NAME = "voxprint_common-0.1.1-py3-none-any.whl"


class FetchError(RuntimeError):
    """The wheel could not be downloaded or its digest did not match."""


def direct_url(requirement: str) -> str:
    """Return the URL from a pip direct requirement.

    Args:
        requirement: Text such as ``name @ https://...``.

    Returns:
        The URL with surrounding space removed.

    Raises:
        FetchError: The requirement has no ``@`` URL.
    """
    text = requirement.strip()
    marker = " @ "
    if marker not in text:
        raise FetchError(f"requirement has no URL: {requirement}")
    url = text.split(marker, 1)[1].strip()
    if not url.startswith("https://"):
        raise FetchError(f"requirement URL must be https: {url}")
    return url


def parse_sha256_sidecar(text: str, wheel_name: str = WHEEL_NAME) -> str:
    """Return the hex digest from a ``.sha256`` sidecar.

    Args:
        text: File contents. The first line is ``digest``, ``digest  name``,
            or ``digest *name``.
        wheel_name: Filename that must match when the line names one.

    Returns:
        The 64-character hex digest.

    Raises:
        FetchError: The line is empty, the digest is not hex, or the name differs.
    """
    line = ""
    for candidate in text.splitlines():
        if candidate.strip():
            line = candidate.strip()
            break
    if not line:
        raise FetchError("the .sha256 file is empty")
    parts = line.split()
    digest = parts[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FetchError("the .sha256 file does not start with a sha256 digest")
    if len(parts) > 1:
        name = parts[-1].lstrip("*")
        if name != wheel_name:
            raise FetchError(f"the .sha256 file names {name}, expected {wheel_name}")
    return digest


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_wheel(path: Path, sidecar: str, expected: str, wheel_name: str = WHEEL_NAME) -> str:
    """Check ``path`` against the sidecar and the pinned digest.

    Args:
        path: Downloaded wheel.
        sidecar: Contents of the ``.sha256`` file.
        expected: Digest recorded beside the pin.
        wheel_name: Filename the sidecar may name.

    Returns:
        The matching digest.

    Raises:
        FetchError: The wheel, the sidecar, and the pin do not all match.
    """
    actual = file_sha256(path)
    published = parse_sha256_sidecar(sidecar, wheel_name)
    pinned = expected.strip().lower()
    if actual != published or actual != pinned:
        raise FetchError(
            "voxprint-common wheel sha256 does not match the release .sha256 "
            f"(file {actual}, sidecar {published}, pin {pinned})"
        )
    return actual


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-clarity"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            dest.write_bytes(response.read())
    except OSError as exc:
        raise FetchError(f"download failed for {url}: {exc}") from exc


def load_lock(path: Path) -> tuple[str, str, str]:
    """Return the wheel URL, the sidecar URL, and the pinned digest.

    Args:
        path: ``runtime_lock.json``.

    Raises:
        FetchError: A required field is missing.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"runtime lock could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise FetchError("runtime lock must be an object")
    requirement = payload.get("voxprint-common_requirement")
    expected = payload.get("voxprint-common_sha256")
    sidecar = payload.get("voxprint-common_sha256_url")
    if not isinstance(requirement, str) or not isinstance(expected, str):
        raise FetchError("runtime lock is missing the voxprint-common pin")
    wheel = direct_url(requirement)
    if not isinstance(sidecar, str) or not sidecar.strip():
        sidecar = wheel + ".sha256"
    return wheel, sidecar.strip(), expected.strip()


def fetch(lock: Path, dest: Path) -> Path:
    """Download the wheel and the sidecar, then verify the digest.

    Args:
        lock: ``runtime_lock.json``.
        dest: Directory that receives the wheel. Created when missing.

    Returns:
        Path of the verified wheel.
    """
    wheel_url, sidecar_url, expected = load_lock(lock)
    dest.mkdir(parents=True, exist_ok=True)
    wheel_path = dest / WHEEL_NAME
    sidecar_path = dest / f"{WHEEL_NAME}.sha256"
    _download(wheel_url, wheel_path)
    _download(sidecar_url, sidecar_path)
    verify_wheel(wheel_path, sidecar_path.read_text(encoding="utf-8"), expected)
    return wheel_path


def main(argv: list[str] | None = None) -> int:
    """Download or check the pinned wheel. Prints the wheel path on success."""
    parser = argparse.ArgumentParser(description="Fetch and verify the voxprint-common wheel.")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--wheel", type=Path, help="Already downloaded wheel. Skips the wheel download.")
    parser.add_argument("--sidecar", type=Path, help="Already downloaded .sha256 file.")
    args = parser.parse_args(argv)
    try:
        if args.wheel is not None:
            if args.sidecar is None:
                raise FetchError("--sidecar is required with --wheel")
            _wheel_url, _sidecar_url, expected = load_lock(args.lock)
            verify_wheel(args.wheel, args.sidecar.read_text(encoding="utf-8"), expected)
            print(args.wheel)
            return 0
        if args.dest is None:
            raise FetchError("--dest is required when downloading")
        print(fetch(args.lock, args.dest))
    except FetchError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
