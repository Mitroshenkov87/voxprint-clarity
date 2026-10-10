"""``.vxclarity`` projects. A ZIP with an uncompressed ``mimetype`` entry, like EPUB.

The public Audiobook Builder tree (949dc13) has no ``.vxbook`` writer. This
format follows that suite's document shape as specified for Clarity: stored
mimetype first, then JSON sidecars, unknown fields kept, a higher major
version refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clarity.appinfo import APP_NAME, APP_VERSION
from clarity.errors import ClarityError
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings
from clarity.pipeline.resume import load_done, load_plan

FORMAT_MAJOR = 1
MIME = "application/vnd.voxprint.clarity+zip"
SUFFIX = ".vxclarity"
MIB = 1024 * 1024
RECENT_LIMIT = 8

_MANIFEST = "manifest.json"
_SETTINGS = "settings.json"
_RESUME = "resume.json"
_THUMB = "thumb.png"
_MIME_NAME = "mimetype"

_SETTING_FIELDS = (
    "mode",
    "upscale",
    "anime",
    "upscale_strength",
    "sharpen",
    "white_balance",
    "levels",
    "quality_factor",
    "sigma",
    "tile",
    "batch",
    "overlap",
    "container",
    "codec",
    "copy_audio",
    "max_frames",
)


class ProjectFormatError(ClarityError):
    """The file is not a Clarity project this build can read."""


@dataclass
class Project:
    path: Path
    manifest: dict[str, Any]
    settings_raw: dict[str, Any]
    resume: dict[str, Any]
    settings: Settings
    destination: Path | None
    source: Path | None
    thumb: bytes | None


def major_of(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ProjectFormatError("format_version is missing")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    head = text.split(".", 1)[0]
    if not head.lstrip("-").isdigit():
        raise ProjectFormatError(f"format_version {value!r} is not a number")
    return int(head)


def _require_major(value: object) -> int:
    major = major_of(value)
    if major != FORMAT_MAJOR:
        raise ProjectFormatError(
            f"This project is format version {value}. Clarity reads major version {FORMAT_MAJOR}."
        )
    return major


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint(path: Path, *, project: Path | None = None, full: bool = False) -> dict[str, Any]:
    """Size plus the first and last mebibyte. The full file hash is optional."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(MIB)
        if size > MIB:
            handle.seek(size - MIB)
            tail = handle.read(MIB)
        else:
            tail = head
        digest = ""
        if full:
            handle.seek(0)
            hasher = hashlib.sha256()
            while True:
                block = handle.read(MIB)
                if not block:
                    break
                hasher.update(block)
            digest = hasher.hexdigest()
    relative = path.name
    if project is not None:
        try:
            relative = str(path.resolve().relative_to(project.resolve().parent))
        except ValueError:
            relative = path.name
    return {
        "absolute": str(path.resolve()),
        "relative": relative,
        "size": size,
        "head_sha256": _sha256(head),
        "tail_sha256": _sha256(tail),
        "sha256": digest,
    }


def _matches(path: Path, source: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    recorded_size = source.get("size")
    if isinstance(recorded_size, int) and path.stat().st_size != recorded_size:
        return False
    found = fingerprint(path)
    head = str(source.get("head_sha256") or "")
    tail = str(source.get("tail_sha256") or "")
    if head and found["head_sha256"] != head:
        return False
    if tail and found["tail_sha256"] != tail:
        return False
    return True


def resolve_source(project: Path, source: dict[str, Any]) -> Path | None:
    """Prefer a fingerprint match of the absolute path, then the path relative to the project."""
    absolute = Path(str(source.get("absolute") or ""))
    relative_text = str(source.get("relative") or "")
    relative = project.parent / relative_text if relative_text else None
    if absolute.is_file() and _matches(absolute, source):
        return absolute
    if relative is not None and relative.is_file() and _matches(relative, source):
        return relative
    if absolute.is_file():
        return absolute
    if relative is not None and relative.is_file():
        return relative
    return None


def _read_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        payload = archive.read(name)
    except KeyError as exc:
        raise ProjectFormatError(f"{name} is missing") from exc
    try:
        data = json.loads(payload.decode("utf-8"))
    except ValueError as exc:
        raise ProjectFormatError(f"{name} is not JSON") from exc
    if not isinstance(data, dict):
        raise ProjectFormatError(f"{name} must be an object")
    return data


def _settings_from(raw: dict[str, Any]) -> tuple[Settings, Path | None]:
    kwargs: dict[str, Any] = {}
    for key in _SETTING_FIELDS:
        if key in raw:
            kwargs[key] = raw[key]
    container = kwargs.get("container", Container.MKV.value)
    codec = kwargs.get("codec", VideoCodec.AV1.value)
    try:
        kwargs["container"] = Container(str(container))
        kwargs["codec"] = VideoCodec(str(codec))
    except ValueError as exc:
        raise ProjectFormatError(str(exc)) from exc
    destination = raw.get("destination")
    path = Path(str(destination)) if isinstance(destination, str) and destination else None
    return Settings(**kwargs), path


def _settings_document(settings: Settings, destination: Path | None, previous: dict[str, Any]) -> dict[str, Any]:
    merged = dict(previous)
    merged.update(settings.digest_payload())
    if destination is not None:
        merged["destination"] = str(destination)
    return merged


def work_dir_for(project: Path, resume: dict[str, Any]) -> Path:
    recorded = resume.get("work")
    if isinstance(recorded, str) and recorded and Path(recorded).is_dir():
        return Path(recorded)
    return project.with_name(project.name + ".work")


def resume_from_work(work: Path) -> dict[str, Any]:
    plan = load_plan(work / "plan.json") or {}
    done = load_done(work / "journal.jsonl")
    journal = [
        {"index": record.index, "start": record.start, "stop": record.stop, "sha256": record.sha256}
        for record in sorted(done.values(), key=lambda item: item.index)
    ]
    return {
        "work": str(work),
        "ranges": plan.get("ranges", []),
        "cuts": plan.get("cuts", []),
        "settings_sha256": plan.get("settings_sha256", ""),
        "source": plan.get("source", ""),
        "journal": journal,
    }


def _zip_info(name: str, *, stored: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.extra = b""
    info.flag_bits = 0
    return info


def _write_archive(path: Path, entries: list[tuple[str, bytes, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(partial, "w", allowZip64=False) as archive:
        for name, payload, stored in entries:
            archive.writestr(_zip_info(name, stored=stored), payload)
    os.replace(partial, path)


def save_project(
    path: Path,
    *,
    source: Path,
    settings: Settings,
    destination: Path | None = None,
    resume: dict[str, Any] | None = None,
    thumb: bytes | None = None,
    full_hash: bool = False,
    previous: Project | None = None,
) -> Project:
    """Write a project. Unknown keys already in ``previous`` are copied forward."""
    if path.suffix.lower() != SUFFIX:
        path = path.with_suffix(SUFFIX)
    if not source.is_file():
        raise ProjectFormatError("the source video was not found")
    old_manifest = dict(previous.manifest) if previous is not None else {}
    old_settings = dict(previous.settings_raw) if previous is not None else {}
    old_resume = dict(previous.resume) if previous is not None else {}
    created = str(old_manifest.get("created") or _stamp())
    manifest = dict(old_manifest)
    manifest.update(
        {
            "format": "vxclarity",
            "format_version": FORMAT_MAJOR,
            "app": APP_NAME,
            "app_version": APP_VERSION,
            "created": created,
            "modified": _stamp(),
            "source": fingerprint(source, project=path, full=full_hash),
        }
    )
    _require_major(manifest["format_version"])
    settings_raw = _settings_document(settings, destination, old_settings)
    resume_raw = dict(old_resume)
    if resume is not None:
        resume_raw.update(resume)
    entries: list[tuple[str, bytes, bool]] = [
        (_MIME_NAME, MIME.encode("ascii"), True),
        (_MANIFEST, _dump(manifest), False),
        (_SETTINGS, _dump(settings_raw), False),
        (_RESUME, _dump(resume_raw), False),
    ]
    stored_thumb = thumb if thumb is not None else (previous.thumb if previous is not None else None)
    if stored_thumb:
        entries.append((_THUMB, stored_thumb, False))
    _write_archive(path, entries)
    loaded = load_project(path)
    return loaded


def _dump(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")


def load_project(path: Path) -> Project:
    if not path.is_file():
        raise ProjectFormatError("the project file was not found")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ProjectFormatError("the project is not a zip file") from exc
    with archive:
        names = archive.namelist()
        if not names or names[0] != _MIME_NAME:
            raise ProjectFormatError("mimetype must be the first entry")
        info = archive.getinfo(_MIME_NAME)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ProjectFormatError("mimetype must be stored uncompressed")
        found = archive.read(_MIME_NAME)
        if found != MIME.encode("ascii"):
            raise ProjectFormatError("mimetype is not a Clarity project")
        manifest = _read_json(archive, _MANIFEST)
        _require_major(manifest.get("format_version"))
        settings_raw = _read_json(archive, _SETTINGS)
        resume = _read_json(archive, _RESUME)
        thumb = archive.read(_THUMB) if _THUMB in names else None
    settings, destination = _settings_from(settings_raw)
    source_raw = manifest.get("source")
    source = resolve_source(path, source_raw) if isinstance(source_raw, dict) else None
    return Project(
        path=path,
        manifest=manifest,
        settings_raw=settings_raw,
        resume=resume,
        settings=settings,
        destination=destination,
        source=source,
        thumb=thumb,
    )


def sync_resume(path: Path, work: Path) -> Project:
    """Copy the on-disk journal into an existing project, keeping every other field."""
    current = load_project(path)
    if current.source is None or not current.source.is_file():
        raise ProjectFormatError("the source video was not found")
    snapshot = dict(current.resume)
    snapshot.update(resume_from_work(work))
    return save_project(
        path,
        source=current.source,
        settings=current.settings,
        destination=current.destination,
        resume=snapshot,
        thumb=current.thumb,
        previous=current,
    )


def state_dir() -> Path:
    override = os.environ.get("VOXPRINT_CLARITY_STATE", "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        windows_base = Path(local) if local else Path.home()
        return windows_base / "Voxprint" / "Clarity"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    linux_base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return linux_base / "voxprint" / "clarity"


def recent_file(folder: Path | None = None) -> Path:
    return (folder if folder is not None else state_dir()) / "recent.json"


def load_recent(folder: Path | None = None) -> list[str]:
    path = recent_file(folder)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    raw = data.get("paths") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def remember_recent(path: Path, folder: Path | None = None) -> list[str]:
    text = str(path.resolve())
    paths = [item for item in load_recent(folder) if item != text]
    paths.insert(0, text)
    paths = paths[:RECENT_LIMIT]
    target = recent_file(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"paths": paths}, indent=1) + "\n", encoding="utf-8")
    return paths
