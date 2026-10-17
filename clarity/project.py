"""``.vxclarity`` projects. The suite ZIP container, same shape as ``.vxbook`` v1.0.

``mimetype`` is stored first. ``manifest.json`` uses ``format_version``
``"MAJOR.MINOR"``, ``created_by``, ``generator``, ``source``, and a ``files``
map of sha256 and size. A reader accepts any minor of major 1 and refuses a
different major. Clarity-only video fingerprints live under ``x_clarity``.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voxprint_common.hashing import sha256_bytes, sha256_file

from clarity.appinfo import APP_NAME, APP_VERSION
from clarity.errors import ClarityError
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings
from clarity.pipeline.resume import load_done, load_plan

FORMAT_VERSION = "1.0"
FORMAT_MAJOR = 1
MIME = "application/vnd.voxprint.clarity+zip"
SUFFIX = ".vxclarity"
MIB = 1024 * 1024
RECENT_LIMIT = 8
_MAX_ENTRIES = 1000
_MAX_TOTAL = 512 * MIB
_MAX_JSON = 4 * MIB
_MAX_RATIO = 100
_MAX_DEPTH = 32
_VERSION_TEXT = re.compile(r"^(\d+)\.(\d+)$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

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
    "crf",
    "copy_audio",
    "max_frames",
)


class ProjectFormatError(ClarityError):
    """The file is not a Clarity project this build can read."""


@dataclass
class Project:
    """A loaded ``.vxclarity`` file and the settings the window can apply."""

    path: Path
    manifest: dict[str, Any]
    settings_raw: dict[str, Any]
    resume: dict[str, Any]
    settings: Settings
    destination: Path | None
    source: Path | None
    thumb: bytes | None
    extras: dict[str, bytes] = field(default_factory=dict)


def parse_format_version(value: object) -> tuple[int, int]:
    """``"MAJOR.MINOR"``. An integer is the major with minor 0 (draft files)."""
    if isinstance(value, bool) or value is None:
        raise ProjectFormatError("format_version is missing")
    if isinstance(value, int):
        return value, 0
    if isinstance(value, str):
        match = _VERSION_TEXT.fullmatch(value.strip())
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise ProjectFormatError(f"format_version {value!r} must be MAJOR.MINOR")


def _require_supported(value: object) -> tuple[int, int]:
    major, minor = parse_format_version(value)
    if major != FORMAT_MAJOR:
        raise ProjectFormatError(
            f"This project is format version {value}. Clarity reads major version {FORMAT_MAJOR}. "
            "Update Clarity to open a project made by a newer version."
        )
    return major, minor


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(payload: bytes) -> str:
    return sha256_bytes(payload)


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
        digest = sha256_file(path) if full else ""
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


def video_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    """The video location record. New files keep it in ``x_clarity.video``."""
    extra = manifest.get("x_clarity")
    if isinstance(extra, dict) and isinstance(extra.get("video"), dict):
        return extra["video"]
    source = manifest.get("source")
    if isinstance(source, dict) and ("head_sha256" in source or "absolute" in source):
        return source
    return {}


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ProjectFormatError(f"duplicate JSON key {key}")
        data[key] = value
    return data


def _walk_depth(value: object, level: int) -> None:
    if level > _MAX_DEPTH:
        raise ProjectFormatError("JSON nesting is deeper than 32")
    if isinstance(value, dict):
        for item in value.values():
            _walk_depth(item, level + 1)
        return
    if isinstance(value, list):
        for item in value:
            _walk_depth(item, level + 1)


def _read_json_bytes(payload: bytes, name: str) -> dict[str, Any]:
    if len(payload) > _MAX_JSON:
        raise ProjectFormatError(f"{name} is larger than 4 MiB")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ProjectFormatError(f"{name} is not UTF-8") from exc
    if text.startswith("\ufeff"):
        raise ProjectFormatError(f"{name} must not start with a BOM")
    try:
        data = json.loads(text, object_pairs_hook=_object_pairs)
    except ProjectFormatError:
        raise
    except ValueError as exc:
        raise ProjectFormatError(f"{name} is not JSON") from exc
    if not isinstance(data, dict):
        raise ProjectFormatError(f"{name} must be an object")
    _walk_depth(data, 1)
    return data


def _file_record(payload: bytes) -> dict[str, Any]:
    return {"sha256": sha256_bytes(payload), "size": len(payload)}


def _safe_entry_name(name: str) -> None:
    if not name or name in {".", "./"}:
        raise ProjectFormatError("an archive entry has an empty name")
    if "\\" in name or "\x00" in name or any(ord(char) < 32 for char in name):
        raise ProjectFormatError(f"unsafe archive entry {name!r}")
    if name.startswith("/") or (len(name) >= 2 and name[1] == ":"):
        raise ProjectFormatError(f"unsafe archive entry {name!r}")
    parts = [part for part in name.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ProjectFormatError(f"unsafe archive entry {name!r}")


def _unix_kind(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0o170000


def _check_archive(archive: zipfile.ZipFile) -> list[str]:
    infos = archive.infolist()
    if len(infos) > _MAX_ENTRIES:
        raise ProjectFormatError("the project has too many entries")
    names = [info.filename for info in infos]
    folded = [name.lower() for name in names]
    if len(folded) != len(set(folded)):
        raise ProjectFormatError("archive entry names are not unique")
    claimed = 0
    for info in infos:
        _safe_entry_name(info.filename)
        if any(ord(char) > 127 for char in info.filename) and not (info.flag_bits & 0x800):
            raise ProjectFormatError(f"{info.filename} is not marked as UTF-8")
        if info.flag_bits & 0x1:
            raise ProjectFormatError(f"{info.filename} is encrypted")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ProjectFormatError(f"{info.filename} uses an unsupported compression method")
        kind = _unix_kind(info)
        if kind not in {0, 0o100000, 0o040000}:
            raise ProjectFormatError(f"{info.filename} is not a regular file")
        if info.file_size < 0:
            raise ProjectFormatError(f"{info.filename} has a bad size")
        claimed += info.file_size
        if claimed > _MAX_TOTAL:
            raise ProjectFormatError("the project is larger than 512 MiB uncompressed")
        if info.file_size > 0 and (info.compress_size <= 0 or info.file_size > info.compress_size * _MAX_RATIO):
            raise ProjectFormatError(f"{info.filename} expands more than {_MAX_RATIO}:1")
    if not names or names[0] != _MIME_NAME:
        raise ProjectFormatError("mimetype must be the first entry")
    mime_info = archive.getinfo(_MIME_NAME)
    if mime_info.compress_type != zipfile.ZIP_STORED:
        raise ProjectFormatError("mimetype must be stored uncompressed")
    if mime_info.extra:
        raise ProjectFormatError("mimetype extra field must be empty")
    return names


@dataclass
class _Budget:
    used: int = 0


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: _Budget) -> bytes:
    """Count real output bytes while inflating, including entries the reader will ignore."""
    room = _MAX_TOTAL - budget.used
    chunks: list[bytes] = []
    total = 0
    try:
        handle = archive.open(info, "r")
    except (RuntimeError, zipfile.BadZipFile, ValueError) as exc:
        raise ProjectFormatError(f"{info.filename} could not be read") from exc
    with handle:
        while True:
            block = handle.read(64 * 1024)
            if not block:
                break
            total += len(block)
            if total > room:
                raise ProjectFormatError("the project is larger than 512 MiB uncompressed")
            if total > 0 and (info.compress_size <= 0 or total > info.compress_size * _MAX_RATIO):
                raise ProjectFormatError(f"{info.filename} expands more than {_MAX_RATIO}:1")
            chunks.append(block)
    budget.used += total
    return b"".join(chunks)


def _verify_files(payloads: dict[str, bytes], manifest: dict[str, Any], names: list[str]) -> dict[str, bytes]:
    listed = manifest.get("files")
    if not isinstance(listed, dict):
        raise ProjectFormatError("files is missing")
    checked: dict[str, bytes] = {}
    for name, meta in listed.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise ProjectFormatError("files entries must map a path to sha256 and size")
        digest = meta.get("sha256")
        size = meta.get("size")
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ProjectFormatError(f"files[{name}] sha256 must be 64 lowercase hex")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProjectFormatError(f"files[{name}] size is missing")
        if name not in names or name not in payloads:
            raise ProjectFormatError(f"{name} is listed in files but missing from the archive")
        payload = payloads[name]
        if len(payload) != size or sha256_bytes(payload) != digest:
            raise ProjectFormatError(f"{name} does not match the files checksum")
        checked[name] = payload
    for required in (_SETTINGS, _RESUME):
        if required not in names:
            raise ProjectFormatError(f"{required} is missing")
        if required not in listed:
            raise ProjectFormatError(f"{required} is not listed in files")
    return checked


def _require_actor(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ProjectFormatError(f"{label} must be an object with app and version")
    app = value.get("app")
    version = value.get("version")
    if not isinstance(app, str) or not app.strip():
        raise ProjectFormatError(f"{label}.app is missing")
    if not isinstance(version, str) or not version.strip():
        raise ProjectFormatError(f"{label}.version is missing")


def _check_source(value: object) -> None:
    if not isinstance(value, dict):
        raise ProjectFormatError("source must be an object")
    kind = value.get("format")
    if not isinstance(kind, str) or not kind.strip():
        raise ProjectFormatError("source.format is missing")
    filename = value.get("filename")
    if filename is not None and (not isinstance(filename, str) or "/" in filename or "\\" in filename or not filename):
        raise ProjectFormatError("source.filename must be a file name")
    digest = value.get("sha256")
    if digest is not None and (not isinstance(digest, str) or _HEX64.fullmatch(digest) is None):
        raise ProjectFormatError("source.sha256 must be 64 lowercase hex")


def _creator(previous: dict[str, Any]) -> dict[str, str]:
    found = previous.get("created_by")
    if isinstance(found, dict) and isinstance(found.get("app"), str) and isinstance(found.get("version"), str):
        return {"app": found["app"], "version": found["version"]}
    app = previous.get("app")
    version = previous.get("app_version")
    if isinstance(app, str) and app.strip() and isinstance(version, str) and version.strip():
        return {"app": app, "version": version}
    return {"app": APP_NAME, "version": APP_VERSION}


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
    """Return the work folder recorded in a project, or one beside the project file.

    Args:
        project: Path of the ``.vxclarity`` file.
        resume: The project's ``resume.json`` object.

    Returns:
        The recorded directory when it exists, otherwise ``project`` plus ``.work``.
    """
    recorded = resume.get("work")
    if isinstance(recorded, str) and recorded and Path(recorded).is_dir():
        return Path(recorded)
    return project.with_name(project.name + ".work")


def resume_from_work(work: Path) -> dict[str, Any]:
    """Build the ``resume.json`` object from a work folder on disk.

    Args:
        work: Folder that holds ``plan.json`` and ``journal.jsonl``.

    Returns:
        Work path, ranges, cuts, settings digest, source, and journal rows.
    """
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


def _zip_stamp(created: str) -> tuple[int, int, int, int, int, int]:
    text = created.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second)


def _zip_info(name: str, *, stored: bool, created: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_zip_stamp(created))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.extra = b""
    info.flag_bits = 0
    return info


def _write_archive(path: Path, entries: list[tuple[str, bytes, bool]], created: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(partial, "w", allowZip64=False) as archive:
        for name, payload, stored in entries:
            archive.writestr(_zip_info(name, stored=stored, created=created), payload)
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
    video = fingerprint(source, project=path, full=full_hash)
    kind = source.suffix.lower().lstrip(".") or "video"
    suite_source: dict[str, Any] = {"format": kind, "filename": source.name}
    if video["sha256"]:
        suite_source["sha256"] = video["sha256"]
    clarity = old_manifest.get("x_clarity")
    x_clarity = dict(clarity) if isinstance(clarity, dict) else {}
    x_clarity["product"] = "vxclarity"
    x_clarity["modified"] = _stamp()
    x_clarity["video"] = video
    manifest = {key: value for key, value in old_manifest.items() if key not in {"format", "app", "app_version", "modified", "files"}}
    manifest.update(
        {
            "format_version": FORMAT_VERSION,
            "created_by": _creator(old_manifest),
            "generator": {"app": APP_NAME, "version": APP_VERSION},
            "created": created,
            "source": suite_source,
            "x_clarity": x_clarity,
        }
    )
    _require_supported(manifest["format_version"])
    _require_actor(manifest["created_by"], "created_by")
    _require_actor(manifest["generator"], "generator")
    _check_source(manifest["source"])
    settings_raw = _settings_document(settings, destination, old_settings)
    resume_raw = dict(old_resume)
    if resume is not None:
        resume_raw.update(resume)
    settings_bytes = _dump(settings_raw)
    resume_bytes = _dump(resume_raw)
    stored_thumb = thumb if thumb is not None else (previous.thumb if previous is not None else None)
    extras = dict(previous.extras) if previous is not None else {}
    files: dict[str, Any] = {
        _SETTINGS: _file_record(settings_bytes),
        _RESUME: _file_record(resume_bytes),
    }
    if stored_thumb:
        files[_THUMB] = _file_record(stored_thumb)
    for name in sorted(extras):
        if name in {_MIME_NAME, _MANIFEST, _SETTINGS, _RESUME, _THUMB}:
            continue
        files[name] = _file_record(extras[name])
    manifest["files"] = files
    entries: list[tuple[str, bytes, bool]] = [
        (_MIME_NAME, MIME.encode("ascii"), True),
        (_MANIFEST, _dump(manifest), False),
        (_SETTINGS, settings_bytes, False),
        (_RESUME, resume_bytes, False),
    ]
    if stored_thumb:
        entries.append((_THUMB, stored_thumb, False))
    for name in sorted(extras):
        if name in {_MIME_NAME, _MANIFEST, _SETTINGS, _RESUME, _THUMB}:
            continue
        entries.append((name, extras[name], False))
    _write_archive(path, entries, created)
    loaded = load_project(path)
    return loaded


def _dump(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")


def load_project(path: Path) -> Project:
    """Read and check a ``.vxclarity`` archive.

    Args:
        path: Project file.

    Returns:
        Manifest, settings, resume state, and any extra members.

    Raises:
        ProjectFormatError: When the file is missing, not a zip, or fails the v1 checks.
    """
    if not path.is_file():
        raise ProjectFormatError("the project file was not found")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ProjectFormatError("the project is not a zip file") from exc
    with archive:
        names = _check_archive(archive)
        budget = _Budget()
        payloads = {info.filename: _read_member(archive, info, budget) for info in archive.infolist()}
        found = payloads[_MIME_NAME]
        if found != MIME.encode("ascii"):
            raise ProjectFormatError("mimetype is not a Clarity project")
        manifest = _read_json_bytes(payloads[_MANIFEST], _MANIFEST)
        _require_supported(manifest.get("format_version"))
        _require_actor(manifest.get("created_by"), "created_by")
        if "generator" in manifest:
            _require_actor(manifest.get("generator"), "generator")
        if "source" in manifest:
            _check_source(manifest.get("source"))
        checked = _verify_files(payloads, manifest, names)
        settings_raw = _read_json_bytes(checked[_SETTINGS], _SETTINGS)
        resume = _read_json_bytes(checked[_RESUME], _RESUME)
        thumb = checked.get(_THUMB) if _THUMB in names else None
        known = {_MIME_NAME, _MANIFEST, _SETTINGS, _RESUME, _THUMB}
        extras = {name: payload for name, payload in checked.items() if name not in known}
    settings, destination = _settings_from(settings_raw)
    source_raw = video_fingerprint(manifest)
    source = resolve_source(path, source_raw) if source_raw else None
    return Project(
        path=path,
        manifest=manifest,
        settings_raw=settings_raw,
        resume=resume,
        settings=settings,
        destination=destination,
        source=source,
        thumb=thumb,
        extras=extras,
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
    """Return the per-user folder that stores ``recent.json``.

    ``VOXPRINT_CLARITY_STATE`` wins. Otherwise this is
    ``%LOCALAPPDATA%/Voxprint/Clarity`` on Windows and
    ``$XDG_DATA_HOME/voxprint/clarity`` (or ``~/.local/share/voxprint/clarity``) elsewhere.

    Returns:
        The directory path. It is not created here.
    """
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
    """Return the path of the recent-projects list.

    Args:
        folder: State directory. ``None`` uses ``state_dir``.

    Returns:
        ``folder / "recent.json"``.
    """
    return (folder if folder is not None else state_dir()) / "recent.json"


def load_recent(folder: Path | None = None) -> list[str]:
    """Read recent project paths.

    Args:
        folder: State directory. ``None`` uses ``state_dir``.

    Returns:
        Paths newest first, or an empty list when the file is missing or unreadable.
    """
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
    """Put ``path`` first in the recent list and keep at most eight entries.

    Args:
        path: Project file to remember.
        folder: State directory. ``None`` uses ``state_dir``.

    Returns:
        The updated path list.
    """
    text = str(path.resolve())
    paths = [item for item in load_recent(folder) if item != text]
    paths.insert(0, text)
    paths = paths[:RECENT_LIMIT]
    target = recent_file(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"paths": paths}, indent=1) + "\n", encoding="utf-8")
    return paths
