"""Per-chunk resume. A finished chunk is renamed into place; the journal records it."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    """One finished scene chunk: its index, frame range, and file digest."""

    index: int
    start: int
    stop: int
    sha256: str


def settings_digest(payload: dict[str, Any]) -> str:
    """Hash the canonical JSON of a settings payload.

    Args:
        payload: Usually ``Settings.digest_payload``.

    Returns:
        Lowercase SHA-256 hex.
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    """Replace ``path`` only after the new JSON is fully written.

    Args:
        path: Destination. Parent directories are created.
        payload: Any object ``json.dumps`` accepts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    """Write ``plan.json`` atomically.

    Args:
        path: Usually ``work/plan.json``.
        plan: Probe, ranges, settings digest, and command lists.
    """
    write_json_atomic(path, plan)


def append_journal(path: Path, record: ChunkRecord) -> None:
    """Append one finished chunk as a JSON line.

    Args:
        path: Usually ``work/journal.jsonl``.
        record: Index, frame range, and SHA-256 of the published chunk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"index": record.index, "start": record.start, "stop": record.stop, "sha256": record.sha256},
        sort_keys=True,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_done(path: Path) -> dict[int, ChunkRecord]:
    """Completed chunks. A torn last line is ignored."""
    if not path.is_file():
        return {}
    done: dict[int, ChunkRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        try:
            record = ChunkRecord(
                index=int(data["index"]),
                start=int(data["start"]),
                stop=int(data["stop"]),
                sha256=str(data["sha256"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        done[record.index] = record
    return done


def pending(ranges: list[tuple[int, int]], done: dict[int, ChunkRecord]) -> list[tuple[int, int, int]]:
    """``(index, start, stop)`` for chunks that are not in the journal."""
    work: list[tuple[int, int, int]] = []
    for index, (start, stop) in enumerate(ranges):
        record = done.get(index)
        if record is not None and record.start == start and record.stop == stop and record.sha256:
            continue
        work.append((index, start, stop))
    return work


def chunk_path(folder: Path, index: int) -> Path:
    """Return the published path for one chunk.

    Args:
        folder: Directory that holds ``00000.mkv`` and the rest.
        index: Zero-based chunk index.

    Returns:
        ``folder / "{index:05d}.mkv"``.
    """
    return folder / f"{index:05d}.mkv"


def publish_chunk(partial: Path, final: Path) -> None:
    """Publish a finished chunk. Callers write ``partial`` first."""
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)


def file_sha256(path: Path) -> str:
    """Hash a file in 1 MiB reads.

    Args:
        path: File to read.

    Returns:
        Lowercase SHA-256 hex.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def load_plan(path: Path) -> dict[str, Any] | None:
    """Read a plan file.

    Args:
        path: Usually ``work/plan.json``.

    Returns:
        The object, or ``None`` when the file is missing or not a JSON object.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def plan_matches(plan: dict[str, Any], *, digest: str, source: str, ranges: list[tuple[int, int]]) -> bool:
    """Return whether a saved plan still describes this source and these settings.

    Args:
        plan: Object from ``load_plan``.
        digest: Current settings digest.
        source: Current source path, as stored in the plan.
        ranges: Current ``(start, stop)`` pairs.

    Returns:
        True when digest, source, and ranges all match.
    """
    stored = plan.get("ranges")
    expected = [list(pair) for pair in ranges]
    return plan.get("settings_sha256") == digest and plan.get("source") == source and stored == expected


def reset_chunks(folder: Path) -> None:
    """Drop a stale journal and chunk files. The work folder itself stays."""
    (folder / "journal.jsonl").unlink(missing_ok=True)
    chunks = folder / "chunks"
    if not chunks.is_dir():
        return
    for path in chunks.iterdir():
        if path.is_file() and (path.suffix == ".mkv" or path.name.endswith(".partial")):
            path.unlink()
