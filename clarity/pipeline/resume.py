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
    index: int
    start: int
    stop: int
    sha256: str


def settings_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    write_json_atomic(path, plan)


def append_journal(path: Path, record: ChunkRecord) -> None:
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
    return folder / f"{index:05d}.mkv"


def publish_chunk(partial: Path, final: Path) -> None:
    """Publish a finished chunk. Callers write ``partial`` first."""
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
