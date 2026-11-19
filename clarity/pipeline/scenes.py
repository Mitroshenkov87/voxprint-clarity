"""Scene-cut chunking. State resets at a cut. Long scenes split at ``max_frames``."""
from __future__ import annotations

import re

_PTS_TIME = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_SCENE_SCORE = re.compile(r"lavfi\.scene_score=([0-9]+(?:\.[0-9]+)?)")


def scene_command(ffmpeg: str, source: str, threshold: float = 0.4) -> list[str]:
    """ffmpeg argv that prints a scene score and ``pts_time`` for each cut.

    ``metadata=print:file=-`` writes the selected frames to stdout. The first
    frame is not a cut. ``threshold`` is ffmpeg's scene probability, 0..1.
    """
    select = f"select='gt(scene,{threshold:.4f})',metadata=print:file=-"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source,
        "-an",
        "-vf",
        select,
        "-f",
        "null",
        "-",
    ]


def parse_scene_cuts(text: str, fps: float) -> list[int]:
    """Frame indexes where a new scene begins. Empty when ``fps`` is not positive."""
    if fps <= 0:
        return []
    pending_time: float | None = None
    cuts: list[int] = []
    for line in text.splitlines():
        found = _PTS_TIME.search(line)
        if found:
            pending_time = float(found.group(1))
            continue
        if _SCENE_SCORE.search(line) and pending_time is not None:
            frame = int(round(pending_time * fps))
            if frame > 0:
                cuts.append(frame)
            pending_time = None
    return sorted(set(cuts))


def plan_chunks(frame_count: int, cuts: list[int] | tuple[int, ...], max_frames: int = 1000) -> list[tuple[int, int]]:
    """Half-open frame ranges ``[start, stop)``.

    ``cuts`` are frame indexes where a new scene begins. Indexes outside the
    clip are ignored. A scene longer than ``max_frames`` is split so a crash
    can resume a piece instead of the whole shot.
    """
    count = int(frame_count)
    if count < 1:
        raise ValueError("frame_count must be positive")
    limit = int(max_frames)
    if limit < 1:
        raise ValueError("max_frames must be positive")
    bounds = [0]
    for cut in cuts:
        index = int(cut)
        if 0 < index < count:
            bounds.append(index)
    bounds.append(count)
    bounds = sorted(set(bounds))
    ranges: list[tuple[int, int]] = []
    for start, stop in zip(bounds, bounds[1:]):
        cursor = start
        while cursor < stop:
            ranges.append((cursor, min(stop, cursor + limit)))
            cursor += limit
    return ranges
