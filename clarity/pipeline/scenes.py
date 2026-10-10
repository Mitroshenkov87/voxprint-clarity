"""Scene-cut chunking. State resets at a cut. Long scenes split at ``max_frames``."""
from __future__ import annotations


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
