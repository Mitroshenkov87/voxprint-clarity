"""Progress text for a long TensorRT engine build.

The engine says ``building the TensorRT engine`` or ``loading the cached
TensorRT engine``. Those become the stage names ``building_engine`` and
``loading_cached_engine`` on a ``--json`` progress line, and the same words
on the window status line. The first build can take minutes.
"""
from __future__ import annotations

from clarity.i18n import languages, tr


def progress_stage(message: str) -> str | None:
    """Return the engine-build stage named by ``message``.

    Args:
        message: A progress line from Clarity or from the engine worker.

    Returns:
        ``building_engine``, ``loading_cached_engine``, or None.
    """
    text = message.strip().lower()
    if "building the tensorrt engine" in text:
        return "building_engine"
    if "loading the cached tensorrt engine" in text or "tensorrt engine is cached" in text:
        return "loading_cached_engine"
    for language in languages():
        if text == tr("status.building_engine", language).strip().lower():
            return "building_engine"
        if text == tr("status.loading_cached_engine", language).strip().lower():
            return "loading_cached_engine"
    return None
