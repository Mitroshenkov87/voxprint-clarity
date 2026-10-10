"""UI strings. English is the source. Russian is the translation."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_LOCALES = Path(__file__).resolve().parents[1] / "locales"


@lru_cache(maxsize=4)
def _catalog(language: str) -> dict[str, str]:
    path = _LOCALES / f"{language}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be an object")
    return {str(key): str(value) for key, value in data.items()}


def languages() -> tuple[str, ...]:
    return ("en", "ru")


def current_language() -> str:
    value = os.environ.get("VOXPRINT_LANG", "en").strip().lower()
    if value.startswith("ru"):
        return "ru"
    return "en"


def tr(key: str, language: str | None = None) -> str:
    """Return the string for ``key``. A missing key is the key itself."""
    lang = language or current_language()
    catalog = _catalog(lang)
    if key in catalog:
        return catalog[key]
    if lang != "en":
        return _catalog("en").get(key, key)
    return key


def clear_cache() -> None:
    _catalog.cache_clear()
