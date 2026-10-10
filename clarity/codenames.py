"""Check ``BUILD.json`` against the vendored suite codename registry.

CI reads ``docs/RELEASES-CODENAMES.md``. That file is a copy, not a live
query of the other repositories. Before tagging, update the suite table and
this copy in the same commit as the ``BUILD.json`` codename.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from clarity.appinfo import repo_root

REGISTRY = Path("docs") / "RELEASES-CODENAMES.md"
THIS_APP = "clarity"


class CodenameError(RuntimeError):
    """The codename repeats another app, or the vendored table was not updated."""


def parse_registry(text: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Used names by app, and the reserved next name by app.

    Only the markdown tables are read. Prose that mentions a name does not
    count as a claim on it.
    """
    section = ""
    used: dict[str, list[str]] = {}
    reserved: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("## Used"):
            section = "used"
            continue
        if line.startswith("## Assigned"):
            section = "reserved"
            continue
        if line.startswith("## "):
            section = ""
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"App", ""} or set(cells[0]) <= {"-", ":"}:
            continue
        if section == "used" and len(cells) >= 2:
            used[cells[0]] = [part.strip() for part in cells[1].split(",") if part.strip()]
        elif section == "reserved" and len(cells) >= 2 and cells[1]:
            reserved[cells[0]] = cells[1]
    return used, reserved


def foreign_names(used: dict[str, list[str]], reserved: dict[str, str], app: str = THIS_APP) -> set[str]:
    """Names already taken by another app, including names reserved for one."""
    taken: set[str] = set()
    for other, names in used.items():
        if other != app:
            taken.update(names)
    for other, name in reserved.items():
        if other != app and name:
            taken.add(name)
    return taken


def check(root: Path | None = None) -> str:
    """Return this app's codename, or raise ``CodenameError``."""
    base = repo_root() if root is None else root
    data = json.loads((base / "BUILD.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CodenameError("BUILD.json must be an object")
    codename = data.get("codename")
    if not isinstance(codename, str) or not codename.strip():
        raise CodenameError("BUILD.json codename must be a non-empty string")
    codename = codename.strip()
    used, reserved = parse_registry((base / REGISTRY).read_text(encoding="utf-8"))
    owned = used.get(THIS_APP, [])
    if codename not in owned:
        raise CodenameError(
            f"{codename} is not listed for {THIS_APP} in {REGISTRY.as_posix()}. "
            "Add it to that vendored table in the same commit as BUILD.json. "
            "CI does not query the other repositories."
        )
    taken = foreign_names(used, reserved)
    if codename in taken:
        raise CodenameError(f"{codename} is already used or reserved by another suite app")
    return codename


def main() -> int:
    """Check this checkout's codename and print the result.

    Returns:
        ``0`` when the name is free for Clarity, ``1`` when the check fails.
    """
    try:
        name = check()
    except (CodenameError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"codename ok ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
