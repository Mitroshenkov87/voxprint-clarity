"""The feature map names every parser flag and records an owner decision."""
from __future__ import annotations

import argparse
from pathlib import Path

from clarity.cli import _command_parser, build_parser

_MAP = Path(__file__).resolve().parents[1] / "docs" / "ui" / "FEATURE-MAP.md"


def _names(parser: argparse.ArgumentParser) -> set[str]:
    found: set[str] = set()
    for action in parser._actions:
        if action.option_strings:
            found.update(action.option_strings)
        elif not isinstance(action, argparse._SubParsersAction) and action.dest != argparse.SUPPRESS:
            found.add(action.dest)
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                found.add(name)
                found.update(_names(child))
    return found


def test_feature_map_covers_the_cli_and_keeps_the_owner_decision() -> None:
    text = _MAP.read_text(encoding="utf-8")
    required = _names(build_parser()) | _names(_command_parser())
    missing = sorted(name for name in required if name not in text)
    assert missing == []
    assert "needs owner decision" in text
    assert "| kept |" in text
