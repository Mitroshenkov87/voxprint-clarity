"""MkDocs hook. Writes the CLI reference and the API page before each build."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set before PySide6 loads. A docs build has no display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from clarity.cli import _command_parser, build_parser

_ROOT = Path(__file__).resolve().parents[1]


def on_pre_build(config: object) -> None:
    """Write generated pages into the docs tree before MkDocs collects files.

    Args:
        config: MkDocs config. The hook reads ``docs_dir``.
    """
    docs_dir = Path(getattr(config, "docs_dir"))
    write_generated(docs_dir)


def write_generated(docs_dir: Path) -> None:
    """Write the CLI reference, the API page, and a copy of the roadmap.

    Args:
        docs_dir: The MkDocs ``docs_dir``.
    """
    reference = docs_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "cli.md").write_text(render_cli_reference(), encoding="utf-8")
    (reference / "api.md").write_text(render_api_page(), encoding="utf-8")
    roadmap = (_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    (docs_dir / "roadmap.md").write_text(roadmap, encoding="utf-8")


def render_cli_reference() -> str:
    """Render the window launcher and the headless commands from the live parsers.

    Returns:
        Markdown for ``docs/reference/cli.md``.
    """
    lines = [
        "# Command reference",
        "",
        "Generated from `clarity.cli.build_parser` and `clarity.cli._command_parser`",
        "when this site is built. Change the parsers to change this page.",
        "",
        "The narrative is in the [command line](../CLI.md) guide.",
        "",
        "## Window launcher",
        "",
        _parser_section(build_parser()),
        "",
        "## Headless commands",
        "",
    ]
    command = _command_parser()
    lines.append(_parser_section(command))
    for _name, sub in _subparsers(command):
        lines.extend(["", _parser_section(sub)])
        for _nested, leaf in _subparsers(sub):
            lines.extend(["", _parser_section(leaf)])
    lines.append("")
    return "\n".join(lines)


def render_api_page() -> str:
    """Render mkdocstrings directives for the public modules.

    Returns:
        Markdown for ``docs/reference/api.md``.
    """
    lines = [
        "# API reference",
        "",
        "Generated from the `clarity` package and `main` when this site is built.",
        "Docstrings use the Google convention.",
        "",
    ]
    for name in _public_modules():
        lines.extend(
            [
                f"::: {name}",
                "    options:",
                "      show_root_heading: true",
                "",
            ]
        )
    return "\n".join(lines)


def _public_modules() -> list[str]:
    names: list[str] = []
    for path in sorted((_ROOT / "clarity").rglob("*.py")):
        relative = path.relative_to(_ROOT)
        if path.name == "__init__.py":
            module = ".".join(relative.parent.parts)
        else:
            module = ".".join(relative.with_suffix("").parts)
        names.append(module)
    names.append("main")
    return names


def _subparsers(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.ArgumentParser]]:
    found: list[tuple[str, argparse.ArgumentParser]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                if isinstance(sub, argparse.ArgumentParser):
                    found.append((str(name), sub))
    return found


def _parser_section(parser: argparse.ArgumentParser) -> str:
    description = (parser.description or "").strip()
    lines = [f"### `{parser.prog}`", ""]
    if description:
        lines.extend([description, ""])
    lines.extend(
        [
            "| Argument | Required | Choices | Default | Help |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for action in parser._actions:
        row = _argument_row(action)
        if row is not None:
            lines.append(row)
    nested = _subparsers(parser)
    if nested:
        names = ", ".join(f"`{name}`" for name, _sub in nested)
        lines.extend(["", f"Subcommands: {names}."])
    return "\n".join(lines)


def _argument_row(action: argparse.Action) -> str | None:
    if isinstance(action, argparse._SubParsersAction):
        return None
    if action.help is argparse.SUPPRESS:
        return None
    if action.option_strings:
        label = ", ".join(f"`{flag}`" for flag in action.option_strings)
        required = "yes" if action.required else "no"
    else:
        label = f"`{action.metavar or action.dest}`"
        required = "no" if action.nargs in {"?", "*"} else "yes"
    choices = ""
    if action.choices is not None:
        choices = ", ".join(f"`{item}`" for item in action.choices)
    default = _default(action)
    help_text = " ".join(str(action.help or "").split())
    cells = [_cell(label), _cell(required), _cell(choices), _cell(default), _cell(help_text)]
    return "| " + " | ".join(cells) + " |"


def _default(action: argparse.Action) -> str:
    if action.default is argparse.SUPPRESS or action.default is None:
        return ""
    if isinstance(action.default, bool):
        return "true" if action.default else "false"
    return str(action.default)


def _cell(value: str) -> str:
    return value.replace("|", "\\|")
