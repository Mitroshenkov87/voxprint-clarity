"""The CLI reference is rendered from the live parsers."""
from __future__ import annotations

from scripts.mkdocs_hooks import render_api_page, render_cli_reference


def test_cli_reference_lists_parser_flags() -> None:
    text = render_cli_reference()
    assert "Generated from `clarity.cli.build_parser`" in text
    assert "### `clarity restore`" in text
    assert "`--input`" in text
    assert "`--output`" in text
    assert "`--dry-run`" in text
    assert "`--json`" in text
    assert "### `clarity project resume`" in text
    assert "### `clarity models fetch`" in text


def test_api_page_lists_public_modules() -> None:
    text = render_api_page()
    assert "::: clarity.pipeline.runner" in text
    assert "::: clarity.cli" in text
    assert "::: main" in text
