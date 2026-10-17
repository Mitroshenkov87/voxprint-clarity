"""mdl.json must publish exactly the weights clarity/models.py pins."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from voxprint_common.gpu_probe import VramTier
from voxprint_common.mdl.catalog import load_manifest, plan_files

from clarity.models import CATALOG
from clarity.ui.job_presets import models_ready

_ROOT = Path(__file__).resolve().parents[1]
_MDL = _ROOT / "mdl.json"
_REQUIRED = {"drunet-deblocking-color", "drunet-color"}
_AUTHORS = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/",
)
_MIRROR = re.compile(
    r"^https://huggingface\.co/Mitroshenkov87/voxprint-mirror-(?P<id>[a-z0-9-]+)"
    r"/resolve/(?P<rev>[0-9a-f]{40})/(?P<file>[^/]+)$"
)


def _document() -> dict:
    return json.loads(_MDL.read_text(encoding="utf-8"))


def test_ids_match_the_catalogue_in_order() -> None:
    assert [row["id"] for row in _document()["models"]] == [model.id for model in CATALOG]


def test_files_sizes_hashes_and_licences_match_the_catalogue() -> None:
    rows = {row["id"]: row for row in _document()["models"]}
    for model in CATALOG:
        row = rows[model.id]
        assert row["version"] == "1"
        assert row["vram_tier"] == "any"
        assert row["licence"] == model.licence
        assert "status" not in row and "post_install" not in row
        (item,) = row["files"]
        assert item["path"] == model.filename
        assert item["size"] == model.size
        assert item["sha256"] == model.sha256


def test_install_modes() -> None:
    for row in _document()["models"]:
        assert row["on_demand"] is (row["id"] not in _REQUIRED)


def test_author_release_first_then_a_commit_pinned_mirror() -> None:
    rows = {row["id"]: row for row in _document()["models"]}
    for model in CATALOG:
        author, mirror = rows[model.id]["files"][0]["mirrors"]
        assert author == model.download_url
        assert author.startswith(_AUTHORS)
        match = _MIRROR.fullmatch(mirror)
        assert match is not None, mirror
        assert match["id"] == model.id
        assert match["file"] == model.filename


def test_shared_parser_accepts_the_manifest() -> None:
    manifest = load_manifest(_MDL)
    assert manifest.app == "clarity"
    planned = plan_files([manifest], VramTier.GB8, models_root=Path("models"))
    assert {item.sha256 for item in planned} == {
        model.sha256 for model in CATALOG if model.id in _REQUIRED
    }


def test_ready_needs_only_the_required_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    assert models_ready() is False
    from voxprint_common.paths import models_dir

    for item in plan_files([load_manifest(_MDL)], VramTier.GB8, models_root=models_dir()):
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        with item.destination.open("wb") as handle:
            handle.truncate(item.size)
    assert models_ready() is True
