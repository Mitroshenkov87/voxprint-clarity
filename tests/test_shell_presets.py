"""Preset mapping, Expert mode, and the placeholder catalogue."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clarity.models import CATALOG
from clarity.pipeline.encode import Container, VideoCodec, encode_chunk_command
from clarity.pipeline.job import Settings
from clarity.pipeline.job import models_for
from clarity.ui.job_presets import (
    adopted_video,
    apply_job,
    fresh_state,
    job_from_state,
    load_clarity_manifest,
    model_choices,
    models_ready,
)
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.presets import PresetMatch

_UI = Path(__file__).resolve().parents[1] / "clarity" / "ui"
_PRESETS = _UI / "presets" / "clarity.json"


def test_defaults_match_the_job() -> None:
    assert job_from_state(fresh_state().values) == Settings()


def test_preset_choices_pick_the_catalogue_models() -> None:
    state = fresh_state()
    state.choose("cleanup", "light")
    light = job_from_state(state.values)
    assert [model.id for model in models_for(light)] == [model.id for model in models_for(Settings())]
    state.choose("cleanup", "medium")
    medium = job_from_state(state.values)
    assert medium.mode == "clean"
    assert medium.sigma == 25
    assert [model.id for model in models_for(medium)] == [model.id for model in models_for(Settings())]
    state.choose("cleanup", "strong")
    state.choose("size", "2x")
    state.choose("look", "anime")
    state.choose("fix", "plain")
    state.choose("sharpness", "sharper")
    state.choose("quality", "best")
    state.choose("file", "mp4")
    job = job_from_state(state.values)
    expect = Settings(
        mode="strong",
        upscale=True,
        anime=True,
        sharpen=0.6,
        white_balance=False,
        levels=False,
        container=Container.MP4,
        crf=16,
    )
    assert job == expect
    assert job.sigma == 15
    assert [model.id for model in models_for(job)] == [model.id for model in models_for(expect)]
    assert state.values["job.upscale_model"] == models_for(expect)[-1].id


def test_file_quality_is_the_av1_crf_and_balanced_is_the_old_default() -> None:
    state = fresh_state()
    assert job_from_state(state.values).crf == 20
    state.choose("quality", "smaller")
    smaller = job_from_state(state.values)
    assert smaller.crf == 32
    assert smaller.quality_factor == 40
    command = encode_chunk_command(
        "ffmpeg",
        encoder="av1_nvenc",
        destination="chunk.mkv",
        width=16,
        height=16,
        fps=24,
        cq=smaller.crf,
    )
    assert command[command.index("-cq") + 1] == "32"
    state.choose("quality", "balanced")
    assert state.status("quality").kind is PresetMatch.EXACT
    state.set_value("job.crf", 22)
    assert state.status("quality").kind is PresetMatch.CUSTOM
    state.reset_group("quality")
    assert state.values["job.crf"] == 20
    assert state.status("quality").kind is PresetMatch.EXACT


def test_colours_off_still_runs_noise_and_blocks() -> None:
    state = fresh_state()
    state.choose("fix", "plain")
    job = job_from_state(state.values)
    assert job.white_balance is False
    assert job.levels is False
    assert [model.id for model in models_for(job)] == [model.id for model in models_for(Settings())]
    state.set_value("job.white_balance", True)
    assert state.status("fix").kind is PresetMatch.MODIFIED
    state.set_value("job.levels", False)
    state.set_value("job.white_balance", False)
    state.set_value("job.levels", True)
    assert state.status("fix").kind is PresetMatch.MODIFIED


def test_cleanup_edit_is_modified_then_custom_and_reset_keeps_size() -> None:
    state = fresh_state()
    state.set_value("job.sharpen", 0.5)
    assert state.status("sharpness").kind is PresetMatch.CUSTOM
    assert state.status("cleanup").kind is PresetMatch.EXACT
    state.set_value("job.sigma", 20.0)
    assert state.status("cleanup").kind is PresetMatch.MODIFIED
    state.set_value("job.restore_models", "edited")
    assert state.status("cleanup").kind is PresetMatch.CUSTOM
    state.choose("size", "2x")
    state.reset_group("cleanup")
    assert state.status("cleanup").kind is PresetMatch.EXACT
    assert state.values["job.sigma"] == 15
    assert state.values["job.sharpen"] == 0.5
    assert state.values["job.upscale"] is True
    assert job_from_state(state.values).upscale is True
    state.reset_group("sharpness")
    assert state.values["job.sharpen"] == 0.35
    assert state.status("sharpness").kind is PresetMatch.EXACT


def test_expert_mode_does_not_drop_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    state = fresh_state(tmp_path / "state")
    state.set_value("job.sigma", 9.0)
    state.choose("look", "anime")
    snapshot = state.snapshot()
    mode = ExpertMode()
    assert mode.enabled() is False
    assert mode.set_enabled(True) is True
    assert mode.set_enabled(False) is False
    assert state.snapshot() == snapshot


def test_shared_video_sample_uses_h264_and_the_app_file_does_not() -> None:
    adopted = adopted_video()
    strength = adopted.shared_presets.group("strength")
    assert strength is not None
    gentle = strength.option("gentle")
    strong = strength.option("strong")
    assert gentle is not None and strong is not None
    assert gentle.as_map()["video.encoder"] == "h264"
    assert strong.as_map()["video.encoder"] == "h265"
    text = _PRESETS.read_text(encoding="utf-8").lower()
    for token in ("h264", "h265", "x264", "x265"):
        assert token not in text


def test_placeholder_manifest_covers_the_catalogue_and_stays_unavailable() -> None:
    manifest = load_clarity_manifest()
    assert manifest is not None
    assert model_choices(manifest) == ()
    assert models_ready() is True
    published = {model.id: model for model in manifest.models}
    for row in published.values():
        assert row.unavailable is True
        assert row.reason == "placeholder"
        assert row.files == ()
    jobs = (
        Settings(),
        Settings(mode="strong"),
        Settings(upscale=True),
        Settings(upscale=True, anime=True),
        Settings(upscale=True, upscale_strength=0.4),
    )
    used = {model.id for job in jobs for model in models_for(job)}
    assert used == set(published)
    assert {model.id for model in CATALOG} == set(published)
    document = json.loads((_UI.parents[1] / "mdl.json").read_text(encoding="utf-8"))
    assert document["schema"] == 1
    assert document["app"] == "clarity"


def test_ui_python_does_not_name_a_catalogue_id() -> None:
    names = {model.id for model in CATALOG}
    for path in _UI.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in names:
            assert name not in text


def test_anime_chip_waits_for_upscale() -> None:
    state = fresh_state()
    state.choose("look", "anime")
    held = job_from_state(state.values)
    assert state.values["job.anime"] is True
    assert held.anime is False
    assert held.upscale is False
    state.choose("size", "2x")
    running = job_from_state(state.values)
    assert running.anime is True
    assert running.upscale is True


def test_apply_job_round_trip() -> None:
    job = Settings(
        mode="strong",
        upscale=True,
        anime=False,
        upscale_strength=0.25,
        sharpen=0.2,
        white_balance=False,
        levels=False,
        quality_factor=22.0,
        sigma=3.0,
        tile=64,
        batch=4,
        overlap=8,
        container=Container.MP4,
        codec=VideoCodec.HEVC,
    )
    state = fresh_state()
    apply_job(state, job)
    assert job_from_state(state.values) == job
