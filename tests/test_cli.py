"""Headless commands and the codename registry. No GPU."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from voxprint_common.cli_protocol import EXIT_CANCELLED, EXIT_FAILED, EXIT_OK, EXIT_USAGE

from clarity.appinfo import build_info
from clarity.cli import ExitCode, main, protocol_code
from clarity.errors import Cancelled
from clarity.codenames import check, foreign_names, parse_registry
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.probe import parse_probe
from clarity.pipeline.job import Settings
from clarity.pipeline.runner import JobSpec, dry_run_restore
from clarity.project import save_project

_ROOT = Path(__file__).resolve().parents[1]
_PROBE = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 8,
            "height": 8,
            "avg_frame_rate": "1/1",
            "nb_frames": "4",
            "color_transfer": "bt709",
            "field_order": "progressive",
        }
    ]
}
_ENCODERS = " V..... av1_nvenc             NVIDIA NVENC AV1 encoder\n"
_TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".iss", ".ps1", ".sh", ".toml", ".txt", ".svg", ".desktop", ".xml"}
_SKIP = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "site"}


def _fake_tool(argv: list[str]) -> str:
    if argv[0] == "ffprobe":
        return json.dumps(_PROBE)
    if any("gt(scene" in part for part in argv):
        return "frame:0    pts:1    pts_time:1\nlavfi.scene_score=0.8\n"
    return _ENCODERS


def _result(lines: str) -> dict[str, object]:
    parsed = [json.loads(line) for line in lines.splitlines() if line.strip()]
    assert parsed
    last = parsed[-1]
    assert isinstance(last, dict)
    return last


def test_probe_uses_duration_when_the_container_omits_a_frame_count() -> None:
    probe = parse_probe(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 16,
                    "height": 16,
                    "avg_frame_rate": "1/1",
                    "color_transfer": "bt709",
                }
            ],
            "format": {"duration": "2.0"},
        }
    )
    assert probe.frames == 2


def test_codename_is_listed_only_for_this_app() -> None:
    assert check(_ROOT)
    text = (_ROOT / "docs" / "RELEASES-CODENAMES.md").read_text(encoding="utf-8")
    used, reserved = parse_registry(text)
    name = json.loads((_ROOT / "BUILD.json").read_text(encoding="utf-8"))["codename"]
    assert name in used["clarity"]
    assert name not in foreign_names(used, reserved)
    assert "Torah" in text and "Tanakh" in text


def test_codename_and_the_forbidden_word_are_not_hardcoded() -> None:
    name = str(json.loads((_ROOT / "BUILD.json").read_text(encoding="utf-8"))["codename"])
    allowed = {Path("BUILD.json"), Path("docs") / "RELEASES-CODENAMES.md"}
    # The two halves are joined so this source file stays free of the blocked word.
    banned = "Bibl" + "ical"
    for path in _ROOT.rglob("*"):
        if not path.is_file() or _SKIP.intersection(path.parts):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(_ROOT)
        assert banned not in text, rel
        if rel not in allowed:
            assert name not in text, rel


def test_restore_dry_run_json_does_not_touch_the_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    monkeypatch.setattr("clarity.pipeline.runner._run_text", _fake_tool)

    def boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run called the GPU path")

    monkeypatch.setattr("clarity.pipeline.runner.live_control", boom)
    code = main(
        [
            "restore",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.mkv"),
            "--mode",
            "strong",
            "--upscale",
            "--format",
            "mp4",
            "--dry-run",
            "--json",
        ]
    )
    assert code == int(ExitCode.OK)
    captured = capsys.readouterr()
    assert captured.err == ""
    result = _result(captured.out)
    assert result["ok"] is True
    assert result["dry_run"] is True
    plan = result["plan"]
    assert isinstance(plan, dict)
    assert plan["gpu_used"] is False
    assert plan["executed"] is False
    assert plan["encoder"] == "av1_nvenc"
    assert plan["container"] == "mp4"
    assert plan["upscale"] is True
    chunks = plan["chunks"]
    assert isinstance(chunks, list) and chunks
    decode = chunks[0]["decode"]
    encode = chunks[0]["encode"]
    assert isinstance(decode, list) and "-hwaccel" in decode and "cuda" in decode
    assert isinstance(encode, list) and "av1_nvenc" in encode
    video = encode.index("-c:v")
    assert encode[encode.index("-pix_fmt", video) + 1] == "yuv420p"
    assert "concat" in plan and isinstance(plan["concat"], list)
    assert not (tmp_path / "out.mkv").exists()


def test_dry_run_plan_matches_the_job_helper(tmp_path: Path) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    spec = JobSpec(
        source=source,
        destination=tmp_path / "out.mkv",
        settings=Settings(mode="clean", container=Container.MKV, codec=VideoCodec.AV1),
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )
    plan = dry_run_restore(spec, run_text=_fake_tool)
    assert plan["models"][0] == "drunet-deblocking-color"
    assert plan["chunks"][0]["decode"][0] == "ffmpeg"


def test_project_resume_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    project = save_project(
        tmp_path / "clip.vxclarity",
        source=source,
        settings=Settings(mode="clean"),
        destination=tmp_path / "out.mkv",
    )
    monkeypatch.setattr("clarity.pipeline.runner._run_text", _fake_tool)
    code = main(["project", "resume", str(project.path), "--dry-run", "--json"])
    assert code == 0
    result = _result(capsys.readouterr().out)
    plan = result["plan"]
    assert isinstance(plan, dict)
    assert plan["source"].endswith("in.mkv")
    opened = main(["project", "open", str(project.path), "--json"])
    assert opened == 0
    body = _result(capsys.readouterr().out)
    assert body["mode"] == "clean"


def test_version_models_and_selftest_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version", "--json"]) == 0
    info = _result(capsys.readouterr().out)
    data = json.loads((_ROOT / "BUILD.json").read_text(encoding="utf-8"))
    assert data["version"] == "1.0.0"
    assert info["version"] == "1.0.0"
    assert info["label"].startswith("1.0.0 build ")
    assert "RC" not in info["label"]
    assert info["build"] == build_info().build
    assert info["codename"] == data["codename"]
    assert main(["--json", "models", "list"]) == 0
    listed = _result(capsys.readouterr().out)
    models = listed["models"]
    assert isinstance(models, list) and models
    assert main(["models", "fetch", "--dry-run", "--json", "--id", "drunet-color"]) == 0
    fetched = _result(capsys.readouterr().out)
    rows = fetched["models"]
    assert isinstance(rows, list) and rows[0]["id"] == "drunet-color"
    assert "path" not in rows[0]
    assert main(["selftest", "--dry-run", "--json"]) == 0
    check_body = _result(capsys.readouterr().out)
    assert check_body["gpu_used"] is False


def test_protocol_code_keeps_clarity_numbers() -> None:
    assert [protocol_code(code) for code in ExitCode] == [0, 1, 2, 3]
    assert protocol_code(ExitCode.OK) == EXIT_OK == int(ExitCode.OK)
    assert protocol_code(ExitCode.USAGE) == EXIT_USAGE == int(ExitCode.USAGE)
    assert protocol_code(ExitCode.ENVIRONMENT) == EXIT_FAILED == int(ExitCode.ENVIRONMENT)
    assert protocol_code(ExitCode.JOB) == EXIT_CANCELLED == int(ExitCode.JOB)
    assert ExitCode.JOB.name == "JOB"


def test_cancel_stays_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    monkeypatch.setattr("clarity.cli._gates", lambda _console: None)

    def stop(*_args: object, **_kwargs: object) -> None:
        raise Cancelled("cancelled")

    monkeypatch.setattr("clarity.cli.run_job", stop)
    code = main(["restore", "--input", str(source), "--output", str(tmp_path / "out.mkv"), "--json"])
    assert code == int(ExitCode.USAGE)
    assert code != int(ExitCode.JOB)
    result = _result(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["code"] == 1
    assert result["error"] == "cancelled"


def test_missing_input_is_usage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "restore",
            "--input",
            str(tmp_path / "missing.mkv"),
            "--output",
            str(tmp_path / "out.mkv"),
            "--dry-run",
            "--json",
        ]
    )
    assert code == int(ExitCode.USAGE)
    result = _result(capsys.readouterr().out)
    assert result["ok"] is False


def test_cli_dry_run_with_ffmpeg(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        if os.environ.get("CLARITY_REQUIRE_FFMPEG") == "1":
            pytest.fail("ffmpeg and ffprobe are required in this CI job")
        pytest.skip("ffmpeg is not installed")
    source = tmp_path / "clip.mkv"
    encoded = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=16x16:rate=1:duration=2",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "ffv1",
            str(source),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert encoded.returncode == 0, encoded.stderr
    destination = tmp_path / "out.mkv"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "clarity",
            "restore",
            "--input",
            str(source),
            "--output",
            str(destination),
            "--mode",
            "clean",
            "--format",
            "mkv",
            "--ffmpeg",
            ffmpeg,
            "--ffprobe",
            ffprobe,
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    result = _result(completed.stdout)
    plan = result["plan"]
    assert isinstance(plan, dict)
    assert plan["gpu_used"] is False
    assert plan["frames"] >= 1
    assert not destination.exists()
