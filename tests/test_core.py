"""Pipeline, catalogue, theme, and the pending engine video API. No GPU."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from clarity.appinfo import build_info, release_title, version_label
from clarity.bench import Sample, main as bench_main, render_report, save_crops, tensorrt_providers, write_report
from clarity.bench import conditioned_nchw, network_module
from clarity.engine import assess_launch, gpu_busy_text, submit_restore
from clarity.errors import Cancelled
from clarity.i18n import tr
from clarity.models import CATALOG, get_model, licence_allowed
from clarity.pipeline.colour import grey_world
from clarity.pipeline.condition import deblock_level, noise_map, sigma_level
from clarity.pipeline.encode import (
    Container,
    VideoCodec,
    assert_encoder_allowed,
    audio_args,
    decode_command,
    encode_command,
    parse_encoder_list,
    select_encoder,
)
from clarity.pipeline.job import Hooks, Settings, commands_for, process_frames, restore_ranges
from clarity.pipeline.probe import parse_probe, require_sdr
from clarity.pipeline.resume import append_journal, chunk_path, load_done, pending, publish_chunk, write_plan
from clarity.pipeline.resume import ChunkRecord
from clarity.pipeline.scenes import parse_scene_cuts, plan_chunks
from clarity.pipeline.sharpen import sharpen_frames
from clarity.platform_gate import assess_platform, linux_supported, windows_supported
from clarity.ui import theme


def test_h264_is_refused_and_av1_is_default() -> None:
    with pytest.raises(Exception):
        assert_encoder_allowed("libx264")
    with pytest.raises(Exception):
        assert_encoder_allowed("h264_nvenc")
    with pytest.raises(Exception):
        assert_encoder_allowed("libx265")
    assert select_encoder(VideoCodec.AV1, {"av1_nvenc", "libx264"}) == "av1_nvenc"
    assert select_encoder(VideoCodec.AV1, {"libsvtav1"}) == "libsvtav1"
    with pytest.raises(Exception):
        select_encoder(VideoCodec.AV1, {"libx264"})
    assert select_encoder(VideoCodec.HEVC, {"hevc_nvenc"}) == "hevc_nvenc"
    with pytest.raises(Exception):
        select_encoder(VideoCodec.HEVC, {"libx265"})
    command = encode_command(
        "ffmpeg",
        "in.mkv",
        container=Container.MKV,
        encoder="av1_nvenc",
        destination="out.mkv",
        copy_audio=True,
        width=1920,
        height=1080,
        fps=24,
    )
    text = " ".join(command)
    video_at = command.index("-c:v")
    assert command[command.index("-pix_fmt", video_at) + 1] == "yuv420p"
    assert "av1_nvenc" in text
    assert "libx264" not in text
    assert "h264" not in text
    assert audio_args(copy=True) == ["-c:a", "copy"]
    assert audio_args(copy=False)[1] == "libopus"
    decode = decode_command("ffmpeg", "in.mkv", deinterlace=True)
    assert decode[decode.index("-hwaccel") + 1] == "cuda"
    assert "bwdif" in decode
    listed = parse_encoder_list(" V....D av1_nvenc           NVIDIA\n A....D libopus            Opus\n")
    assert "av1_nvenc" in listed
    assert "libopus" in listed


def test_hdr_is_refused() -> None:
    sdr = parse_probe(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "24000/1001",
                    "nb_frames": "48",
                    "color_transfer": "bt709",
                    "field_order": "progressive",
                },
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "subtitle", "codec_name": "subrip"},
            ]
        }
    )
    require_sdr(sdr)
    assert sdr.has_subtitles is True
    assert sdr.fps == pytest.approx(24 / 1.001)
    hdr = parse_probe(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 1920,
                    "height": 1080,
                    "color_transfer": "smpte2084",
                    "nb_frames": "10",
                }
            ]
        }
    )
    with pytest.raises(Exception, match="SDR"):
        require_sdr(hdr)


def test_scene_chunks_reset_at_cuts() -> None:
    assert plan_chunks(10, [4], max_frames=100) == [(0, 4), (4, 10)]
    assert plan_chunks(2500, [], max_frames=1000) == [(0, 1000), (1000, 2000), (2000, 2500)]
    text = "frame:0    pts:0    pts_time:0\nlavfi.scene_score=0.2\nframe:1    pts:24    pts_time:1.0\nlavfi.scene_score=0.9\n"
    assert parse_scene_cuts(text, 24.0) == [24]
    assert parse_scene_cuts(text, 0.0) == []


def test_colour_is_constant_across_the_chunk() -> None:
    frames = np.zeros((4, 2, 2, 3), dtype=np.float32)
    frames[..., 0] = 0.2
    frames[..., 1] = 0.4
    frames[..., 2] = 0.8
    balanced = grey_world(frames)
    means = balanced.reshape(-1, 3).mean(axis=0)
    assert np.allclose(means, means.mean())
    sharp = sharpen_frames(frames, 0)
    assert np.allclose(sharp, frames)
    assert deblock_level(40) == pytest.approx(0.6)
    assert sigma_level(15) == pytest.approx(15 / 255)


def test_resume_skips_finished_chunks(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    write_plan(plan, {"chunks": 2})
    journal = tmp_path / "journal.jsonl"
    append_journal(journal, ChunkRecord(0, 0, 10, "abc"))
    journal.write_text(journal.read_text(encoding="utf-8") + "{torn", encoding="utf-8")
    done = load_done(journal)
    assert pending([(0, 10), (10, 20)], done) == [(1, 10, 20)]
    partial = tmp_path / "chunks" / "00001.mkv.partial"
    partial.parent.mkdir()
    partial.write_bytes(b"ok")
    final = chunk_path(tmp_path / "chunks", 1)
    publish_chunk(partial, final)
    assert final.read_bytes() == b"ok"
    assert json.loads(plan.read_text(encoding="utf-8"))["chunks"] == 2


def _identity(model: object, nchw: np.ndarray) -> np.ndarray:
    rgb = nchw[:, :3]
    scale = int(getattr(model, "scale"))
    if scale == 1:
        return rgb
    return np.repeat(np.repeat(rgb, scale, axis=2), scale, axis=3)


def test_process_frames_streams_one_engine_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_iter(
        frames: np.ndarray,
        wrapped: Callable[[np.ndarray], np.ndarray],
        tile: int,
        overlap: int,
        *,
        scale: int,
        batch: int,
    ) -> Iterator[np.ndarray]:
        del tile, overlap, scale, batch
        seen.append(int(frames.shape[0]))
        count = int(frames.shape[0])
        for index in range(count):
            nchw = np.transpose(frames[index : index + 1], (0, 3, 1, 2))
            output = np.asarray(wrapped(nchw))
            yield np.transpose(output[0], (1, 2, 0))

    monkeypatch.setattr("clarity.pipeline.job.iter_tile_frames", fake_iter)
    frames = np.full((2, 8, 8, 3), 0.4, dtype=np.float32)
    settings = Settings(
        mode="strong",
        upscale=False,
        sharpen=0,
        white_balance=False,
        levels=False,
        tile=8,
        overlap=0,
        batch=2,
    )
    out = process_frames(frames, settings, _identity, tile=8, batch=2)
    assert out.shape == frames.shape
    assert seen == [2]


def test_process_frames_clean_and_optional_2x() -> None:
    pytest.importorskip("voxprint_engine")
    frames = np.full((2, 8, 8, 3), 0.4, dtype=np.float32)
    settings = Settings(
        mode="clean",
        upscale=False,
        sharpen=0,
        white_balance=False,
        levels=False,
        tile=8,
        overlap=0,
        batch=2,
    )
    out = process_frames(frames, settings, _identity, tile=8, batch=2)
    assert out.shape == frames.shape
    settings.upscale = True
    settings.anime = True
    up = process_frames(frames, settings, _identity, tile=8, batch=2)
    assert up.shape == (2, 16, 16, 3)


def test_chunk_loop_pauses_and_halves_on_spill() -> None:
    pytest.importorskip("voxprint_engine")
    frames = np.full((4, 8, 8, 3), 0.5, dtype=np.float32)
    settings = Settings(mode="strong", sharpen=0, white_balance=False, levels=False, tile=8, overlap=0, batch=4)
    calls = {"spill": 0, "sleep": 0}

    def decide(elapsed: float, median: float | None, throttle: float, paused: bool) -> object:
        del elapsed, median, throttle
        if calls["sleep"] == 0 and not paused:
            return type("D", (), {"action": "pause", "reason": "hot"})()
        return type("D", (), {"action": "resume" if paused else "run", "reason": "ok"})()

    def sleep(_seconds: float) -> None:
        calls["sleep"] += 1

    def note_spill(batch: int, spilled: bool) -> int:
        calls["spill"] += 1
        return max(1, batch // 2) if spilled else batch

    usage = {"n": 0}

    def shared() -> int:
        usage["n"] += 1
        return 0 if usage["n"] == 1 else 300_000_000

    hooks = Hooks(
        infer=_identity,
        decide=decide,
        sleep=sleep,
        note_spill=note_spill,
        halve_tile=lambda tile: tile // 2,
        shared_usage=shared,
        empty_cache=lambda: None,
        choose_tile=lambda *_args: 8,
    )
    records = restore_ranges([(0, 4)], {}, lambda _start, _stop: frames, settings, hooks)
    assert len(records) == 1
    assert calls["sleep"] == 1
    assert calls["spill"] == 1
    assert records[0].sha256


def test_commands_use_nvdec_and_refuse_hdr() -> None:
    probe = parse_probe(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "av1",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "24/1",
                    "nb_frames": "24",
                    "color_transfer": "bt709",
                    "field_order": "tt",
                }
            ]
        }
    )
    commands = commands_for(
        probe,
        Settings(upscale=True),
        ffmpeg="ffmpeg",
        source="in.mkv",
        destination="out.mkv",
        encoders={"av1_nvenc"},
    )
    assert "bwdif" in commands["decode"]
    encode = " ".join(commands["encode"])
    assert "av1_nvenc" in encode
    assert "-cq 20" in encode
    assert "-s 2560x1440" in encode
    probe_hdr = parse_probe(
        {"streams": [{"codec_type": "video", "width": 10, "height": 10, "color_transfer": "smpte2084", "nb_frames": "1"}]}
    )
    with pytest.raises(Exception, match="SDR"):
        commands_for(
            probe_hdr,
            Settings(),
            ffmpeg="ffmpeg",
            source="in.mkv",
            destination="out.mkv",
            encoders={"av1_nvenc"},
        )


def test_model_licences_are_permissive() -> None:
    assert licence_allowed("MIT")
    assert licence_allowed("Apache-2.0")
    assert licence_allowed("BSD-3-Clause")
    assert not licence_allowed("CC-BY-NC-4.0")
    assert not licence_allowed("GPL-3.0")
    assert not licence_allowed("")
    for model in CATALOG:
        assert licence_allowed(model.licence)
        assert len(model.sha256) == 64
        assert model.download_url.startswith("https://")


def test_theme_contrast_and_translations() -> None:
    for foreground, background in theme.CONTRAST_PAIRS:
        assert theme.contrast_ratio(foreground, background) >= 4.5
    assert theme.IDENTITY == "#2dd4bf"
    assert theme.IDENTITY != theme.GREEN
    assert theme.IDENTITY != "#86efac"
    assert theme.IDENTITY != "#f5d76e"
    assert theme.splash_art().is_file()
    assert tr("start", "en") == "Start"
    assert tr("start", "ru") == "Старт"
    english = json.loads(Path("locales/en.json").read_text(encoding="utf-8"))
    russian = json.loads(Path("locales/ru.json").read_text(encoding="utf-8"))
    assert set(english) == set(russian)


def test_platform_gate() -> None:
    assert windows_supported(26100)
    assert not windows_supported(22631)
    assert linux_supported('ID=ubuntu\nVERSION_ID="25.04"\n')
    assert not linux_supported('ID=ubuntu\nVERSION_ID="24.04"\n')
    assert linux_supported("ID=arch\n")
    assert linux_supported('ID=fedora\nVERSION_ID="42"\n')
    assert not linux_supported('ID=fedora\nVERSION_ID="41"\n')
    assert assess_platform(system="win32", win_build=26100).ok
    assert not assess_platform(system="win32", win_build=22000).ok
    assert not assess_platform(system="linux", os_release='ID=ubuntu\nVERSION_ID="22.04"\n').ok


def test_launch_uses_the_common_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card below compute capability 8.9 is refused by voxprint-common."""
    from voxprint_common.gpu_probe import GpuDevice

    monkeypatch.setattr("clarity.engine._require", lambda: object())
    low = [
        GpuDevice(
            index=0,
            name="NVIDIA GeForce RTX 3060",
            driver="610.00",
            compute=(8, 6),
            memory_mib=12288,
        )
    ]
    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", lambda runner=None: low)
    gate = assess_launch()
    assert gate.ok is False
    assert "8.6" in gate.message

    ada = [
        GpuDevice(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            driver="610.00",
            compute=(8, 9),
            memory_mib=24564,
        )
    ]
    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", lambda runner=None: ada)
    allowed = assess_launch()
    assert allowed.ok is True
    assert allowed.warning == ""


def test_lanczos_flag_does_not_start_clarity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")
    gate = assess_launch()
    assert gate.ok is False
    assert "VOXPRINT_CI_LANCZOS" in gate.message


class _Worker:
    def __init__(self, replies: list[dict[str, object]]) -> None:
        self.replies = list(replies)
        self.sent: list[dict[str, object]] = []
        self.killed = False

    def _send(self, message: dict[str, object]) -> None:
        self.sent.append(message)

    def _read(self, timeout: float) -> dict[str, object]:
        del timeout
        return self.replies.pop(0)

    def interrupt(self) -> None:
        self.killed = True


def test_restore_waits_when_the_gpu_lock_times_out() -> None:
    worker = _Worker(
        [
            {"op": "error", "ok": False, "message": "the GPU is busy: enlarge / enlarge:photo"},
            {"op": "result", "ok": True, "message": "onnx CUDA", "dst": "out.npy"},
        ]
    )
    seen: list[str] = []
    reply = submit_restore(
        worker,
        {"op": "restore_batch", "owner": "clarity", "id": "chunk"},
        on_status=seen.append,
        cancel=lambda: False,
        pause=lambda: None,
    )
    assert reply["dst"] == "out.npy"
    assert len(worker.sent) == 2
    assert worker.killed is False
    assert seen == [gpu_busy_text("the GPU is busy: enlarge / enlarge:photo")]
    assert "Waiting" in seen[0]


def test_restore_stops_when_the_worker_was_cancelled() -> None:
    worker = _Worker([{"op": "error", "ok": False, "message": "cancelled while waiting for the GPU"}])
    with pytest.raises(Cancelled):
        submit_restore(
            worker,
            {"op": "restore_batch", "owner": "clarity"},
            on_status=lambda _text: None,
            cancel=lambda: False,
            pause=lambda: None,
        )
    assert worker.killed is False
    assert len(worker.sent) == 1


def test_bench_report_mentions_fps(tmp_path: Path) -> None:
    markdown, payload = render_report("NVIDIA GeForce RTX 4090", [Sample("drunet-color", "pytorch-fp16", 12.5, 1800, 8, "ok")], "clip.mp4")
    assert "12.500" in markdown
    assert payload["clip_sha256"]
    md_path, json_path = write_report(tmp_path, "gpu", [], "clip.mp4")
    assert md_path.is_file() and json_path.is_file()


def test_tensorrt_providers_are_fp16_without_cpu(tmp_path: Path) -> None:
    providers = tensorrt_providers(tmp_path)
    name, options = providers[0]
    assert name == "TensorrtExecutionProvider"
    assert options["trt_fp16_enable"] is True
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_engine_cache_path"] == str(tmp_path)
    assert providers[1] == "CUDAExecutionProvider"
    joined = " ".join(item if isinstance(item, str) else item[0] for item in providers)
    assert "CPUExecutionProvider" not in joined


def test_pillow_pin_includes_the_pcf_bomb_fix() -> None:
    """Pillow 12.3.0 checks PCF fonts for decompression bombs (CVE-2026-54059)."""
    root = Path(__file__).resolve().parents[1]
    requirement = "Pillow>=12.3.0,<13"
    assert requirement in (root / "requirements.txt").read_text(encoding="utf-8")
    assert requirement in (root / "pyproject.toml").read_text(encoding="utf-8")
    lock = json.loads((root / "installer" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert lock["pillow"] == "12.3.0"
    script = (root / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    assert "Pillow==$pillow" in script
    installed = tuple(int(part) for part in version("pillow").split(".")[:3])
    assert installed >= (12, 3, 0)


def test_save_crops_writes_pngs(tmp_path: Path) -> None:
    before = np.zeros((8, 8, 3), dtype=np.float32)
    after = np.ones((8, 8, 3), dtype=np.float32)
    left, right = save_crops(before, after, tmp_path, "demo")
    assert left.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert right.is_file()
    assert right.stat().st_size > 8


def test_conditioned_preview_adds_the_deblock_channel() -> None:
    model = get_model("drunet-deblocking-color")
    nchw = conditioned_nchw(model, np.zeros((4, 4, 3), dtype=np.float32))
    assert nchw.shape == (1, 4, 4, 4)
    assert nchw[0, 3, 0, 0] == pytest.approx(deblock_level(40))


def test_drunet_noise_map_matches_a_batch_of_two() -> None:
    level = deblock_level(40)
    extra = noise_map(2, 8, 8, level)
    assert extra.shape == (2, 1, 8, 8)
    assert float(extra[0, 0, 0, 0]) == pytest.approx(level)
    assert np.allclose(extra[0], extra[1])
    with pytest.raises(ValueError):
        noise_map(0, 8, 8, level)
    seen: list[tuple[int, ...]] = []
    levels: list[float] = []

    def infer(_model: object, nchw: np.ndarray) -> np.ndarray:
        seen.append(tuple(int(part) for part in nchw.shape))
        levels.append(float(nchw[0, 3, 0, 0]))
        assert np.allclose(nchw[0, 3], nchw[1, 3])
        return nchw[:, :3]

    frames = np.zeros((2, 8, 8, 3), dtype=np.float32)
    settings = Settings(
        mode="clean",
        upscale=False,
        sharpen=0,
        white_balance=False,
        levels=False,
        tile=8,
        overlap=0,
        batch=2,
    )
    out = process_frames(frames, settings, infer, tile=8, batch=2, use_engine=False)
    assert out.shape == (2, 8, 8, 3)
    assert seen == [(2, 4, 8, 8), (2, 4, 8, 8)]
    assert levels[0] == pytest.approx(deblock_level(40))
    assert levels[1] == pytest.approx(sigma_level(15))


def test_bench_times_the_raw_module() -> None:
    class Raw:
        def forward(self, tensor: object) -> object:
            return tensor

    raw = Raw()

    class Descriptor:
        def __init__(self) -> None:
            self.model = raw

        def __call__(self, _tensor: object) -> object:
            raise AssertionError("descriptor")

    assert network_module(Descriptor()) is raw
    assert network_module(raw) is raw


def test_version_label_reads_build_json() -> None:
    data = json.loads(Path("BUILD.json").read_text(encoding="utf-8"))
    info = build_info()
    text = version_label()
    assert data["version"] == "1.0.0-rc"
    assert info.version == "1.0.0"
    assert info.rc is None
    assert info.show_rc is True
    assert text.startswith("1.0.0 RC build ")
    if info.build is None:
        assert "dev" in text
        assert "build 0" not in text
    else:
        assert f"build {info.build}" in text
    assert text.endswith(f'"{data["codename"]}"')
    assert release_title() == f"VOXPrint AI Clarity {text}"
    override = os.environ.get("VOXPRINT_BUILD", "").strip() or os.environ.get("VOXPRINT_BUILD_NUMBER", "").strip()
    run = os.environ.get("GITHUB_RUN_NUMBER", "").strip()
    if override.isdigit():
        assert info.build == int(override)
    elif run.isdigit():
        offset = data.get("offset")
        extra = offset if isinstance(offset, int) else 0
        assert info.build == int(run) + extra
    else:
        assert info.build == data["build"]


def test_bench_refuses_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise RuntimeError("no gpu")

    monkeypatch.setattr("clarity.bench.require_gpu", boom)
    assert bench_main(["--frames", "1"]) == 2
