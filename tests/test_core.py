"""Pipeline, catalogue, theme, and the pending engine video API. No GPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clarity.bench import Sample, main as bench_main, render_report, save_crops, tensorrt_providers, write_report
from clarity.bench import conditioned_nchw
from clarity.engine_shim import Gate, call_assess, video_module
from clarity.i18n import tr
from clarity.models import CATALOG, get_model, licence_allowed
from clarity.pipeline.colour import grey_world
from clarity.pipeline.condition import deblock_level, sigma_level
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


def test_temporal_windows_cover_every_frame_once() -> None:
    video = video_module()
    for count, length, overlap in ((1, 1, 0), (10, 1, 0), (10, 4, 1), (2, 8, 2), (20, 8, 3)):
        emitted: list[int] = []
        for window in video.temporal_windows(count, length, overlap):
            for slot in range(window.keep_start, window.keep_stop):
                emitted.append(window.input_start + slot)
        assert emitted == list(range(count))


def test_run_batch_keeps_the_batch_axis() -> None:
    video = video_module()

    class _Input:
        name = "images"
        type = "tensor(float16)"

    class _Session:
        def get_inputs(self) -> list[_Input]:
            return [_Input()]

        def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert feed["images"].dtype == np.float16
            return [feed["images"].astype(np.float32) + 1]

    batch = np.zeros((3, 3, 4, 5), dtype=np.float32)
    out = video.run_batch(_Session(), batch)
    assert out.shape == (3, 3, 4, 5)
    assert out.dtype == np.float32
    assert float(out.mean()) == pytest.approx(1.0)


def test_tile_frames_identity_roundtrip() -> None:
    video = video_module()
    rng = np.random.default_rng(0)
    frames = rng.random((2, 30, 40, 3), dtype=np.float32)

    def infer(batch: np.ndarray) -> np.ndarray:
        return batch

    restored = video.tile_frames(frames, infer, tile=16, overlap=4, scale=1, batch=3)
    assert restored.shape == frames.shape
    assert np.allclose(restored, frames, atol=1e-5)


def test_sysmem_spill_threshold() -> None:
    video = video_module()
    assert video.sysmem_spill(None, 0) is False
    assert video.sysmem_spill(300, 0, threshold=256) is True
    assert video.sysmem_spill(100, 0, threshold=256) is False
    assert video.parse_shared_usage('"counter","4096"\n') == 4096


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


def test_process_frames_clean_and_optional_2x() -> None:
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


def test_old_engine_warning_becomes_a_refusal() -> None:
    class _Gate:
        def __init__(self, ok: bool, message: str = "", warning: str = "") -> None:
            self.ok = ok
            self.message = message
            self.warning = warning

    def old_assess(refuse_below: object = None) -> _Gate:
        if refuse_below is not None:
            raise TypeError("unexpected keyword")
        return _Gate(True, "", "GPU compute capability 8.6 is below 8.9")

    gate = call_assess(old_assess)
    assert gate.ok is False
    assert "8.6" in gate.warning or "8.6" in gate.message

    def new_assess(refuse_below: tuple[int, int] | None = None) -> Gate:
        assert refuse_below == (8, 9)
        return Gate(False, "needs Ada", "")

    refused = call_assess(new_assess)
    assert refused.ok is False


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


def test_bench_refuses_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise RuntimeError("no gpu")

    monkeypatch.setattr("clarity.bench.require_gpu", boom)
    assert bench_main(["--frames", "1"]) == 2
