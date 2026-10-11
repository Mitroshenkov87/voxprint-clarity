"""Regressions from the first RTX 4090 run. No GPU and no PowerShell 5.1."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from clarity.bench import export_example_shape, export_with_engine, tensorrt_providers
from clarity.cli import main
from clarity.errors import ClarityError
from clarity.models import CATALOG, clean_models, strong_model
from clarity.net import install_system_trust
from clarity.pipeline.encode import Container, encode_chunk_command, encode_command, encoder_pix_fmt
from clarity.pipeline.job import Settings, downscale_half
from clarity.pipeline.runner import _prepare_infer
from clarity.precision import PRECISIONS, apply_precision
from clarity.ui.platform import (
    gui_qt_platform,
    require_windows_platform_plugin,
    selftest_qt_platform,
    windows_platform_plugin,
)

_ROOT = Path(__file__).resolve().parents[1]


def _ps51_native_argv(text: str) -> list[str]:
    """Windows PowerShell 5.1 drops quotes and splits native arguments on whitespace."""
    return text.replace('"', "").split()


def test_every_catalogue_model_declares_a_loadable_precision() -> None:
    for model in CATALOG:
        assert model.precision in PRECISIONS
        if model.arch == "drunet":
            assert model.precision == "fp32"


def test_fp32_load_does_not_call_half() -> None:
    class _Net:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def half(self) -> _Net:
            self.calls.append("half")
            raise RuntimeError("does not support fp16")

        def bfloat16(self) -> _Net:
            self.calls.append("bf16")
            return self

        def float(self) -> _Net:
            self.calls.append("fp32")
            return self

    for model in CATALOG:
        net = _Net()
        if model.precision == "fp32":
            assert apply_precision(net, model.precision) is net
            assert net.calls == ["fp32"]
        elif model.precision == "fp16":
            with pytest.raises(RuntimeError, match="fp16"):
                apply_precision(net, "fp16")
        else:
            assert apply_precision(net, model.precision) is net
            assert net.calls == ["bf16"]


def test_clean_mode_exports_drunet_with_four_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []
    batches: list[tuple[int, ...]] = []

    def ensure(manifest: dict[str, object], path: Path, on_progress: object) -> Path:
        del path
        assert callable(on_progress)
        seen.append(manifest)
        return Path("drunet.onnx")

    def open_session(path: Path, progress: object) -> object:
        del path
        assert callable(progress)
        return object()

    def run_batch(session: object, nchw: np.ndarray) -> np.ndarray:
        del session
        batches.append(tuple(int(part) for part in nchw.shape))
        return np.zeros((nchw.shape[0], 3, nchw.shape[2], nchw.shape[3]), dtype=np.float32)

    monkeypatch.setattr("clarity.pipeline.runner.ensure_weight", lambda manifest: Path(str(manifest["id"])))
    monkeypatch.setattr("clarity.pipeline.runner.ensure_model_onnx", ensure)
    monkeypatch.setattr("clarity.pipeline.runner.open_restore_session", open_session)
    monkeypatch.setattr("clarity.pipeline.runner.run_onnx_batch", run_batch)
    infer = _prepare_infer(Settings(mode="clean"), lambda _fraction, _text: None)
    model = clean_models()[0]
    assert model.precision == "fp32"
    assert model.channels == 4
    output = infer(model, np.zeros((1, 4, 8, 6), dtype=np.float32))
    assert output.shape == (1, 3, 8, 6)
    assert seen[0]["in_channels"] == 4
    assert batches == [(1, 4, 8, 6)]
    denoise = clean_models()[1]
    infer(denoise, np.zeros((1, 4, 8, 6), dtype=np.float32))
    assert seen[1]["in_channels"] == 4


def test_drunet_export_plan_uses_1x4xhxw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    height, width = 64, 48
    for model in clean_models():
        assert model.arch == "drunet"
        assert model.precision == "fp32"
        assert model.manifest()["in_channels"] == 4
        assert export_example_shape(model.channels, height, width) == (1, 4, height, width)
    captured: dict[str, int] = {}

    def export(weight: Path, dest: Path, on_progress: object = None, *, channels: int = 3) -> None:
        del weight, on_progress
        captured["channels"] = channels
        dest.write_bytes(b"onnx")

    class _Export:
        export_weight = staticmethod(export)

    monkeypatch.setattr("clarity.bench.importlib.import_module", lambda name: _Export)
    weight = tmp_path / "drunet.pth"
    weight.write_bytes(b"pth")
    note = export_with_engine(weight, tmp_path / "drunet.onnx", channels=4)
    assert captured["channels"] == 4
    assert "fp16" in note


def test_prepare_infer_converts_a_three_channel_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    heard: list[str] = []

    def ensure(manifest: dict[str, object], path: Path, on_progress: object) -> Path:
        del manifest, path
        assert callable(on_progress)
        on_progress(0.2, "converting the weight to ONNX")
        calls.append("onnx")
        return Path("model.onnx")

    def open_session(path: Path, progress: object) -> object:
        del path
        assert callable(progress)
        progress(0.0, "building the TensorRT engine")
        calls.append("session")
        return object()

    def run_batch(session: object, nchw: np.ndarray) -> np.ndarray:
        del session
        calls.append("batch")
        return np.zeros_like(nchw)

    monkeypatch.setattr("clarity.pipeline.runner.ensure_weight", lambda manifest: Path(str(manifest["id"])))
    monkeypatch.setattr("clarity.pipeline.runner.ensure_model_onnx", ensure)
    monkeypatch.setattr("clarity.pipeline.runner.open_restore_session", open_session)
    monkeypatch.setattr("clarity.pipeline.runner.run_onnx_batch", run_batch)
    infer = _prepare_infer(Settings(mode="strong"), lambda _fraction, text: heard.append(text))
    model = strong_model()
    assert model.channels == 3
    output = infer(model, np.zeros((1, 3, 2, 2), dtype=np.float32))
    assert output.shape == (1, 3, 2, 2)
    assert calls == ["onnx", "session", "batch"]
    assert "building the TensorRT engine" in heard


def test_output_pix_fmt_follows_the_source_depth() -> None:
    assert encoder_pix_fmt("") == "yuv420p"
    assert encoder_pix_fmt("yuv420p") == "yuv420p"
    assert encoder_pix_fmt("yuv420p10le") == "p010le"
    assert encoder_pix_fmt("p010le") == "p010le"
    command = encode_command(
        "ffmpeg",
        "in.mkv",
        container=Container.MKV,
        encoder="av1_nvenc",
        destination="out.mkv",
        copy_audio=True,
        width=16,
        height=16,
        fps=24,
        pix_fmt="p010le",
    )
    video = command.index("-c:v")
    assert command.index("-pix_fmt") < video
    assert command[command.index("-pix_fmt", video) + 1] == "p010le"
    chunk = encode_chunk_command(
        "ffmpeg",
        encoder="av1_nvenc",
        destination="chunk.mkv",
        width=16,
        height=16,
        fps=24,
    )
    video = chunk.index("-c:v")
    assert chunk[chunk.index("-pix_fmt", video) + 1] == "yuv420p"


def test_restore_json_reports_a_job_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.mkv"
    source.write_bytes(b"video")
    monkeypatch.setattr("clarity.cli._gates", lambda _console: None)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise MemoryError("Unable to allocate 22.7 GiB")

    monkeypatch.setattr("clarity.cli.run_job", explode)
    code = main(
        ["restore", "--input", str(source), "--output", str(tmp_path / "out.mkv"), "--json"]
    )
    assert code == 3
    captured = capsys.readouterr()
    result = json.loads(captured.out.strip().splitlines()[-1])
    assert result["type"] == "result"
    assert result["ok"] is False
    assert "MemoryError" in result["error"]
    assert "22.7" in result["error"]
    assert "MemoryError" in captured.err


def test_downscale_half_uses_area_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _Tensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def permute(self, *axes: int) -> _Tensor:
            return _Tensor(tuple(self.shape[axis] for axis in axes))

        def to(self, **kwargs: object) -> _Tensor:
            calls["to"] = kwargs
            return self

        def float(self) -> _Tensor:
            return self

        def cpu(self) -> _Tensor:
            return self

        def numpy(self) -> np.ndarray:
            return np.zeros(self.shape, dtype=np.float32)

    class _Functional:
        @staticmethod
        def interpolate(tensor: _Tensor, scale_factor: float, mode: str) -> _Tensor:
            calls["scale"] = scale_factor
            calls["mode"] = mode
            count, channels, height, width = tensor.shape
            return _Tensor((count, channels, height // 2, width // 2))

    class _NN:
        functional = _Functional

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch:
        cuda = _Cuda
        nn = _NN

        @staticmethod
        def from_numpy(array: np.ndarray) -> _Tensor:
            return _Tensor(tuple(int(part) for part in array.shape))

    monkeypatch.setattr("clarity.pipeline.job.torch", _Torch)
    out = downscale_half(np.zeros((1, 4, 4, 3), dtype=np.float32))
    assert calls["mode"] == "area"
    assert calls["scale"] == 0.5
    assert calls["to"] == {"device": "cuda"}
    assert out.shape == (1, 2, 2, 3)


def test_gui_does_not_force_offscreen() -> None:
    assert gui_qt_platform() is None
    assert selftest_qt_platform() == "offscreen"
    source = (_ROOT / "clarity" / "cli.py").read_text(encoding="utf-8")
    assert "isatty" not in source
    assert 'setdefault("QT_QPA_PLATFORM"' not in source


def test_missing_windows_plugin_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("clarity.ui.platform.sys.platform", "win32")
    monkeypatch.setattr("clarity.ui.platform.windows_platform_plugin", lambda: tmp_path / "qwindows.dll")
    with pytest.raises(ClarityError, match="Windows platform plugin"):
        require_windows_platform_plugin()


def test_windows_platform_plugin_loads_on_windows() -> None:
    path = windows_platform_plugin()
    if sys.platform != "win32":
        assert not path.is_file()
        require_windows_platform_plugin()
        return
    assert path.is_file()
    require_windows_platform_plugin()


def test_system_trust_is_idempotent() -> None:
    install_system_trust()
    install_system_trust()


def test_onnx_and_pillow_stay_pinned() -> None:
    requirements = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    package = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    bench = (_ROOT / "requirements-bench.txt").read_text(encoding="utf-8")
    bench_source = (_ROOT / "clarity" / "bench.py").read_text(encoding="utf-8")
    assert "onnx>=" in requirements
    assert "onnx>=" in package
    assert "truststore>=" in requirements
    assert "onnxruntime-gpu==1.31.0" in bench
    lock = json.loads((_ROOT / "installer" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert lock["pillow"] == "12.3.0"
    assert "onnxruntime_gpu" not in lock
    assert "tensorrt_cu13" not in lock
    assert "torch.onnx" not in bench_source
    assert "export_weight" in bench_source


def test_tensorrt_fp32_leaves_fp16_off(tmp_path: Path) -> None:
    _name, options = tensorrt_providers(tmp_path, precision="fp32")[0]
    assert options["trt_fp16_enable"] is False


def test_lgpl_check_ignores_disabled_gpl_libraries() -> None:
    script = (_ROOT / "scripts" / "test-gpu.ps1").read_text(encoding="utf-8")
    assert "--enable-gpl" in script
    assert "--enable-libx264" in script
    assert "--enable-libx265" in script
    assert "python -c $providerCheck" not in script
    assert "provider_check.py" not in script
    assert "9183489" in script
    assert "--no-deps" not in script

    def rejected(build: str) -> bool:
        return bool(
            re.search(r"--enable-gpl", build)
            or re.search(r"--enable-libx264", build)
            or re.search(r"--enable-libx265", build)
        )

    lgpl = "configuration: --disable-gpl --disable-libx264 --disable-libx265"
    gpl = "configuration: --enable-gpl --enable-libx264"
    assert rejected(lgpl) is False
    assert rejected(gpl) is True
    stripped = _ps51_native_argv('python -c "import os\nprint(\'ok\')"')
    assert stripped[0:2] == ["python", "-c"]
    assert len(stripped) > 3
    kept = _ps51_native_argv(r"python C:\results\provider_check.py")
    assert kept == ["python", r"C:\results\provider_check.py"]


def test_runtime_junction_is_created_after_the_files_exist() -> None:
    windows = (_ROOT / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    linux = (_ROOT / "installer" / "linux" / "install-shared.sh").read_text(encoding="utf-8")
    iss = (_ROOT / "installer" / "VoxprintClarity.iss").read_text(encoding="utf-8")
    assert "Pillow==$pillow" in windows
    assert "[switch]$LinkOnly" in windows
    assert "mklink /J failed" in windows
    assert "tensorrt-cu13" not in windows
    assert "--no-deps" not in windows
    assert "onnxruntime-gpu" not in linux
    assert "--no-deps" not in linux
    assert "9183489" in windows
    assert "9183489" in linux
    link_lines = [line for line in iss.splitlines() if "LinkOnly" in line]
    assert link_lines
    assert all("RuntimeInstalled" not in line for line in link_lines)


def test_engine_streaming_and_onnx_live_in_the_tag() -> None:
    job = (_ROOT / "clarity" / "pipeline" / "job.py").read_text(encoding="utf-8")
    runner = (_ROOT / "clarity" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    assert "iter_tile_frames" in job
    assert "restore_frames(" not in job
    assert "ensure_model_onnx" in runner
    assert not (_ROOT / "patches").exists()
    assert not (_ROOT / "clarity" / "ort_check.py").exists()
