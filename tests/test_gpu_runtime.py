"""DLL search path, the CPU-only session error, and JSON phase lines."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from clarity.cli import Console
from clarity.gpu_runtime import (
    GpuRuntimeError,
    missing_runtime_message,
    native_gap_directories,
    prepare_gpu_runtime,
    register_native_gap,
    require_gpu_session,
    require_tensorrt_and_cuda,
    reset_gpu_runtime,
)
from clarity.native_gap import call_capturing_stderr
from clarity.progress import pulse_until


def _runtime_tree(root: Path) -> dict[str, Path]:
    """Nested cu13 layout. cuBLAS exists only under torch/lib."""
    arch = root / "nvidia" / "cuda_runtime" / "cu13" / "bin" / "x86_64"
    arch.mkdir(parents=True)
    (arch / "cudart64_13.dll").write_bytes(b"x")
    cudnn = root / "nvidia" / "cudnn" / "bin"
    cudnn.mkdir(parents=True)
    (cudnn / "cudnn64_9.dll").write_bytes(b"x")
    trt = root / "tensorrt_libs"
    trt.mkdir(parents=True)
    (trt / "nvinfer_10.dll").write_bytes(b"x")
    torch_lib = root / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "cublas64_13.dll").write_bytes(b"x")
    (torch_lib / "cublasLt64_13.dll").write_bytes(b"x")
    return {"arch": arch, "cudnn": cudnn, "trt": trt, "torch": torch_lib}


def test_gap_directories_include_nested_cu13_and_torch_lib(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    paths = _runtime_tree(root)
    found = [path.resolve() for path in native_gap_directories([root])]
    assert paths["arch"].resolve() in found
    assert paths["cudnn"].resolve() in found
    assert paths["trt"].resolve() in found
    assert paths["torch"].resolve() in found
    assert (paths["arch"].parent).resolve() not in found


def test_missing_runtime_message_lists_only_absent_libraries(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    arch = root / "nvidia" / "cu13" / "bin" / "x86_64"
    arch.mkdir(parents=True)
    (arch / "cudart64_13.dll").write_bytes(b"x")
    torch_lib = root / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "cublas64_13.dll").write_bytes(b"x")
    (torch_lib / "cublasLt64_13.dll").write_bytes(b"x")
    text = missing_runtime_message([root])
    assert text == "Missing from the DLL search path: cudnn*, nvinfer_10"
    empty = tmp_path / "empty"
    empty.mkdir()
    absent = missing_runtime_message([empty])
    assert absent == (
        "Missing from the DLL search path: cudart64_13, cublas64_13, cublasLt64_13, cudnn*, nvinfer_10"
    )


def test_empty_torch_lib_is_registered(tmp_path: Path) -> None:
    root = tmp_path / "site"
    lib = root / "torch" / "lib"
    lib.mkdir(parents=True)
    assert [path.resolve() for path in native_gap_directories([root])] == [lib.resolve()]


def test_register_prepends_the_nested_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "site-packages"
    paths = _runtime_tree(root)
    added: list[str] = []

    def add(path: str) -> object:
        added.append(path)
        return path

    monkeypatch.setattr(os, "add_dll_directory", add, raising=False)
    env: dict[str, str] = {"PATH": "/usr/bin"}
    found = register_native_gap([root], platform="win32", env=env)
    assert [Path(path).resolve() for path in added] == [path.resolve() for path in found]
    parts = env["PATH"].split(os.pathsep)
    assert parts[-1] == "/usr/bin"
    listed = {Path(part).resolve() for part in parts[:-1]}
    for folder in paths.values():
        assert folder.resolve() in listed


def test_register_keeps_windows_dll_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "site"
    trt = root / "tensorrt_libs"
    trt.mkdir(parents=True)
    (trt / "nvinfer_10.dll").write_bytes(b"x")
    added: list[str] = []

    def add(path: str) -> object:
        added.append(path)
        return path

    monkeypatch.setattr(os, "add_dll_directory", add, raising=False)
    env: dict[str, str] = {}
    first = register_native_gap([root], platform="win32", env=env)
    second = register_native_gap([root], platform="win32", env=env)
    assert first == [trt]
    assert [Path(path) for path in added] == [trt, trt]
    assert env["PATH"].split(os.pathsep)[0] == str(trt)
    assert second == [trt]
    assert register_native_gap([root], platform="linux", env={}) == [trt]


def test_prepare_calls_the_engine_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_gpu_runtime()
    calls: list[str] = []

    def helper() -> None:
        calls.append("engine")

    monkeypatch.setattr("clarity.gpu_runtime._engine_prepare", lambda: helper)
    seen: list[str] = []
    monkeypatch.setattr("clarity.gpu_runtime.register_native_gap", lambda _roots: seen.append("gap") or [])
    prepare_gpu_runtime()
    prepare_gpu_runtime()
    assert calls == ["engine"]
    assert seen == ["gap"]
    reset_gpu_runtime()


def test_prepare_registers_the_gap_when_the_helper_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_gpu_runtime()

    def missing() -> object:
        raise ImportError("prepare_native_runtime")

    seen: list[int] = []
    monkeypatch.setattr("clarity.gpu_runtime._engine_prepare", missing)
    monkeypatch.setattr("clarity.gpu_runtime.register_native_gap", lambda _roots: seen.append(1) or [])
    prepare_gpu_runtime()
    prepare_gpu_runtime()
    assert seen == [1]
    reset_gpu_runtime()


def test_session_provider_fallback_names_the_missing_dll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("clarity.gpu_runtime.prepare_gpu_runtime", lambda: None)
    monkeypatch.setattr("clarity.gpu_runtime.site_package_roots", lambda: [tmp_path])

    def fake(name: str) -> tuple[list[str], str]:
        if name == "CUDAExecutionProvider":
            return ["CPUExecutionProvider"], "CUDAExecutionProvider fell back to ['CPUExecutionProvider']"
        return [name], ""

    session = types.ModuleType("voxprint_engine.session")
    session.session_providers = fake  # type: ignore[attr-defined]
    engine = types.ModuleType("voxprint_engine")
    monkeypatch.setitem(sys.modules, "voxprint_engine", engine)
    monkeypatch.setitem(sys.modules, "voxprint_engine.session", session)
    with pytest.raises(GpuRuntimeError, match="nvinfer_10") as caught:
        require_tensorrt_and_cuda()
    text = str(caught.value)
    for name in ("cudart64_13", "cublas64_13", "cublasLt64_13", "cudnn*", "nvinfer_10"):
        assert name in text


def test_cpu_only_session_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("clarity.gpu_runtime.site_package_roots", lambda: [tmp_path])

    class _Session:
        @staticmethod
        def get_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    with pytest.raises(GpuRuntimeError, match="cublas64_13") as caught:
        require_gpu_session(_Session())
    assert "nvinfer_10" in str(caught.value)
    assert require_gpu_session(object()) == []


def test_stderr_capture_reads_the_process_stream() -> None:
    def write() -> str:
        os.write(2, b"loader line\n")
        return "ok"

    result, text = call_capturing_stderr(write)
    assert result == "ok"
    assert "loader line" in text


def test_json_event_flushes_phase_and_frame_fields(capsys: pytest.CaptureFixture[str]) -> None:
    console = Console(True)
    console.event("building_engine", 0.0, "building", elapsed_s=4.2)
    building = json.loads(capsys.readouterr().out)
    assert building["type"] == "progress"
    assert building["phase"] == "building_engine"
    assert building["stage"] == "building_engine"
    assert building["elapsed_s"] == 4.2
    console.event("frames", 0.5, "4/8", frames_done=4, frames_total=8, fps=3.5, chunk=1)
    frames = json.loads(capsys.readouterr().out)
    assert frames["phase"] == "frames"
    assert frames["frames_done"] == 4
    assert frames["frames_total"] == 8
    assert frames["fps"] == 3.5
    assert frames["chunk"] == 1
    assert "stage" not in frames


def test_pulse_emits_elapsed_until_stopped() -> None:
    stop = threading.Event()
    seen: list[float] = []

    def emit(elapsed: float) -> None:
        seen.append(elapsed)
        stop.set()

    thread = pulse_until(stop, emit, time.monotonic(), 0.05)
    thread.join(timeout=1)
    assert seen
    assert seen[0] >= 0
