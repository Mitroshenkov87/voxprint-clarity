"""The shared runtime pin. No downloads."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from clarity.gpu_runtime import native_gap_directories

_ROOT = Path(__file__).resolve().parents[1]


def _pin():
    path = _ROOT / "installer" / "pin_runtime.py"
    spec = importlib.util.spec_from_file_location("clarity_pin_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load installer/pin_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lock_and_requirements(tmp_path: Path) -> tuple[Path, Path]:
    lock = tmp_path / "runtime_lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    requirements = tmp_path / "requirements-gpu.txt"
    requirements.write_text("torch==2.14.1\n", encoding="utf-8")
    return lock, requirements


def test_cpu_uninstall_targets_only_the_cpu_distribution() -> None:
    module = _pin()
    assert module.cpu_uninstall_args() == ["uninstall", "-y", "onnxruntime"]


def test_torch_args_use_the_gpu_requirements_file() -> None:
    module = _pin()
    args = module.torch_args(_ROOT / "requirements-gpu.txt")
    assert args[0] == "install"
    assert "-r" in args
    assert str(_ROOT / "requirements-gpu.txt") in args
    assert module.SPANDREL_SPEC == "spandrel>=0.4,<0.5"
    assert module.SPANDREL_SPEC in args


def test_uv_command_drops_the_pip_yes_flag() -> None:
    module = _pin()
    pip = module.pip_command("python", ["uninstall", "-y", "onnxruntime"])
    assert pip == ["python", "-m", "pip", "uninstall", "-y", "onnxruntime"]
    uv = module.pip_command("C:/venv/python.exe", ["uninstall", "-y", "onnxruntime"], uv="uv")
    assert uv == ["uv", "pip", "uninstall", "onnxruntime", "--python", "C:/venv/python.exe"]


def test_apply_pin_skips_uninstall_when_the_cpu_wheel_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _pin()
    calls: list[list[str]] = []

    def record(cmd: list[str], *, required: bool) -> None:
        assert required is True
        calls.append(cmd)

    seen: list[str] = []

    def providers(python: str) -> None:
        seen.append(python)

    monkeypatch.setattr(module, "_run", record)
    monkeypatch.setattr(module, "cpu_onnxruntime_installed", lambda _python: False)
    monkeypatch.setattr(module, "require_gpu_providers", providers)
    lock, requirements = _lock_and_requirements(tmp_path)
    module.apply_pin("python", lock, requirements)
    assert len(calls) == 1
    assert calls[0][:5] == ["python", "-m", "pip", "install", "-r"]
    assert seen == ["python"]


def test_apply_pin_uninstalls_a_cpu_wheel_and_requires_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _pin()
    calls: list[list[str]] = []
    probes = iter((True, False))
    seen: list[str] = []

    def record(cmd: list[str], *, required: bool) -> None:
        assert required is True
        calls.append(cmd)

    def providers(python: str) -> None:
        seen.append(python)

    monkeypatch.setattr(module, "_run", record)
    monkeypatch.setattr(module, "cpu_onnxruntime_installed", lambda _python: next(probes))
    monkeypatch.setattr(module, "require_gpu_providers", providers)
    lock, requirements = _lock_and_requirements(tmp_path)
    module.apply_pin("python", lock, requirements, uv="uv")
    assert calls[0][0] == "uv"
    assert calls[1][:3] == ["uv", "pip", "uninstall"]
    assert "onnxruntime" in calls[1]
    assert "-y" not in calls[1]
    assert seen == ["python"]


def test_apply_pin_fails_when_the_cpu_wheel_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _pin()

    def providers(_python: str) -> None:
        raise AssertionError("providers are checked only after the CPU wheel is gone")

    monkeypatch.setattr(module, "_run", lambda cmd, *, required: None)
    monkeypatch.setattr(module, "cpu_onnxruntime_installed", lambda _python: True)
    monkeypatch.setattr(module, "require_gpu_providers", providers)
    lock, requirements = _lock_and_requirements(tmp_path)
    with pytest.raises(module.PinError, match="still installed"):
        module.apply_pin("python", lock, requirements)


def test_missing_providers_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _pin()

    def fake_run(cmd: list[str], **_kwargs: object) -> object:
        return module.subprocess.CompletedProcess(cmd, 1, "missing CUDAExecutionProvider; found CPUExecutionProvider", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(module.PinError, match="still missing"):
        module.require_gpu_providers("python")


def test_gap_directories_match_the_app_scan(tmp_path: Path) -> None:
    module = _pin()
    root = tmp_path / "site-packages"
    arch = root / "nvidia" / "cuda_runtime" / "cu13" / "bin" / "x86_64"
    arch.mkdir(parents=True)
    (arch / "cudart64_13.dll").write_bytes(b"x")
    torch_lib = root / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "cublas64_13.dll").write_bytes(b"x")
    (torch_lib / "cublasLt64_13.dll").write_bytes(b"x")
    app = [path.resolve() for path in native_gap_directories([root])]
    installer = [path.resolve() for path in module._gap_directories([root])]
    assert installer == app
    assert arch.resolve() in installer
    assert torch_lib.resolve() in installer
    text = module._missing_runtime_message(installer)
    assert text == "Missing from the DLL search path: cudnn*, nvinfer_10"


def test_lock_rejects_a_damaged_file(tmp_path: Path) -> None:
    module = _pin()
    path = tmp_path / "runtime_lock.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(module.PinError):
        module.load_lock(path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(module.PinError):
        module.load_lock(path)
    lock = module.load_lock(_ROOT / "installer" / "runtime_lock.json")
    assert lock["nvidia_index"] == "https://pypi.nvidia.com"
    assert "[gpu]" in lock["engine"]
