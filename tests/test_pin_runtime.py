"""The shared ONNX Runtime and TensorRT pin. No downloads."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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


def test_gpu_pin_removes_the_cpu_wheel_before_the_lock_pair() -> None:
    module = _pin()
    lock = module.load_lock(_ROOT / "installer" / "runtime_lock.json")
    steps = module.gpu_pin_steps(lock)
    commands = [args for args, _required in steps]
    assert commands[0][:3] == ["uninstall", "-y", "onnxruntime"]
    assert "onnxruntime-gpu" in commands[0]
    install = commands[1]
    assert install[0] == "install"
    assert "onnxruntime-gpu==1.31.0" in install
    assert "tensorrt-cu13-libs==10.14.1.48" in install
    assert install[install.index("--extra-index-url") + 1].startswith("https://")
    assert commands[2] == ["uninstall", "-y", "onnxruntime"]
    assert commands[3] == ["install", "--no-deps", "tensorrt-cu13==10.14.1.48"]
    assert commands[4] == ["uninstall", "-y", "onnxruntime"]
    assert steps[0][1] is False
    assert steps[1][1] is True
    assert steps[3][1] is True


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


def test_lock_rejects_a_missing_pin(tmp_path: Path) -> None:
    module = _pin()
    path = tmp_path / "runtime_lock.json"
    path.write_text('{"onnxruntime-gpu": "1.31.0"}\n', encoding="utf-8")
    with pytest.raises(module.PinError):
        module.gpu_pin_steps(module.load_lock(path))
