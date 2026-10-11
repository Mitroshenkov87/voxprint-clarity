"""Shared suite paths, refcount, and a mocked repair. No GPU and no downloads."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from clarity.cli import main
from clarity.engine import NVIDIA_DRIVER_URL, assess_launch, driver_refusal
from clarity.suite_share import (
    component_version,
    hold_component,
    read_manifest,
    release_component,
    shared_root,
    voxprint_home,
    write_marker,
)
from clarity.ui.suite_dialog import RepairProgress

_ROOT = Path(__file__).resolve().parents[1]


def test_home_follows_the_override_and_the_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "override"))
    assert voxprint_home() == tmp_path / "override"
    assert shared_root() == tmp_path / "override" / "shared"

    monkeypatch.delenv("VOXPRINT_HOME", raising=False)

    class _Paths:
        def voxprint_home(self) -> Path:
            return tmp_path / "engine-home"

    monkeypatch.setattr("clarity.suite_share._engine_paths", _Paths())
    assert voxprint_home() == tmp_path / "engine-home"


def test_home_falls_back_when_the_engine_is_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VOXPRINT_HOME", raising=False)
    monkeypatch.setattr("clarity.suite_share._engine_paths", None)
    monkeypatch.setattr("clarity.suite_share.sys.platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert voxprint_home() == tmp_path / "xdg" / "voxprint"
    monkeypatch.setenv("XDG_DATA_HOME", "relative")
    assert voxprint_home() == Path.home() / ".local" / "share" / "voxprint"
    monkeypatch.setattr("clarity.suite_share.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert voxprint_home() == tmp_path / "local" / "Voxprint"


def test_refcount_keeps_a_shared_component_until_the_last_user(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    assert hold_component("ffmpeg", "lgpl", app="clarity", root=root) == 1
    assert hold_component("ffmpeg", "lgpl", app="clarity", root=root) == 2
    assert hold_component("ffmpeg", "lgpl", app="enlarge", root=root) == 1
    binary = root / "ffmpeg" / "ffmpeg"
    binary.parent.mkdir(parents=True)
    binary.write_text("ffmpeg", encoding="utf-8")
    assert release_component("ffmpeg", app="clarity", root=root) == 1
    assert binary.is_file()
    assert release_component("ffmpeg", app="clarity", root=root) == 0
    assert binary.is_file()
    assert release_component("ffmpeg", app="enlarge", root=root) == 0
    assert not binary.exists()


def test_runtime_is_removed_only_when_python_torch_and_the_engine_are_free(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    hold_component("python", component_version("python"), root=root)
    hold_component("torch", component_version("torch"), root=root)
    hold_component("engine", component_version("engine"), root=root)
    runtime = root / "runtime" / "venv"
    runtime.mkdir(parents=True)
    weights = root / "models" / "weight.bin"
    weights.parent.mkdir(parents=True)
    weights.write_text("weight", encoding="utf-8")
    write_marker("models", root)
    hold_component("models", "default", root=root)
    release_component("python", root=root)
    release_component("torch", root=root)
    assert runtime.is_dir()
    release_component("engine", root=root)
    assert not (root / "runtime").exists()
    release_component("models", root=root)
    assert weights.is_file()
    assert not (root / "models" / "READY").exists()


def test_pins_match_the_runtime_lock() -> None:
    lock = json.loads((_ROOT / "installer" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert component_version("python") == lock["python"]
    assert component_version("torch") == f"{lock['torch']}+{lock['cuda']}"
    assert component_version("engine") == "engine-v1.0.0"
    script = (_ROOT / "installer" / "linux" / "install-shared.sh").read_text(encoding="utf-8")
    windows = (_ROOT / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    package = (_ROOT / "installer" / "linux" / "build-package.sh").read_text(encoding="utf-8")
    iss = (_ROOT / "installer" / "VoxprintClarity.iss").read_text(encoding="utf-8")
    for text in (script, windows):
        assert "3.14" in text
        assert "2.14.1" in text
        assert "cu130" in text
        assert "engine-v1.0.0" in text
    assert "ffmpeg-master-latest-linux64-lgpl" in script
    assert "ffmpeg-master-latest-win64-lgpl" in windows
    assert "voxprint-clarity-install" in package
    assert "shared/ffmpeg" in package
    assert "--install-suite" in iss


def test_selftest_json_repairs_a_missing_component(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    installed: list[str] = []

    def probe(name: str, root: Path) -> bool:
        if name != "ffmpeg":
            return True
        return (root / "ffmpeg" / "READY").is_file()

    def install(name: str, root: Path, progress: object) -> None:
        del progress
        installed.append(name)
        write_marker("ffmpeg", root)

    monkeypatch.setattr("clarity.suite_repair.default_probe", probe)
    monkeypatch.setattr("clarity.suite_repair.install_component", install)
    monkeypatch.setattr("clarity.cli._selftest_window", lambda: 0)

    assert main(["selftest", "--dry-run", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["gpu_used"] is False
    assert "suite dependencies" in dry["checks"]
    assert installed == []

    assert main(["selftest", "--json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    progress = [row for row in lines if row.get("type") == "progress"]
    assert any(row["message"] == "repairing ffmpeg" for row in progress)
    assert any(row["message"] == "ffmpeg repaired" for row in progress)
    assert installed == ["ffmpeg"]
    users = read_manifest(tmp_path / "home" / "shared")["components"]["ffmpeg"]["users"]
    assert users["clarity"] == 1

    assert main(["selftest", "--json"]) == 0
    assert installed == ["ffmpeg"]
    users = read_manifest(tmp_path / "home" / "shared")["components"]["ffmpeg"]["users"]
    assert users["clarity"] == 1


def test_old_driver_includes_the_nvidia_download_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)

    class _Gate:
        def __init__(self) -> None:
            self.ok = False
            self.message = "NVIDIA driver 551.76 is below the required 600 series"
            self.warning = ""

    class _Hardware:
        def assess_gpu(self, refuse_below: tuple[int, int]) -> _Gate:
            del refuse_below
            return _Gate()

    monkeypatch.setattr("clarity.engine._engine_api", object())
    monkeypatch.setattr("clarity.engine._hardware", _Hardware())
    gate = assess_launch()
    assert gate.ok is False
    assert "551.76" in gate.message
    assert "R600" in gate.message
    assert NVIDIA_DRIVER_URL in gate.message
    assert NVIDIA_DRIVER_URL in driver_refusal("551.76")


def test_repair_dialog_opens_only_while_repairing() -> None:
    app = QApplication.instance() or QApplication([])
    progress = RepairProgress()
    progress(1.0, "python ready")
    assert progress.dialog is None
    progress(0.2, "repairing ffmpeg")
    assert progress.dialog is not None
    assert "ffmpeg" in progress.dialog.labelText()
    progress.dialog.close()
    app.processEvents()
