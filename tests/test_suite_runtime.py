"""Engine holder, ffmpeg repair, and the driver link. No GPU and no downloads."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from voxprint_common.gpu_lock import Acquired, Holder, try_acquire
from voxprint_common.gpu_probe import GpuDevice, old_driver_message

from clarity.cli import _ensure_shared, main
from clarity.appinfo import APP_NAME
from clarity.engine import (
    ENGINE_REQUIREMENT,
    ENGINE_TAG,
    NVIDIA_DRIVER_URL,
    SUITE_RUNTIME,
    EngineUnavailable,
    OWNER,
    assess_launch,
    driver_refusal,
    ensure_shared_runtime,
    hold_gpu,
    release_shared_runtime,
)
from clarity.errors import ClarityError
from clarity.ffmpeg_setup import ensure_ffmpeg
from clarity.ui.suite_dialog import RepairProgress

_ROOT = Path(__file__).resolve().parents[1]


class _Report:
    def __init__(self, *, ok: bool, payload: dict[str, Any] | None = None) -> None:
        self.ok = ok
        self.components: list[object] = []
        self._payload = payload or {"ok": ok, "skipped": False, "components": []}

    def as_dict(self) -> dict[str, Any]:
        return self._payload


class _SuiteInstallError(Exception):
    pass


def test_runtime_holder_is_clarity_and_install_errors_become_clarity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    class _Module:
        SuiteInstallError = _SuiteInstallError

        def ensure_runtime(self, app: str, progress: object = None, **_extra: object) -> _Report:
            seen["ensure"] = app
            if callable(progress):
                progress("Checking python 3.14")
            return _Report(ok=True)

        def release_runtime(self, app: str, progress: object = None) -> _Report:
            seen["release"] = app
            if callable(progress):
                progress("Removing python 3.14")
            return _Report(ok=True)

    module = _Module()

    def load(name: str) -> _Module:
        assert name == "enlarge.suite_runtime"
        return module

    monkeypatch.setattr("clarity.engine._load_module", load)
    report = ensure_shared_runtime()
    released = release_shared_runtime()
    assert seen == {"ensure": "clarity", "release": "clarity"}
    assert report.ok is True
    assert released.ok is True

    def refuse(app: str, progress: object = None, **_extra: object) -> _Report:
        del app, progress
        raise _SuiteInstallError("Python 3.14 was not found")

    module.ensure_runtime = refuse  # type: ignore[method-assign]
    with pytest.raises(ClarityError, match="Python 3.14 was not found"):
        ensure_shared_runtime()


def test_missing_suite_runtime_names_the_pinned_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("clarity.engine._load_module", lambda name: None)
    with pytest.raises(EngineUnavailable, match="run repair"):
        ensure_shared_runtime()
    with pytest.raises(EngineUnavailable, match="run repair"):
        release_shared_runtime()
    with pytest.raises(EngineUnavailable, match="VOXPrint inference engine missing"):
        ensure_shared_runtime()


def test_enlarge_bundle_message_stands_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Module:
        SuiteInstallError = _SuiteInstallError

        def ensure_runtime(self, app: str, progress: object = None, **_extra: object) -> _Report:
            del app, progress, _extra
            raise _SuiteInstallError(
                "Python 3.14 was not found. Install the Enlarge bundle, which carries that runtime."
            )

    monkeypatch.setattr("clarity.engine._load_module", lambda name: _Module())
    with pytest.raises(ClarityError, match="Clarity installs Python 3.14 with uv") as caught:
        ensure_shared_runtime()
    assert "Enlarge" not in str(caught.value)


def test_ensure_runtime_does_not_install_the_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Module:
        SuiteInstallError = _SuiteInstallError

        @staticmethod
        def ensure_runtime(app: str, progress: object = None, **extra: object) -> _Report:
            seen["app"] = app
            seen["extra"] = extra
            if callable(progress):
                progress("Checking python 3.14")
            return _Report(ok=True)

    def load(name: str) -> object:
        assert name == "enlarge.suite_runtime"
        return _Module()

    monkeypatch.setattr("clarity.engine._load_module", load)
    report = ensure_shared_runtime()
    assert report.ok is True
    assert seen["app"] == "clarity"
    extra = seen["extra"]
    assert isinstance(extra, dict)
    assert "probes" not in extra
    assert "installers" not in extra


def test_skipped_runtime_does_not_touch_the_model_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Skipped:
        ok = True
        skipped = True
        components: tuple[object, ...] = ()

    class _Module:
        SuiteInstallError = _SuiteInstallError

        @staticmethod
        def read_manifest() -> dict[str, object]:
            raise AssertionError("manifest")

        @staticmethod
        def ensure_runtime(app: str, progress: object = None, **_extra: object) -> _Skipped:
            del app, progress, _extra
            return _Skipped()

    monkeypatch.setattr("clarity.engine._load_module", lambda name: _Module() if "suite_runtime" in name else None)
    report = ensure_shared_runtime()
    assert report.skipped is True


def test_shared_check_repairs_ffmpeg_after_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def ensure(progress: object) -> _Report:
        assert callable(progress)
        progress("Checking python 3.14")
        order.append("runtime")
        return _Report(ok=True)

    def ffmpeg(progress: object) -> str:
        assert callable(progress)
        progress("Installing ffmpeg")
        order.append("ffmpeg")
        return "repaired"

    monkeypatch.setattr("clarity.cli.ensure_shared_runtime", ensure)
    monkeypatch.setattr("clarity.cli.ensure_ffmpeg", ffmpeg)
    messages: list[str] = []
    report = _ensure_shared(messages.append)
    assert report.ok is True
    assert order == ["runtime", "ffmpeg"]
    assert messages == ["Checking python 3.14", "Installing ffmpeg"]


def test_selftest_dry_run_does_not_call_the_engine(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def refuse(progress: object) -> _Report:
        del progress
        raise AssertionError("runtime was called")

    monkeypatch.setattr("clarity.cli.ensure_shared_runtime", refuse)
    monkeypatch.setattr("clarity.cli.ensure_ffmpeg", refuse)
    assert main(["selftest", "--dry-run", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["gpu_used"] is False
    assert "shared runtime" in body["checks"]


def test_selftest_json_includes_the_engine_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def ensure(progress: object) -> _Report:
        assert callable(progress)
        progress("Checking python 3.14")
        return _Report(ok=True)

    def ffmpeg(progress: object) -> str:
        assert callable(progress)
        progress("Installing ffmpeg")
        return "repaired"

    monkeypatch.setattr("clarity.cli.ensure_shared_runtime", ensure)
    monkeypatch.setattr("clarity.cli.ensure_ffmpeg", ffmpeg)
    monkeypatch.setattr("clarity.cli._selftest_window", lambda: 0)
    monkeypatch.setattr("clarity.cli.gpu_stack_error", lambda: "")
    monkeypatch.setattr("clarity.cli.require_tensorrt_and_cuda", lambda: None)
    assert main(["selftest", "--json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    progress = [row for row in lines if row.get("type") == "progress"]
    assert any(row["message"] == "Installing ffmpeg" for row in progress)
    result = lines[-1]
    assert result["type"] == "result"
    assert result["ok"] is True
    assert result["runtime"]["ok"] is True


def test_release_tag_accepts_the_rc_build() -> None:
    version = json.loads((_ROOT / "BUILD.json").read_text(encoding="utf-8"))["version"]
    assert version == "1.0.0-rc"
    pattern = r"v\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?"
    shared = (_ROOT / ".github" / "workflows" / "suite-test.yml").read_text(encoding="utf-8")
    release = (_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert pattern in shared
    assert pattern in release
    assert "tag.startswith('v'+version)" in shared
    compiled = re.compile(pattern)
    assert compiled.fullmatch("v1.0.0-rc.1005")
    assert "v1.0.0-rc.1005".startswith("v" + version)
    assert compiled.fullmatch("v1.0.0")
    assert not "v1.0.0".startswith("v" + version)


def test_ci_and_release_share_one_selftest() -> None:
    root = _ROOT / ".github" / "workflows"
    shared = (root / "suite-test.yml").read_text(encoding="utf-8")
    ci = (root / "ci.yml").read_text(encoding="utf-8")
    release = (root / "release.yml").read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/suite-test.yml" in ci
    assert "uses: ./.github/workflows/suite-test.yml" in release
    assert "check_release_tag: false" in ci
    assert "check_release_tag: true" in release
    assert shared.count("python -m clarity --selftest --no-gpu") == 1
    assert "VOXPRINT_CI_LANCZOS" not in shared
    assert "ruff check clarity main.py tests scripts" in shared
    assert "if: inputs.check_release_tag" in shared
    for text in (ci, release):
        assert "python -m clarity --selftest" not in text
        assert "ruff check" not in text


def _ready_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    seen: dict[str, str] = {}

    def ensure(progress: object) -> _Report:
        assert callable(progress)
        seen["lanczos"] = os.environ.get("VOXPRINT_CI_LANCZOS", "")
        progress("Checking python 3.14")
        return _Report(ok=True)

    def ffmpeg(progress: object) -> str:
        assert callable(progress)
        return "skipped"

    monkeypatch.setattr("clarity.cli.ensure_shared_runtime", ensure)
    monkeypatch.setattr("clarity.cli.ensure_ffmpeg", ffmpeg)
    return seen


def test_no_gpu_selftest_checks_imports_window_and_plan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    seen = _ready_runtime(monkeypatch)
    opened = {"window": 0}

    def window() -> int:
        opened["window"] += 1
        return 0

    monkeypatch.setattr("clarity.cli._selftest_window", window)
    assert main(["selftest", "--no-gpu", "--json"]) == 0
    assert seen["lanczos"] == "1"
    assert os.environ.get("VOXPRINT_CI_LANCZOS") != "1"
    assert opened["window"] == 1
    result = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()][-1]
    assert result["ok"] is True
    assert result["no_gpu"] is True
    plan = result["plan"]
    assert plan["encoder"] == "av1_nvenc"
    assert plan["pix_fmt"] == "yuv420p"
    assert plan["chunks"] >= 1

    opened["window"] = 0
    assert main(["--selftest", "--no-gpu"]) == 0
    assert opened["window"] == 1
    assert "selftest ok" in capsys.readouterr().out


def test_gpu_selftest_opens_tensorrt_and_cuda_sessions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    seen = _ready_runtime(monkeypatch)
    opened = {"window": 0}
    probes: list[str] = []

    def window() -> int:
        opened["window"] += 1
        return 0

    monkeypatch.setattr("clarity.cli._selftest_window", window)
    monkeypatch.setattr("clarity.cli.gpu_stack_error", lambda: "")
    monkeypatch.setattr("clarity.cli.require_tensorrt_and_cuda", lambda: probes.append("session"))
    assert main(["selftest", "--json"]) == 0
    assert seen["lanczos"] == ""
    assert opened["window"] == 1
    assert probes == ["session"]
    result = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()][-1]
    assert result["ok"] is True
    assert result["no_gpu"] is False
    assert main(["selftest", "--no-gpu", "--json"]) == 0
    assert probes == ["session"]
    assert main(["--no-gpu"]) == 1
    assert "--no-gpu applies only to --selftest" in capsys.readouterr().err


def test_gpu_selftest_refuses_a_missing_torch_stack(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _ready_runtime(monkeypatch)

    def window() -> int:
        raise AssertionError("window")

    monkeypatch.setattr("clarity.cli._selftest_window", window)
    monkeypatch.setattr(
        "clarity.cli.gpu_stack_error",
        lambda: "PyTorch and spandrel are required. Install requirements-gpu.txt.",
    )
    assert main(["selftest", "--json"]) == 1
    assert "PyTorch and spandrel are required" in capsys.readouterr().err


def test_no_gpu_selftest_does_not_require_torch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _ready_runtime(monkeypatch)

    def window() -> int:
        return 0

    def boom() -> str:
        raise AssertionError("gpu stack")

    monkeypatch.setattr("clarity.cli._selftest_window", window)
    monkeypatch.setattr("clarity.cli.gpu_stack_error", boom)
    assert main(["selftest", "--no-gpu", "--json"]) == 0
    assert "gpu stack" not in capsys.readouterr().err


def test_selftest_stops_when_the_runtime_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def window() -> int:
        raise AssertionError("window")

    monkeypatch.setattr("clarity.cli.ensure_shared_runtime", lambda progress: _Report(ok=False))
    monkeypatch.setattr("clarity.cli.ensure_ffmpeg", lambda progress: "skipped")
    monkeypatch.setattr("clarity.cli._selftest_window", window)
    assert main(["selftest", "--json"]) == 1
    captured = capsys.readouterr()
    assert "incomplete" in captured.err


def test_install_suite_fetches_clarity_models(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def fetch(ids: list[str], dry_run: bool = False) -> list[object]:
        assert dry_run is False
        fetched.extend(ids)
        return []

    monkeypatch.setattr("clarity.cli.ensure_shared_runtime", lambda progress: _Report(ok=True))
    monkeypatch.setattr("clarity.cli.ensure_ffmpeg", lambda progress: "ready")
    monkeypatch.setattr("clarity.cli.fetch_weights", fetch)
    assert main(["--install-suite"]) == 0
    assert fetched == [
        "drunet-deblocking-color",
        "drunet-color",
        "scunet-color-real-psnr",
        "realesr-general-x4v3",
        "realesr-general-wdn-x4v3",
        "realesr-animevideov3",
    ]


def test_uninstall_releases_the_clarity_holder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    released: list[str] = []

    def release(progress: object) -> _Report:
        assert callable(progress)
        progress("Removing python 3.14")
        released.append("clarity")
        return _Report(ok=True)

    monkeypatch.setattr("clarity.cli.release_shared_runtime", release)
    monkeypatch.setattr("clarity.cli.unregister_user", lambda key="clarity": [])
    monkeypatch.setattr("clarity.cli.models_directory", lambda: tmp_path)
    assert main(["--unregister-models-user"]) == 0
    assert released == ["clarity"]


def test_old_driver_includes_the_nvidia_download_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    old = [
        GpuDevice(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            driver="551.76",
            compute=(8, 9),
            memory_mib=24564,
        )
    ]
    monkeypatch.setattr("clarity.engine._engine_api", object())
    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", lambda runner=None: old)
    gate = assess_launch()
    assert gate.ok is False
    assert "551.76" in gate.message
    assert "R600" in gate.message
    assert NVIDIA_DRIVER_URL in gate.message
    assert NVIDIA_DRIVER_URL not in old_driver_message("Clarity", "551.76")
    assert NVIDIA_DRIVER_URL in driver_refusal("551.76")


def test_launch_probe_uses_the_clarity_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def assess(app: str, *, refuse_below: object = None, **_kwargs: object) -> object:
        seen["app"] = app
        seen["refuse"] = refuse_below
        return type("Probe", (), {"ok": True, "message": "", "warning": ""})()

    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.setattr("clarity.engine._engine_api", object())
    monkeypatch.setattr("clarity.engine.assess_gpu", assess)
    gate = assess_launch()
    assert gate.ok is True
    assert seen["app"] == APP_NAME == "VOXPrint AI Clarity"
    assert seen["refuse"] == (8, 9)


def test_parent_lock_is_shared_only_when_the_app_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "voxprint-gpu.lock"
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(path))
    moment = datetime.now(timezone.utc)
    payload = {
        "owner": OWNER,
        "pid": os.getppid(),
        "job": "clarity:restore",
        "started": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eta": (moment + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    borrowed = try_acquire(OWNER, "restore", path=path)
    assert isinstance(borrowed, Acquired)
    assert borrowed.borrowed is True
    holder = try_acquire("enlarge", "restore", path=path)
    assert isinstance(holder, Holder)
    assert holder.app == OWNER
    assert path.is_file()


def test_clarity_lock_records_the_app_and_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "voxprint-gpu.lock"
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(path))
    with hold_gpu("clarity:restore", eta_s=60):
        recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded["app"] == OWNER
    assert recorded["stage"] == "clarity:restore"
    assert not path.is_file()


def test_repair_dialog_opens_only_while_installing() -> None:
    app = QApplication.instance() or QApplication([])
    progress = RepairProgress()
    progress("Checking python 3.14")
    assert progress.dialog is None
    progress("Installing ffmpeg")
    assert progress.dialog is not None
    assert "ffmpeg" in progress.dialog.labelText()
    progress.dialog.close()
    app.processEvents()


def test_ffmpeg_repairs_once_and_skips_in_ci(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.setattr("clarity.ffmpeg_setup.shared_directory", lambda: tmp_path)
    monkeypatch.setattr("clarity.ffmpeg_setup.shutil.which", lambda name: None)
    calls: list[str] = []

    def install(progress: object) -> Path:
        del progress
        calls.append("install")
        binary = tmp_path / "ffmpeg" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("ffmpeg", encoding="utf-8")
        return binary

    monkeypatch.setattr("clarity.ffmpeg_setup.install_ffmpeg", install)
    messages: list[str] = []
    assert ensure_ffmpeg(messages.append) == "repaired"
    assert "Installing ffmpeg" in messages
    assert ensure_ffmpeg(messages.append) == "ready"
    assert calls == ["install"]

    def shared() -> Path:
        raise AssertionError("shared_directory")

    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")
    monkeypatch.setattr("clarity.ffmpeg_setup.shared_directory", shared)
    assert ensure_ffmpeg(messages.append) == "skipped"


def test_engine_pin_and_pillow_stay() -> None:
    assert ENGINE_TAG == "engine-v1.4.0"
    assert SUITE_RUNTIME == "enlarge.suite_runtime"
    assert ENGINE_REQUIREMENT == (
        "voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge"
        "/archive/refs/tags/engine-v1.4.0.tar.gz"
    )
    assert "git+" not in ENGINE_REQUIREMENT
    assert "#subdirectory" not in ENGINE_REQUIREMENT
    assert "[gpu]" in ENGINE_REQUIREMENT
    engine = (_ROOT / "clarity" / "engine.py").read_text(encoding="utf-8")
    windows = (_ROOT / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    linux = (_ROOT / "installer" / "linux" / "install-shared.sh").read_text(encoding="utf-8")
    package = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (_ROOT / ".github" / "workflows" / "suite-test.yml").read_text(encoding="utf-8")
    vm = (_ROOT / "scripts" / "vm_test.py").read_text(encoding="utf-8")
    shared = (_ROOT / "scripts" / "gpu-common.ps1").read_text(encoding="utf-8")
    iss = (_ROOT / "installer" / "VoxprintClarity.iss").read_text(encoding="utf-8")
    lock = json.loads((_ROOT / "installer" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert lock["pillow"] == "12.3.0"
    assert lock["engine"] == ENGINE_REQUIREMENT
    common = (
        "voxprint-common @ https://github.com/Mitroshenkov87/voxprint-common"
        "/releases/download/v0.1.1/voxprint_common-0.1.1-py3-none-any.whl"
    )
    app_common = (
        "voxprint-common[ui] @ git+https://github.com/Mitroshenkov87/voxprint-common"
        "@8a94b531cb4f3526425a80a86a1e64624b5de606"
    )
    requirements = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert lock["voxprint-common"] == "0.1.1"
    assert lock["voxprint-common_requirement"] == common
    assert lock["voxprint-common_sha256"] == "c9361d5d00913faf39497fc55124e682ba15b2098c775068c49582c0520cd4d1"
    assert lock["voxprint-common_sha256_url"] == common.split(" @ ", 1)[1] + ".sha256"
    assert common in windows
    assert common in linux
    assert common not in requirements
    assert common not in package
    assert app_common in requirements
    assert app_common in package
    assert "Shared UI tag v0.2.5" in requirements
    assert "git+" not in common
    assert "MinGit" not in common
    assert 'Pillow==$pillow' in windows
    assert "probes=" not in windows
    assert "delete_weight" not in windows
    assert 'ensure_runtime(\n        "clarity"' in windows
    assert "fetch_common.py" in windows
    assert "stage_suite_runtime.py" in windows
    assert "fetch_common.py" in linux
    assert "stage_suite_runtime.py" in linux
    assert "stage_suite_runtime.py" in workflow
    assert "enlarge.suite_runtime" in workflow
    assert "ExtractTemporaryFile('fetch_common.py')" in iss
    assert "ExtractTemporaryFile('stage_suite_runtime.py')" in iss
    assert ENGINE_REQUIREMENT in windows
    assert ENGINE_REQUIREMENT in linux
    assert ENGINE_REQUIREMENT in package
    assert ENGINE_REQUIREMENT in workflow
    assert ENGINE_REQUIREMENT in shared
    assert "pypi.nvidia.com" in windows
    assert "pypi.nvidia.com" in linux
    assert "pypi.nvidia.com" in workflow
    assert "pypi.nvidia.com" in vm
    assert "pypi.nvidia.com" in shared
    assert 'ENGINE_TAG = "engine-v1.4.0"' in engine
    assert "engine-v1.4.0.tar.gz" in engine
    assert "switch to tag" not in engine
    assert "engine-v1.2.1" not in engine
    assert "python install 3.14" in linux
    assert "Enlarge bundle" not in linux
    assert "MinGit" not in windows
    assert "Pillow==12.3.0" in linux
    assert "--install-suite" in linux
    for text in (windows, linux, package, workflow, engine):
        assert "d11d609" not in text
        assert "9183489" not in text
