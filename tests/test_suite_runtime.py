"""Engine holder, ffmpeg repair, and the driver link. No GPU and no downloads."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from clarity.cli import _ensure_shared, main
from clarity.engine import (
    ENGINE_REQUIREMENT,
    ENGINE_TAG,
    NVIDIA_DRIVER_URL,
    EngineUnavailable,
    assess_launch,
    driver_refusal,
    ensure_shared_runtime,
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
        assert name == "voxprint_engine.suite_runtime"
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
    with pytest.raises(EngineUnavailable, match="Voxprint inference engine missing"):
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


def test_clarity_drops_a_sole_hold_on_the_default_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = {
        "schema": 1,
        "components": {"model": {"holders": ["clarity"], "version": "4x-nomos2-realplksr-dysample"}},
    }
    deleted: list[tuple[str, str]] = []
    written: dict[str, object] = {}

    class _Model:
        id = "4x-nomos2-realplksr-dysample"

    class _Store:
        class StoreError(RuntimeError):
            pass

        @staticmethod
        def delete_weight(model: _Model, root: object = None, *, user: str = "enlarge") -> None:
            del root
            deleted.append((model.id, user))

    class _Registry:
        DEFAULT_MODEL_ID = "4x-nomos2-realplksr-dysample"

        @staticmethod
        def load_builtin(*, release: bool) -> dict[str, _Model]:
            assert release is True
            return {_Registry.DEFAULT_MODEL_ID: _Model()}

    class _Module:
        SuiteInstallError = _SuiteInstallError

        @staticmethod
        def shared_dir() -> Path:
            return tmp_path

        @staticmethod
        def read_manifest() -> dict[str, object]:
            return manifest

        @staticmethod
        def _write_manifest(doc: dict[str, object], root: Path) -> None:
            written["doc"] = doc
            written["root"] = root

        @staticmethod
        def ensure_runtime(app: str, progress: object = None, **extra: object) -> _Report:
            assert app == "clarity"
            probes = extra["probes"]
            installers = extra["installers"]
            assert isinstance(probes, dict)
            assert probes["model"](tmp_path) is True
            installers["model"](tmp_path, progress or (lambda _message: None))
            return _Report(ok=True)

    def load(name: str) -> object:
        if name == "voxprint_engine.suite_runtime":
            return _Module()
        if name == "voxprint_engine.store":
            return _Store()
        if name == "voxprint_engine.registry":
            return _Registry()
        if name == "voxprint_engine.onnx_export":
            return type("Onnx", (), {"cached_onnx": staticmethod(lambda _model: None)})()
        return None

    monkeypatch.setattr("clarity.engine._load_module", load)
    report = ensure_shared_runtime()
    assert report.ok is True
    components = manifest["components"]
    assert isinstance(components, dict)
    assert "model" not in components
    assert written["root"] == tmp_path
    assert deleted == [("4x-nomos2-realplksr-dysample", "clarity")]


def test_clarity_keeps_another_apps_default_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = {
        "schema": 1,
        "components": {"model": {"holders": ["enlarge", "clarity"], "version": "4x-nomos2-realplksr-dysample"}},
    }

    class _Module:
        SuiteInstallError = _SuiteInstallError

        @staticmethod
        def shared_dir() -> Path:
            return tmp_path

        @staticmethod
        def read_manifest() -> dict[str, object]:
            return manifest

        @staticmethod
        def _write_manifest(doc: dict[str, object], root: Path) -> None:
            del doc, root

        @staticmethod
        def ensure_runtime(app: str, progress: object = None, **_extra: object) -> _Report:
            del app, progress, _extra
            return _Report(ok=True)

    def load(name: str) -> object | None:
        if name == "voxprint_engine.suite_runtime":
            return _Module()
        if name == "voxprint_engine.store":
            raise AssertionError("weight delete")
        return None

    monkeypatch.setattr("clarity.engine._load_module", load)
    ensure_shared_runtime()
    components = manifest["components"]
    assert isinstance(components, dict)
    entry = components["model"]
    assert isinstance(entry, dict)
    assert entry["holders"] == ["enlarge"]


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
    assert main(["selftest", "--json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    progress = [row for row in lines if row.get("type") == "progress"]
    assert any(row["message"] == "Installing ffmpeg" for row in progress)
    result = lines[-1]
    assert result["type"] == "result"
    assert result["ok"] is True
    assert result["runtime"]["ok"] is True


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


def test_selftest_leaves_execution_providers_to_the_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    seen = _ready_runtime(monkeypatch)
    opened = {"window": 0}

    def window() -> int:
        opened["window"] += 1
        return 0

    monkeypatch.setattr("clarity.cli._selftest_window", window)
    monkeypatch.setattr("clarity.cli.gpu_stack_error", lambda: "")
    assert main(["selftest", "--json"]) == 0
    assert seen["lanczos"] == ""
    assert opened["window"] == 1
    result = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()][-1]
    assert result["ok"] is True
    assert result["no_gpu"] is False
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

    class _Gate:
        def __init__(self) -> None:
            self.ok = False
            self.message = (
                "Enlarge needs an NVIDIA driver from the 600 series or newer. "
                "This computer has driver 551.76. Update the driver, then start Enlarge again."
            )
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
    assert NVIDIA_DRIVER_URL not in _Gate().message
    assert NVIDIA_DRIVER_URL in driver_refusal("551.76")


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
    assert ENGINE_TAG == "engine-v1.2.1"
    assert ENGINE_REQUIREMENT == (
        "voxprint-engine @ https://github.com/Mitroshenkov87/voxprint-enlarge"
        "/archive/refs/tags/engine-v1.2.1.tar.gz"
    )
    assert "git+" not in ENGINE_REQUIREMENT
    assert "#subdirectory" not in ENGINE_REQUIREMENT
    engine = (_ROOT / "clarity" / "engine.py").read_text(encoding="utf-8")
    windows = (_ROOT / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    linux = (_ROOT / "installer" / "linux" / "install-shared.sh").read_text(encoding="utf-8")
    package = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (_ROOT / ".github" / "workflows" / "suite-test.yml").read_text(encoding="utf-8")
    lock = json.loads((_ROOT / "installer" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert lock["pillow"] == "12.3.0"
    assert 'Pillow==$pillow' in windows
    assert 'probes={"model": lambda _root: True}' in windows
    assert 'installers={"model": lambda _root, _progress: None}' in windows
    assert 'delete_weight(model, user="clarity")' in windows
    assert ENGINE_REQUIREMENT in windows
    assert ENGINE_REQUIREMENT in linux
    assert ENGINE_REQUIREMENT in package
    assert ENGINE_REQUIREMENT in workflow
    assert 'ENGINE_TAG = "engine-v1.2.1"' in engine
    assert "engine-v1.2.1.tar.gz" in engine
    assert "switch to tag" not in engine
    assert "python install 3.14" in linux
    assert "Enlarge bundle" not in linux
    assert "MinGit" not in windows
    assert "Pillow==12.3.0" in linux
    assert "--install-suite" in linux
    for text in (windows, linux, package, workflow, engine):
        assert "d11d609" not in text
        assert "9183489" not in text
