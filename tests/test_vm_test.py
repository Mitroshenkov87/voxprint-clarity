"""Contract for the rented-GPU VM scripts. No GPU and no downloads."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _driver():
    path = _ROOT / "scripts" / "vm_test.py"
    spec = importlib.util.spec_from_file_location("clarity_vm_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/vm_test.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ref_label_uses_the_tag_or_the_commit() -> None:
    module = _driver()
    assert module.ref_label("", "b01cd6c") == "main-b01cd6c"
    assert module.ref_label("v1.0.0-rc.1002", "b01cd6c") == "v1.0.0-rc.1002"
    with pytest.raises(ValueError):
        module.ref_label("", "abc")


def test_source_ref_uses_the_tag_or_main_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _driver()

    def fake_run(cmd: list[str], **_kwargs: object) -> object:
        if "describe" in cmd:
            return module.subprocess.CompletedProcess(cmd, 128, "", "not a tag")
        return module.subprocess.CompletedProcess(cmd, 0, "b01cd6c\n", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.source_ref(_ROOT) == "main-b01cd6c"

    def tagged(cmd: list[str], **_kwargs: object) -> object:
        if "describe" in cmd:
            return module.subprocess.CompletedProcess(cmd, 0, "v1.0.0-rc.1002\n", "")
        return module.subprocess.CompletedProcess(cmd, 0, "b01cd6c\n", "")

    monkeypatch.setattr(module.subprocess, "run", tagged)
    assert module.source_ref(_ROOT) == "v1.0.0-rc.1002"

    def missing(cmd: list[str], **_kwargs: object) -> object:
        return module.subprocess.CompletedProcess(cmd, 1, "", "no git")

    monkeypatch.setattr(module.subprocess, "run", missing)
    with pytest.raises(module._VmError):
        module.source_ref(_ROOT)


def test_markers_live_under_done_and_gpu_is_named() -> None:
    module = _driver()
    path = module.marker_path("v1.0.0-rc.1002", "restore-clean")
    assert path == Path.home() / ".done" / "v1.0.0-rc.1002--restore-clean"


def test_lgpl_build_keeps_disabled_libraries() -> None:
    module = _driver()
    lgpl = "configuration: --disable-gpl --disable-libx264 --disable-libx265"
    gpl = "configuration: --enable-gpl --enable-libx264"
    assert module.ffmpeg_build_is_gpl(lgpl) is False
    assert module.ffmpeg_build_is_gpl(gpl) is True


def test_forced_failure_requires_json_and_exit_3() -> None:
    module = _driver()
    stdout = '{"type":"result","ok":false,"error":"the source video was not found"}\n'
    assert module.forced_failure_error(3, stdout) == ""
    assert module.forced_failure_error(1, stdout)
    assert module.forced_failure_error(3, "restore failed\n")


def test_trt_stage_is_read_from_progress_json() -> None:
    module = _driver()
    stdout = '{"type":"progress","fraction":0.0,"message":"Building the TensorRT engine.","stage":"building_engine"}\n'
    assert module.trt_stage_seen(stdout) is True
    assert module.trt_stage_seen('{"type":"result","ok":true}\n') is False


def test_gpu_csv_rejects_an_old_driver_and_accepts_a_4090() -> None:
    module = _driver()
    old = module.parse_gpu_csv("NVIDIA GeForce RTX 4090, 551.86, 8.9", allow_other=False)
    assert old.ok is False
    good = module.parse_gpu_csv("NVIDIA GeForce RTX 4090, 610.88, 8.9", allow_other=False)
    assert good.ok is True
    other = module.parse_gpu_csv("NVIDIA GeForce RTX 4080, 610.88, 8.9", allow_other=False)
    assert other.ok is False
    allowed = module.parse_gpu_csv("NVIDIA GeForce RTX 4080, 610.88, 8.9", allow_other=True)
    assert allowed.ok is True
    low = module.parse_gpu_csv("NVIDIA GeForce RTX 4090, 610.88, 8.6", allow_other=True)
    assert low.ok is False


def test_smoke_and_full_steps_cover_the_plan() -> None:
    module = _driver()
    assert module.SMOKE_STEPS == (
        "gpu",
        "install",
        "selftest",
        "info",
        "clip",
        "restore-clean",
        "restore-2x",
        "bounds",
        "forced-failure",
    )
    assert "case-a" in module.FULL_STEPS
    assert "case-b" in module.FULL_STEPS
    assert "case-c" in module.FULL_STEPS
    assert "bench" in module.FULL_STEPS
    assert module.SMOKE_SECONDS == 15
    assert 10 <= module.SMOKE_SECONDS <= 20
    assert module.RAM_BOUND_MIB == 8192
    assert module.VRAM_BOUND_MIB == 16384


def test_clip_hash_matches_the_benchmark() -> None:
    module = _driver()
    bench = (_ROOT / "clarity" / "bench.py").read_text(encoding="utf-8")
    assert module.CLIP_SHA256 in bench
    assert module.CLIP_URL in bench
    assert module.ENGINE_COMMIT == "9183489"


def test_entries_share_flags_and_exit_codes() -> None:
    names = ("vm_test.sh", "vm_test.ps1", "vm_test.py")
    texts = {name: (_ROOT / "scripts" / name).read_text(encoding="utf-8") for name in names}
    for text in texts.values():
        assert "--smoke" in text
        assert "--full" in text
        assert "--json" in text
        assert "--allow-other-gpu" in text
    assert "3 no GPU" in texts["vm_test.sh"]
    assert "fail 2 bootstrap" in texts["vm_test.sh"]
    assert 'Write-VmFailure 3 "gpu"' in texts["vm_test.ps1"]
    assert "EXIT_NO_GPU = 3" in texts["vm_test.py"]
    assert "gpu-common.ps1" in texts["vm_test.ps1"]
    shared = (_ROOT / "scripts" / "gpu-common.ps1").read_text(encoding="utf-8")
    assert "9183489" in shared
    assert "9183489" in texts["vm_test.py"]
    assert "--enable-gpl" in texts["vm_test.py"]


def test_smoke_without_a_gpu_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _driver()
    monkeypatch.setattr(module, "probe_gpu", lambda *, allow_other: module.GpuStatus(False, "nvidia-smi was not found"))
    code = module.main(["--smoke", "--json", "--results", str(tmp_path)])
    assert code == 3
    captured = capsys.readouterr()
    assert "FAILED: step gpu:" in captured.err
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert summary["exit"] == 3
    assert summary["step"] == "gpu"
    assert summary["app"] == "clarity"
    assert summary["mode"] == "smoke"
    assert summary["type"] == "summary"
    stdout = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert stdout[-1]["step"] == "gpu"
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "crops").is_dir()


def test_missing_mode_is_setup(capsys: pytest.CaptureFixture[str]) -> None:
    module = _driver()
    code = module.main(["--json"])
    assert code == 2
    captured = capsys.readouterr()
    assert "FAILED: step args:" in captured.err
    body = json.loads(captured.out.strip())
    assert body["exit"] == 2
    assert body["step"] == "args"


def test_finished_marker_skips_the_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _driver()
    monkeypatch.setattr(module, "_locate_ffmpeg", lambda _run: None)
    monkeypatch.setattr(module, "_clarity_imports", lambda: True)

    def marker(tag: str, step: str) -> Path:
        return tmp_path / "done" / f"{tag}--{step}"

    monkeypatch.setattr(module, "marker_path", marker)
    called: list[str] = []

    def handlers() -> dict:
        return {name: (lambda _run, name=name: called.append(name)) for name in module.SMOKE_STEPS}

    monkeypatch.setattr(module, "_handlers", handlers)
    done = marker("vtest", "info")
    done.parent.mkdir(parents=True)
    done.write_text("ok\n", encoding="utf-8")
    run = module._Run(
        root=_ROOT,
        results=tmp_path / "results",
        mode="smoke",
        tag="vtest",
        json_mode=False,
        allow_other_gpu=False,
        allow_other_os=False,
        logs=tmp_path / "results" / "logs",
        crops=tmp_path / "results" / "crops",
        cache=tmp_path / "results" / "cache",
        clips=tmp_path / "results" / "clips",
        outputs=tmp_path / "results" / "outputs",
        tools=tmp_path / "results" / "tools",
    )
    run.results.mkdir()
    code = module._run_steps(run)
    assert code == 0
    assert "info" not in called
    assert "restore-clean" in called
    skipped = [item for item in run.steps if item["name"] == "info"]
    assert skipped == [{"name": "info", "status": "skip", "seconds": 0.0, "error": ""}]
    assert marker("vtest", "restore-clean").is_file()


def test_install_seconds_include_the_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _driver()
    monkeypatch.setattr(module, "_locate_ffmpeg", lambda _run: None)
    monkeypatch.setattr(module, "_clarity_imports", lambda: True)
    monkeypatch.setattr(module, "_check_providers", lambda _run, _step: None)

    def marker(tag: str, step: str) -> Path:
        return tmp_path / "done" / f"{tag}--{step}"

    monkeypatch.setattr(module, "marker_path", marker)

    def handlers() -> dict:
        names = {name: (lambda _run, name=name: None) for name in module.SMOKE_STEPS}
        names["install"] = module._step_install
        return names

    monkeypatch.setattr(module, "_handlers", handlers)
    logs = tmp_path / "results" / "logs"
    logs.mkdir(parents=True)
    (logs / "install-seconds.json").write_text(json.dumps({"seconds": 12.5}) + "\n", encoding="utf-8")
    run = module._Run(
        root=_ROOT,
        results=tmp_path / "results",
        mode="smoke",
        tag="main-b01cd6c",
        json_mode=False,
        allow_other_gpu=False,
        allow_other_os=False,
        logs=logs,
        crops=tmp_path / "results" / "crops",
        cache=tmp_path / "results" / "cache",
        clips=tmp_path / "results" / "clips",
        outputs=tmp_path / "results" / "outputs",
        tools=tmp_path / "results" / "tools",
    )
    code = module._run_steps(run)
    assert code == 0
    install = next(item for item in run.steps if item["name"] == "install")
    assert install["status"] == "pass"
    assert install["seconds"] >= 12.5
    assert not (logs / "install-seconds.json").is_file()
    assert marker("main-b01cd6c", "install").is_file()


def test_provider_failure_does_not_keep_the_install_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _driver()
    monkeypatch.setattr(module, "_locate_ffmpeg", lambda _run: None)
    monkeypatch.setattr(module, "_clarity_imports", lambda: True)

    def fail(_run: object, _step: str) -> None:
        raise module._VmError("missing TensorrtExecutionProvider", module.EXIT_FAIL)

    monkeypatch.setattr(module, "_check_providers", fail)

    def marker(tag: str, step: str) -> Path:
        return tmp_path / "done" / f"{tag}--{step}"

    monkeypatch.setattr(module, "marker_path", marker)
    stale = marker("main-b01cd6c", "install")
    stale.parent.mkdir(parents=True)
    stale.write_text("ok\n", encoding="utf-8")

    def handlers() -> dict:
        names = {name: (lambda _run, name=name: None) for name in module.SMOKE_STEPS}
        names["install"] = module._step_install
        return names

    monkeypatch.setattr(module, "_handlers", handlers)
    logs = tmp_path / "results" / "logs"
    logs.mkdir(parents=True)
    run = module._Run(
        root=_ROOT,
        results=tmp_path / "results",
        mode="smoke",
        tag="main-b01cd6c",
        json_mode=True,
        allow_other_gpu=False,
        allow_other_os=False,
        logs=logs,
        crops=tmp_path / "results" / "crops",
        cache=tmp_path / "results" / "cache",
        clips=tmp_path / "results" / "clips",
        outputs=tmp_path / "results" / "outputs",
        tools=tmp_path / "results" / "tools",
    )
    code = module._run_steps(run)
    assert code == 1
    assert not stale.is_file()
    failed = next(item for item in run.steps if item["name"] == "install")
    assert failed["status"] == "fail"
    assert "TensorrtExecutionProvider" in failed["error"]
