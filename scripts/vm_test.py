"""Resumable GPU VM test for Clarity.

``scripts/vm_test.sh`` and ``scripts/vm_test.ps1`` are the entries. Both call
this driver. Exit 0 is a pass, 1 is a test failure, 2 is setup, and 3 means
the machine has no usable GPU. ``--json`` writes one JSON object per line on
stdout. See docs/TESTING-GPU.md.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from ctypes import wintypes
from collections.abc import Callable
from dataclasses import dataclass, field
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any, Never

try:
    import truststore
except ImportError:
    truststore = None

try:
    import winreg
except ImportError:
    winreg = None

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SETUP = 2
EXIT_NO_GPU = 3

RAM_BOUND_MIB = 8192
VRAM_BOUND_MIB = 16384
SMOKE_SECONDS = 15
ENGINE_TAG = "engine-v1.4.0"
ENGINE_URL = (
    "voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge"
    "/archive/refs/tags/engine-v1.4.0.tar.gz"
)
NVIDIA_INDEX = "https://pypi.nvidia.com"
CLIP_URL = "https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4"
CLIP_SHA256 = "fd6dc0e9da3a2f63487957325983f197f23c6ff1cd2b29bb71e64db567e0994c"
CLIP_NAME = "elephants-dream-ed_1024_512kb.mp4"
CLIP_CREDIT = (
    "Elephants Dream (c) copyright Blender Foundation / Netherlands Media Art Institute. "
    "Creative Commons Attribution."
)
FFMPEG_URL = {
    "win32": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip",
    "linux": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl.tar.xz",
}
UV_URL = {
    "win32": "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip",
    "linux": "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz",
}
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TRT_STAGES = frozenset({"building_engine", "loading_cached_engine"})
SMOKE_STEPS = (
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
FULL_STEPS = (
    "gpu",
    "install",
    "selftest",
    "info",
    "clips",
    "case-a",
    "case-b",
    "case-c",
    "bounds",
    "bench",
)
_TRUST_READY = False


class _VmError(Exception):
    """A named step failed. ``code`` is the process status."""

    def __init__(self, message: str, code: int, step: str = "") -> None:
        """Store the process status and the step that failed.

        Args:
            message: Shown after ``FAILED: step``.
            code: 1 test failure, 2 setup, or 3 no GPU.
            step: Step name when the failure happens outside the step loop.
        """
        super().__init__(message)
        self.code = code
        self.step = step


@dataclass(frozen=True)
class GpuStatus:
    """What nvidia-smi reported, or why this machine cannot run the VM test."""

    ok: bool
    message: str
    name: str = ""
    driver: str = ""
    compute: str = ""


@dataclass
class _Run:
    """Paths and flags for one invocation."""

    root: Path
    results: Path
    mode: str
    tag: str
    json_mode: bool
    allow_other_gpu: bool
    allow_other_os: bool
    logs: Path
    crops: Path
    cache: Path
    clips: Path
    outputs: Path
    tools: Path
    ffmpeg: str = ""
    ffprobe: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    bounds: dict[str, float] = field(default_factory=dict)
    install_seconds: float = 0.0


def ref_label(exact_tag: str, sha: str) -> str:
    """Return the results folder name for a checkout.

    An exact git tag is kept. Any other commit is ``main-`` plus the short
    sha. ``BUILD.json`` is not a ref: two checkouts can share a build number.

    Args:
        exact_tag: ``git describe --exact-match --tags``, or empty.
        sha: ``git rev-parse --short=7 HEAD``.

    Returns:
        The tag, or ``main-`` plus the hex sha.

    Raises:
        ValueError: When an untagged sha is not at least 7 hex characters.
    """
    tag = exact_tag.strip()
    if tag:
        if any(char.isspace() for char in tag):
            raise ValueError("the tag must be one token")
        return tag
    short = sha.strip().lower()
    if len(short) < 7 or any(char not in "0123456789abcdef" for char in short):
        raise ValueError("sha must be hex")
    return f"main-{short}"


def source_ref(root: Path) -> str:
    """Name HEAD as a tag, or ``main-`` plus the short sha.

    Args:
        root: The Clarity checkout.

    Returns:
        The label stored in ``summary.json`` and the done-marker key.

    Raises:
        _VmError: When git cannot name HEAD.
    """
    described = subprocess.run(
        ["git", "-C", str(root), "describe", "--exact-match", "--tags", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    exact = (described.stdout or "").strip() if described.returncode == 0 else ""
    parsed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    sha = (parsed.stdout or "").strip()
    if parsed.returncode != 0 or not sha:
        raise _VmError("git could not name HEAD", EXIT_SETUP, "setup")
    try:
        return ref_label(exact, sha)
    except ValueError as exc:
        raise _VmError(str(exc), EXIT_SETUP, "setup") from exc


def marker_path(tag: str, step: str) -> Path:
    """Return the resume marker for one step of one tag.

    Args:
        tag: Results folder name from :func:`source_ref`.
        step: Step name such as ``restore-clean``.

    Returns:
        ``~/.done/<tag>--<step>``. ``gpu`` is never skipped.
    """
    return Path.home() / ".done" / f"{tag}--{step}"


def ffmpeg_build_is_gpl(buildconf: str) -> bool:
    """Return whether an ffmpeg build enables GPL or the GPL H.264/H.265 libraries.

    ``--disable-libx264`` is the LGPL build and is kept.

    Args:
        buildconf: Text from ``ffmpeg -buildconf``.

    Returns:
        True when the build is rejected.
    """
    return "--enable-gpl" in buildconf or "--enable-libx264" in buildconf or "--enable-libx265" in buildconf


def forced_failure_error(returncode: int, stdout: str) -> str:
    """Return why a forced job failure missed the JSON contract.

    Args:
        returncode: Process status from ``python -m clarity restore``.
        stdout: Captured stdout.

    Returns:
        An empty string when a result line has ``ok`` false and the status is 3.
    """
    if returncode != 3:
        return f"expected exit 3, got {returncode}"
    for line in stdout.splitlines():
        body = _json_line(line)
        if body is None:
            continue
        if body.get("type") == "result" and body.get("ok") is False and body.get("error"):
            return ""
    return "stdout had no result object with ok false"


def trt_stage_seen(stdout: str) -> bool:
    """Return whether restore progress reported a TensorRT engine stage.

    Args:
        stdout: JSON lines from ``clarity restore --json``.

    Returns:
        True when a progress line names ``building_engine`` or ``loading_cached_engine``.
    """
    for line in stdout.splitlines():
        body = _json_line(line)
        if body is None or body.get("type") != "progress":
            continue
        if body.get("stage") in TRT_STAGES:
            return True
    return False


def parse_gpu_csv(line: str, *, allow_other: bool) -> GpuStatus:
    """Judge one ``nvidia-smi`` CSV line.

    Args:
        line: ``name, driver_version, compute_cap`` for the first GPU.
        allow_other: Accept a non-4090 RTX 40 series card.

    Returns:
        A status. ``ok`` is false when the driver, compute capability, or name fails.
    """
    cells = [part.strip() for part in line.split(",")]
    if len(cells) < 3:
        return GpuStatus(False, f"nvidia-smi returned {line!r}, which is not name, driver, compute capability.")
    driver = cells[-2]
    compute = cells[-1]
    name = ", ".join(cells[:-2])
    major = _leading_int(driver)
    if major < 600:
        return GpuStatus(False, f"NVIDIA driver {driver} is older than the 600 series.", name, driver, compute)
    cap_major, cap_minor = _compute_pair(compute)
    if cap_major < 8 or (cap_major == 8 and cap_minor < 9):
        return GpuStatus(False, f"GPU compute capability {compute} is below 8.9.", name, driver, compute)
    if "4090" not in name and not allow_other:
        return GpuStatus(
            False,
            f"Expected an RTX 4090 and found {name!r}. Pass --allow-other-gpu to run on another RTX 40 series card.",
            name,
            driver,
            compute,
        )
    return GpuStatus(True, "", name, driver, compute)


def probe_gpu(*, allow_other: bool) -> GpuStatus:
    """Query the first GPU. A missing ``nvidia-smi`` is not a usable GPU.

    Args:
        allow_other: Accept a non-4090 RTX 40 series card.

    Returns:
        The parsed status. ``ok`` is false with a message when the machine cannot run.
    """
    if shutil.which("nvidia-smi") is None:
        return GpuStatus(False, "nvidia-smi was not found. Install an NVIDIA driver from the 600 series or newer.")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    line = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not line:
        return GpuStatus(False, "nvidia-smi did not report a GPU.")
    return parse_gpu_csv(line[0], allow_other=allow_other)


def main(argv: list[str] | None = None) -> int:
    """Run ``--smoke`` or ``--full`` and return the process status.

    Args:
        argv: Flags, without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        0 pass, 1 test failure, 2 setup, 3 no GPU.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    _install_trust()
    os.environ.pop("VOXPRINT_CI_LANCZOS", None)
    os.environ.pop("PYTHONPATH", None)
    try:
        parsed = _parse(args)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_SETUP
    try:
        run = _make_run(parsed)
    except _VmError as exc:
        print(f"FAILED: step {exc.step or 'setup'}: {exc}", file=sys.stderr)
        return exc.code
    try:
        _check_os(run)
        status = probe_gpu(allow_other=run.allow_other_gpu)
        if not status.ok:
            return _finish(run, EXIT_NO_GPU, "gpu", status.message)
        if not _in_venv(run.results):
            if not _install_ready(run):
                try:
                    _install(run)
                except _VmError as exc:
                    return _finish(run, exc.code, "install", str(exc))
            return _reexec(run, args)
        return _run_steps(run)
    except _VmError as exc:
        return _finish(run, exc.code, exc.step or "setup", str(exc))


def _parse(args: list[str]) -> argparse.Namespace:
    parser = _Parser(prog="vm_test")
    parser.json_mode = "--json" in args
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="About 10 minutes. Required before an RC tag.")
    mode.add_argument("--full", action="store_true", help="Lean plan cases A, B, and C, then the benchmark.")
    parser.add_argument("--json", action="store_true", help="Write JSON lines on stdout.")
    parser.add_argument("--results", type=Path, default=None, help="Results directory. Layout matches vm-results/clarity/<tag>/.")
    parser.add_argument("--allow-other-gpu", action="store_true", help="Accept another RTX 40 series GPU.")
    parser.add_argument("--allow-other-os", action="store_true", help="Accept a Windows caption other than Server 2025.")
    return parser.parse_args(args)


class _Parser(argparse.ArgumentParser):
    """Usage errors are setup failures and can be JSON."""

    def error(self, message: str) -> Never:
        """Print a setup failure and exit 2.

        Args:
            message: The argparse error text.
        """
        print(f"FAILED: step args: {message}", file=sys.stderr)
        if getattr(self, "json_mode", False):
            print(
                json.dumps(
                    {"type": "summary", "app": "clarity", "ok": False, "exit": EXIT_SETUP, "step": "args", "error": message}
                ),
                flush=True,
            )
        raise SystemExit(EXIT_SETUP)


def _make_run(parsed: argparse.Namespace) -> _Run:
    root = Path(__file__).resolve().parents[1]
    tag = source_ref(root)
    results = parsed.results if parsed.results is not None else _default_results(root, tag)
    results = results.resolve()
    for folder in ("logs", "crops", "cache", "clips", "outputs", "tools"):
        (results / folder).mkdir(parents=True, exist_ok=True)
    return _Run(
        root=root,
        results=results,
        mode="smoke" if parsed.smoke else "full",
        tag=tag,
        json_mode=bool(parsed.json),
        allow_other_gpu=bool(parsed.allow_other_gpu),
        allow_other_os=bool(parsed.allow_other_os),
        logs=results / "logs",
        crops=results / "crops",
        cache=results / "cache",
        clips=results / "clips",
        outputs=results / "outputs",
        tools=results / "tools",
    )


def _default_results(root: Path, tag: str) -> Path:
    sync = Path("/workspace/vm-results")
    if sync.is_dir():
        return sync / "clarity" / tag
    return root / "vm-results" / "clarity" / tag


def _check_os(run: _Run) -> None:
    if winreg is None or run.allow_other_os:
        return
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            product, _kind = winreg.QueryValueEx(key, "ProductName")
    except OSError as exc:
        raise _VmError(f"Windows product name could not be read ({exc})", EXIT_SETUP) from exc
    if "Windows Server 2025" not in str(product):
        raise _VmError(
            f"This script expects Windows Server 2025. Found: {product}. Pass --allow-other-os when the image caption differs.",
            EXIT_SETUP,
        )


def _run_steps(run: _Run) -> int:
    names = SMOKE_STEPS if run.mode == "smoke" else FULL_STEPS
    handlers = _handlers()
    for name in names:
        if name not in {"gpu", "install"} and marker_path(run.tag, name).is_file():
            run.steps.append({"name": name, "status": "skip", "seconds": 0.0, "error": ""})
            _emit(run, {"type": "step", "name": name, "status": "skip", "seconds": 0.0})
            if not run.json_mode:
                print(f"SKIP {name}")
            continue
        print(f"step {name}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        handler = handlers.get(name)
        if handler is None:
            return _finish(run, EXIT_FAIL, name, f"unknown step {name}")
        try:
            handler(run)
        except _VmError as exc:
            seconds = _step_seconds(name, started, run)
            run.steps.append({"name": name, "status": "fail", "seconds": seconds, "error": str(exc)})
            _emit(run, {"type": "step", "name": name, "status": "fail", "seconds": seconds, "error": str(exc)})
            return _finish(run, exc.code, name, str(exc))
        seconds = _step_seconds(name, started, run)
        _write_marker(run.tag, name)
        run.steps.append({"name": name, "status": "pass", "seconds": seconds, "error": ""})
        _emit(run, {"type": "step", "name": name, "status": "pass", "seconds": seconds})
        if not run.json_mode:
            print(f"PASS {name} ({seconds:.1f}s)")
    return _finish(run, EXIT_PASS, "", "")


def _handlers() -> dict[str, Callable[[_Run], None]]:
    return {
        "gpu": _step_gpu,
        "install": _step_install,
        "selftest": _step_selftest,
        "info": _step_info,
        "clip": _step_clip,
        "clips": _step_clips,
        "restore-clean": _step_restore_clean,
        "restore-2x": _step_restore_2x,
        "case-a": _step_case_a,
        "case-b": _step_case_b,
        "case-c": _step_case_c,
        "bounds": _step_bounds,
        "forced-failure": _step_forced_failure,
        "bench": _step_bench,
    }


def _step_gpu(run: _Run) -> None:
    status = probe_gpu(allow_other=run.allow_other_gpu)
    (run.logs / "gpu.log").write_text(
        json.dumps({"ok": status.ok, "name": status.name, "driver": status.driver, "compute": status.compute, "message": status.message})
        + "\n",
        encoding="utf-8",
    )
    if not status.ok:
        raise _VmError(status.message, EXIT_NO_GPU)


def _step_seconds(name: str, started: float, run: _Run) -> float:
    """Return the step duration, including the parent wheel install for ``install``.

    Args:
        name: Step name.
        started: ``time.perf_counter`` when the child step began.
        run: The current invocation. ``install_seconds`` is the parent elapsed time.

    Returns:
        Seconds rounded to three places.
    """
    elapsed = time.perf_counter() - started
    if name == "install":
        elapsed += run.install_seconds
    return round(elapsed, 3)


def _step_install(run: _Run) -> None:
    run.install_seconds = _take_install_seconds(run)
    try:
        if not _clarity_imports():
            raise _VmError("the virtualenv does not import clarity", EXIT_SETUP)
        _locate_ffmpeg(run)
        _check_providers(run, "install")
    except _VmError:
        marker_path(run.tag, "install").unlink(missing_ok=True)
        raise


def _step_selftest(run: _Run) -> None:
    completed = _python(run, ["-m", "clarity", "--selftest"], "selftest", 600)
    log = (completed.stdout or "") + (completed.stderr or "")
    (run.logs / "selftest.log").write_text(log, encoding="utf-8")
    if completed.returncode != 0:
        code = EXIT_SETUP if completed.returncode == EXIT_SETUP else EXIT_FAIL
        raise _VmError(f"python -m clarity --selftest exited {completed.returncode}", code)
    _check_providers(run, "selftest")


def _step_info(run: _Run) -> None:
    completed = _python(run, ["-m", "clarity", "info", "--json"], "info", 120)
    (run.logs / "info.log").write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        raise _VmError(f"python -m clarity info --json exited {completed.returncode}", EXIT_FAIL)
    body = _last_result(completed.stdout or "")
    meta = json.loads((run.root / "BUILD.json").read_text(encoding="utf-8"))
    if not body or body.get("ok") is not True:
        raise _VmError("info --json did not print a result with ok true", EXIT_FAIL)
    if body.get("version") != meta["version"] or body.get("build") != meta["build"]:
        raise _VmError("info --json does not match BUILD.json", EXIT_FAIL)


def _step_clip(run: _Run) -> None:
    _cut_copy(run, run.clips / "smoke.mp4", 0, SMOKE_SECONDS)


def _step_clips(run: _Run) -> None:
    _cut_copy(run, run.clips / "case-a.mp4", 0, 8)
    _cut_filter(
        run,
        run.clips / "case-b.mp4",
        8,
        8,
        "eq=gamma=0.6:brightness=-0.25,noise=alls=8:allf=t",
    )
    _cut_filter(
        run,
        run.clips / "case-c.mp4",
        16,
        10,
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
    )


def _step_restore_clean(run: _Run) -> None:
    _restore(run, source=run.clips / "smoke.mp4", dest=run.outputs / "restore-clean.mkv", mode="clean", upscale=False, step="restore-clean", require_trt=True, sample=False, timeout=20 * 60)


def _step_restore_2x(run: _Run) -> None:
    _restore(run, source=run.clips / "smoke.mp4", dest=run.outputs / "restore-2x.mkv", mode="clean", upscale=True, step="restore-2x", require_trt=True, sample=True, timeout=20 * 60)


def _step_case_a(run: _Run) -> None:
    _restore(run, source=run.clips / "case-a.mp4", dest=run.outputs / "case-a.mkv", mode="clean", upscale=False, step="case-a", require_trt=True, sample=False, timeout=45 * 60)


def _step_case_b(run: _Run) -> None:
    _restore(run, source=run.clips / "case-b.mp4", dest=run.outputs / "case-b.mkv", mode="strong", upscale=False, step="case-b", require_trt=True, sample=False, timeout=45 * 60)


def _step_case_c(run: _Run) -> None:
    _restore(run, source=run.clips / "case-c.mp4", dest=run.outputs / "case-c.mkv", mode="strong", upscale=True, step="case-c", require_trt=True, sample=True, timeout=45 * 60)


def _step_bounds(run: _Run) -> None:
    name = "restore-2x" if run.mode == "smoke" else "case-c"
    path = run.logs / f"{name}-peaks.json"
    if not path.is_file():
        raise _VmError(f"{name} did not record a RAM/VRAM sample", EXIT_FAIL)
    sample = json.loads(path.read_text(encoding="utf-8"))
    ram = float(sample.get("ram_mib") or 0)
    vram = float(sample.get("vram_mib") or 0)
    run.bounds = {"ram_mib": ram, "vram_mib": vram, "ram_limit_mib": RAM_BOUND_MIB, "vram_limit_mib": VRAM_BOUND_MIB}
    if ram <= 0 or vram <= 0:
        raise _VmError("the restore did not report a RAM and VRAM peak", EXIT_FAIL)
    if ram > RAM_BOUND_MIB:
        raise _VmError(f"peak RAM {ram:.0f} MiB exceeded {RAM_BOUND_MIB} MiB", EXIT_FAIL)
    if vram > VRAM_BOUND_MIB:
        raise _VmError(f"peak VRAM {vram:.0f} MiB exceeded {VRAM_BOUND_MIB} MiB", EXIT_FAIL)


def _step_forced_failure(run: _Run) -> None:
    missing = run.results / "missing-source.mp4"
    dest = run.outputs / "forced-failure.mkv"
    completed = _python(
        run,
        [
            "-m",
            "clarity",
            "restore",
            "--input",
            str(missing),
            "--output",
            str(dest),
            "--json",
            "--ffmpeg",
            run.ffmpeg,
            "--ffprobe",
            run.ffprobe,
        ],
        "forced-failure",
        180,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    (run.logs / "forced-failure.log").write_text(text, encoding="utf-8")
    reason = forced_failure_error(completed.returncode, completed.stdout or "")
    if reason:
        raise _VmError(reason, EXIT_FAIL)


def _step_bench(run: _Run) -> None:
    out = run.results / "bench"
    completed = _python(
        run,
        ["-m", "clarity.bench", "--out", str(out), "--clip-dir", str(run.cache)],
        "bench",
        60 * 60,
    )
    (run.logs / "bench.log").write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        code = EXIT_SETUP if completed.returncode == EXIT_SETUP else EXIT_FAIL
        raise _VmError(f"python -m clarity.bench exited {completed.returncode}", code)
    for name in ("benchmark.md", "benchmark.json"):
        if not (out / name).is_file():
            raise _VmError(f"the benchmark did not write {name}", EXIT_FAIL)
    copied = 0
    for path in out.glob("*.png"):
        shutil.copy2(path, run.crops / f"bench-{path.name}")
        copied += 1
    if copied == 0:
        raise _VmError("the benchmark wrote no before/after crops", EXIT_FAIL)


def _restore(
    run: _Run,
    *,
    source: Path,
    dest: Path,
    mode: str,
    upscale: bool,
    step: str,
    require_trt: bool,
    sample: bool,
    timeout: int,
) -> None:
    if not source.is_file():
        raise _VmError(f"{source.name} is missing", EXIT_SETUP)
    if dest.exists():
        dest.unlink()
    cmd = [
        sys.executable,
        "-m",
        "clarity",
        "restore",
        "--input",
        str(source),
        "--output",
        str(dest),
        "--mode",
        mode,
        "--format",
        "mkv",
        "--json",
        "--ffmpeg",
        run.ffmpeg,
        "--ffprobe",
        run.ffprobe,
    ]
    if upscale:
        cmd.append("--upscale")
    stdout, stderr, code, ram, vram = _stream(cmd, _job_env(run), timeout, sample)
    (run.logs / f"{step}.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
    if sample:
        payload = {"ram_mib": round(ram, 1), "vram_mib": round(vram, 1)}
        (run.logs / f"{step}-peaks.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if code != 0:
        raise _VmError(f"restore exited {code}", EXIT_FAIL if code != EXIT_SETUP else EXIT_SETUP)
    if not dest.is_file() or dest.stat().st_size <= 0:
        raise _VmError("the restore did not write a video", EXIT_FAIL)
    if require_trt and not trt_stage_seen(stdout):
        raise _VmError("the restore did not report a TensorRT engine stage", EXIT_FAIL)
    _crop(run, source, run.crops / f"{step}-before.png")
    _crop(run, dest, run.crops / f"{step}-after.png")


def _stream(cmd: list[str], env: dict[str, str], timeout: int, sample: bool) -> tuple[str, str, int, float, float]:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    except OSError as exc:
        raise _VmError(f"could not start {cmd[0]} ({exc})", EXIT_SETUP) from exc
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()
    ram_peak = 0.0
    vram_peak = 0.0
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if time.monotonic() > deadline:
            proc.kill()
            raise _VmError(f"the command exceeded {timeout} seconds", EXIT_FAIL)
        if sample:
            ram_peak = max(ram_peak, _tree_rss_mib(proc.pid))
            used = _vram_mib()
            if used is not None:
                vram_peak = max(vram_peak, used)
        time.sleep(0.5)
    for thread in threads:
        thread.join(timeout=30)
    return "".join(stdout_lines), "".join(stderr_lines), int(proc.returncode or 0), ram_peak, vram_peak


def _drain(pipe: IO[str] | None, bucket: list[str]) -> None:
    if pipe is None:
        return
    for line in pipe:
        bucket.append(line)


def _python(run: _Run, args: list[str], step: str, timeout: int) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, *args]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_job_env(run), check=False)
    except subprocess.TimeoutExpired as exc:
        raise _VmError(f"{step} exceeded {timeout} seconds", EXIT_FAIL) from exc
    except OSError as exc:
        raise _VmError(f"could not start python ({exc})", EXIT_SETUP) from exc
    return completed


def _cut_copy(run: _Run, dest: Path, start: int, seconds: int) -> None:
    source = _source_clip(run)
    _ffmpeg(
        run,
        ["-ss", str(start), "-i", str(source), "-t", str(seconds), "-c", "copy", str(dest)],
        f"could not cut {dest.name}",
    )


def _cut_filter(run: _Run, dest: Path, start: int, seconds: int, video_filter: str) -> None:
    source = _source_clip(run)
    _ffmpeg(
        run,
        [
            "-i",
            str(source),
            "-ss",
            str(start),
            "-t",
            str(seconds),
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            str(dest),
        ],
        f"could not prepare {dest.name}",
    )


def _crop(run: _Run, source: Path, dest: Path) -> None:
    _ffmpeg(run, ["-i", str(source), "-frames:v", "1", str(dest)], f"could not write {dest.name}")


def _ffmpeg(run: _Run, args: list[str], what: str) -> None:
    cmd = [run.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    completed = _checked(cmd, 180, what, setup=True)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {completed.returncode}"
        raise _VmError(f"{what} ({detail})", EXIT_SETUP)


def _source_clip(run: _Run) -> Path:
    dest = run.cache / CLIP_NAME
    if dest.is_file() and _sha256(dest) == CLIP_SHA256:
        return dest
    if dest.exists():
        dest.unlink()
    _download(CLIP_URL, dest, 64 * 1024 * 1024)
    if _sha256(dest) != CLIP_SHA256:
        dest.unlink(missing_ok=True)
        raise _VmError("the Elephants Dream download hash did not match", EXIT_SETUP)
    return dest


def _install_ready(run: _Run) -> bool:
    return marker_path(run.tag, "install").is_file() and _venv_python(run.results).is_file()


def _install_stamp(run: _Run) -> Path:
    """Return the file that carries the parent install duration to the child."""
    return run.logs / "install-seconds.json"


def _take_install_seconds(run: _Run) -> float:
    """Read and delete the parent install duration so a resume does not add it twice.

    Args:
        run: The current invocation.

    Returns:
        The recorded seconds, or 0 when the file is missing or not a number.
    """
    stamp = _install_stamp(run)
    if not stamp.is_file():
        return 0.0
    try:
        body = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        body = {}
    stamp.unlink(missing_ok=True)
    seconds = body.get("seconds", 0.0) if isinstance(body, dict) else 0.0
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return 0.0
    return float(seconds)


def _check_providers(run: _Run, step: str) -> None:
    """Fail the step unless CUDA and TensorRT execution providers load.

    The install marker is written after this returns, so a CPU-only wheel
    cannot count as a finished install.

    Args:
        run: The current invocation.
        step: ``install`` or ``selftest``.

    Raises:
        _VmError: When the provider script exits non-zero.
    """
    script = run.root / "scripts" / "vm_provider_check.py"
    provider = _python(run, [str(script)], step, 180)
    (run.logs / "providers.log").write_text((provider.stdout or "") + (provider.stderr or ""), encoding="utf-8")
    if provider.returncode != 0:
        detail = (provider.stdout or provider.stderr or "").strip()
        raise _VmError(detail or "CUDAExecutionProvider or TensorrtExecutionProvider did not load", EXIT_FAIL)


def _pin_runtime(run: _Run, uv: str, python: Path) -> None:
    """Install torch and spandrel, then drop a CPU ONNX Runtime wheel if one is present.

    The engine ``gpu`` extra installs ``onnxruntime-gpu``. The CPU distribution
    named ``onnxruntime`` is uninstalled only when that distribution is
    installed. ``uv pip`` is required because this virtualenv is created
    without pip. The provider check after this step fails the install when
    CUDA or TensorRT is still missing.

    Args:
        run: The current invocation.
        uv: The uv executable.
        python: The virtualenv interpreter.
    """
    script = run.root / "installer" / "pin_runtime.py"
    lock = run.root / "installer" / "runtime_lock.json"
    requirements = run.root / "requirements-gpu.txt"
    _checked(
        [
            sys.executable,
            str(script),
            "--uv",
            uv,
            "--python",
            str(python),
            "--lock",
            str(lock),
            "--requirements",
            str(requirements),
        ],
        900,
        "the runtime pin failed",
        setup=True,
    )


def _requirements_without_common(source: Path, dest: Path) -> Path:
    """Copy ``source`` without the ``voxprint-common`` line.

    The verified wheel is installed first. A later ``pip -r`` must not
    download that URL again.

    Args:
        source: ``requirements.txt``.
        dest: File to write.

    Returns:
        ``dest``.
    """
    lines = [line for line in source.read_text(encoding="utf-8").splitlines(True) if "voxprint-common" not in line]
    dest.write_text("".join(lines), encoding="utf-8")
    return dest


def _install(run: _Run) -> None:
    started = time.perf_counter()
    uv = _ensure_uv(run.tools)
    _checked([uv, "python", "install", "3.14"], 600, "uv could not install CPython 3.14", setup=True)
    found = _checked([uv, "python", "find", "3.14"], 120, "Python 3.14 was not found", setup=True)
    base = (found.stdout or "").strip()
    if not base:
        raise _VmError("Python 3.14 was not found after uv python install", EXIT_SETUP)
    py = _venv_python(run.results)
    if not py.is_file():
        _checked([uv, "venv", "--python", base, str(run.results / "venv")], 180, "uv could not create the virtualenv", setup=True)
    lock = run.root / "installer" / "runtime_lock.json"
    fetched = _checked(
        [
            sys.executable,
            str(run.root / "installer" / "fetch_common.py"),
            "--lock",
            str(lock),
            "--dest",
            str(run.tools / "common-wheel"),
        ],
        180,
        "voxprint-common download failed",
        setup=True,
    )
    wheel = (fetched.stdout or "").strip()
    if not wheel:
        raise _VmError("voxprint-common download did not print a wheel path", EXIT_SETUP)
    _checked([uv, "pip", "install", "--python", str(py), wheel], 180, "voxprint-common install failed", setup=True)
    _checked(
        [uv, "pip", "install", "--python", str(py), "--no-deps", "-e", str(run.root)],
        600,
        "the Clarity checkout failed to install",
        setup=True,
    )
    _checked(
        [uv, "pip", "install", "--python", str(py), "torch==2.14.1", "--index-url", TORCH_INDEX],
        900,
        "torch 2.14.1 cu130 failed to install",
        setup=True,
    )
    filtered = _requirements_without_common(run.root / "requirements.txt", run.tools / "requirements.txt")
    _checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "-r",
            str(filtered),
            "-r",
            str(run.root / "requirements-bench.txt"),
        ],
        900,
        "requirements failed to install",
        setup=True,
    )
    _checked(
        [uv, "pip", "install", "--python", str(py), "--extra-index-url", NVIDIA_INDEX, ENGINE_URL],
        900,
        "the inference engine install failed",
        setup=True,
    )
    _checked(
        [
            sys.executable,
            str(run.root / "installer" / "stage_suite_runtime.py"),
            "--python",
            str(py),
            "--lock",
            str(lock),
        ],
        900,
        "enlarge.suite_runtime failed to install",
        setup=True,
    )
    _pin_runtime(run, uv, py)
    _ensure_ffmpeg(run.tools)
    elapsed = round(time.perf_counter() - started, 3)
    _install_stamp(run).write_text(json.dumps({"seconds": elapsed}) + "\n", encoding="utf-8")
    (run.logs / "install.log").write_text(f"install ok\nseconds {elapsed}\n", encoding="utf-8")


def _reexec(run: _Run, args: list[str]) -> int:
    py = _venv_python(run.results)
    if not py.is_file():
        return _finish(run, EXIT_SETUP, "install", "the virtualenv has no python")
    completed = subprocess.run([str(py), str(Path(__file__).resolve()), *args], check=False, env=_job_env(run))
    return int(completed.returncode)


def _ensure_uv(tools: Path) -> str:
    found = shutil.which("uv")
    if found:
        return found
    name = "uv.exe" if os.name == "nt" else "uv"
    local = tools / "uv" / name
    if local.is_file():
        return str(local)
    url = UV_URL.get("win32" if os.name == "nt" else "linux", "")
    if not url:
        raise _VmError("uv has no download for this platform", EXIT_SETUP)
    archive = tools / ("uv.zip" if url.endswith(".zip") else "uv.tar.gz")
    _download(url, archive, 80 * 1024 * 1024)
    dest = tools / "uv"
    dest.mkdir(parents=True, exist_ok=True)
    _extract_named(archive, name, local)
    if os.name != "nt":
        local.chmod(local.stat().st_mode | 0o755)
    archive.unlink(missing_ok=True)
    return str(local)


def _ensure_ffmpeg(tools: Path) -> None:
    binary = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    folder = tools / "ffmpeg"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / binary
    probe_path = folder / probe
    if not target.is_file() or not probe_path.is_file():
        url = FFMPEG_URL.get("win32" if os.name == "nt" else "linux", "")
        if not url.startswith("https://"):
            raise _VmError("ffmpeg has no https LGPL build for this platform", EXIT_SETUP)
        archive = tools / ("ffmpeg.zip" if url.endswith(".zip") else "ffmpeg.tar.xz")
        _download(url, archive, 256 * 1024 * 1024)
        _extract_named(archive, binary, target)
        _extract_named(archive, probe, probe_path)
        archive.unlink(missing_ok=True)
        if os.name != "nt":
            target.chmod(target.stat().st_mode | 0o755)
            probe_path.chmod(probe_path.stat().st_mode | 0o755)
    _reject_ffmpeg(target)


def _locate_ffmpeg(run: _Run) -> None:
    binary = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    folder = run.tools / "ffmpeg"
    target = folder / binary
    probe_path = folder / probe
    if not target.is_file() or not probe_path.is_file():
        raise _VmError("the LGPL ffmpeg is missing from the results tools directory", EXIT_SETUP)
    _reject_ffmpeg(target)
    run.ffmpeg = str(target)
    run.ffprobe = str(probe_path)


def _reject_ffmpeg(binary: Path) -> None:
    build = subprocess.run([str(binary), "-hide_banner", "-buildconf"], capture_output=True, text=True, check=False)
    text = (build.stdout or "") + (build.stderr or "")
    if ffmpeg_build_is_gpl(text):
        raise _VmError("ffmpeg is not an LGPL build", EXIT_SETUP)
    encoders = subprocess.run([str(binary), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    if "av1_nvenc" not in ((encoders.stdout or "") + (encoders.stderr or "")):
        raise _VmError("this ffmpeg has no av1_nvenc", EXIT_SETUP)


def _checked(cmd: list[str], timeout: int, what: str, *, setup: bool) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise _VmError(f"{what} exceeded {timeout} seconds", EXIT_SETUP if setup else EXIT_FAIL) from exc
    except OSError as exc:
        raise _VmError(f"{what} ({exc})", EXIT_SETUP if setup else EXIT_FAIL) from exc
    if completed.returncode != 0:
        tail = ((completed.stderr or "") + "\n" + (completed.stdout or "")).strip().splitlines()
        detail = tail[-1] if tail else f"exit {completed.returncode}"
        raise _VmError(f"{what} ({detail})", EXIT_SETUP if setup else EXIT_FAIL)
    return completed


def _download(url: str, dest: Path, limit: int) -> None:
    if not url.startswith("https://"):
        raise _VmError("downloads must use https", EXIT_SETUP)
    _install_trust()
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-clarity"})
    opener = urllib.request.build_opener(_HttpsRedirect)
    try:
        with opener.open(request, timeout=120) as response:
            final = str(response.geturl())
            if not final.startswith("https://"):
                raise _VmError("the download was redirected off https", EXIT_SETUP)
            written = 0
            with dest.open("wb") as handle:
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > limit:
                        raise _VmError("the download is larger than expected", EXIT_SETUP)
                    handle.write(block)
    except _VmError:
        dest.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        dest.unlink(missing_ok=True)
        raise _VmError(f"download failed: {exc}", EXIT_SETUP) from exc


class _HttpsRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves https."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Refuse a redirect that leaves https.

        Args:
            req: The original request.
            fp: The response body.
            code: The redirect status.
            msg: The redirect reason.
            headers: The response headers.
            newurl: The target URL.

        Returns:
            The redirected request.

        Raises:
            _VmError: When ``newurl`` is not https.
        """
        if not str(newurl).startswith("https://"):
            raise _VmError("the download was redirected off https", EXIT_SETUP)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _extract_named(archive: Path, wanted: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            member = _named_member(bundle.namelist(), wanted)
            if member is None:
                raise _VmError(f"the archive has no {wanted}", EXIT_SETUP)
            with bundle.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        return
    mode = "r:xz" if archive.name.endswith(".tar.xz") else "r:gz"
    with tarfile.open(archive, mode) as bundle:
        chosen: tarfile.TarInfo | None = None
        for info in bundle.getmembers():
            if Path(info.name).name == wanted and info.isfile():
                chosen = info
                break
        if chosen is None:
            raise _VmError(f"the archive has no {wanted}", EXIT_SETUP)
        extracted = bundle.extractfile(chosen)
        if extracted is None:
            raise _VmError(f"the archive has no {wanted}", EXIT_SETUP)
        with extracted, target.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)


def _named_member(names: list[str], wanted: str) -> str | None:
    for name in names:
        if Path(name).name == wanted:
            return name
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tree_rss_mib(pid: int) -> float:
    if os.name == "nt":
        return _windows_tree_rss_mib(pid)
    return _linux_tree_rss_mib(pid)


def _linux_tree_rss_mib(pid: int) -> float:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return 0.0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        current = int(entry.name)
        parents[current] = _status_int(text, "PPid")
        rss[current] = _status_int(text, "VmRSS")
    wanted = {pid}
    changed = True
    while changed:
        changed = False
        for current, parent in parents.items():
            if parent in wanted and current not in wanted:
                wanted.add(current)
                changed = True
    total_kib = sum(rss.get(current, 0) for current in wanted)
    return total_kib / 1024


def _windows_tree_rss_mib(pid: int) -> float:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot in (None, 0, invalid, -1):
        return 0.0
    parents: dict[int, int] = {}
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
        while True:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    kernel32.CloseHandle(snapshot)
    wanted = {pid}
    changed = True
    while changed:
        changed = False
        for current, parent in parents.items():
            if parent in wanted and current not in wanted:
                wanted.add(current)
                changed = True
    total = 0
    for current in wanted:
        handle = kernel32.OpenProcess(0x1000, False, current)
        if not handle:
            continue
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            total += int(counters.WorkingSetSize)
        kernel32.CloseHandle(handle)
    return total / (1024 * 1024)


def _vram_mib() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    line = (completed.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        return float(line[0].split(",")[0].strip())
    except ValueError:
        return None


def _clarity_imports() -> bool:
    completed = subprocess.run([sys.executable, "-c", "import clarity"], capture_output=True, text=True, check=False)
    return completed.returncode == 0


def _venv_python(results: Path) -> Path:
    folder = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    return results / "venv" / folder / name


def _in_venv(results: Path) -> bool:
    expected = _venv_python(results)
    try:
        return expected.is_file() and Path(sys.executable).resolve() == expected.resolve()
    except OSError:
        return False


def _job_env(run: _Run) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("VOXPRINT_CI_LANCZOS", None)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    prefix: list[str] = []
    if run.ffmpeg:
        prefix.append(str(Path(run.ffmpeg).parent))
    env["PATH"] = os.pathsep.join([*prefix, env.get("PATH", "")])
    return env


def _write_marker(tag: str, step: str) -> None:
    path = marker_path(tag, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def _finish(run: _Run, code: int, step: str, error: str) -> int:
    ok = code == EXIT_PASS
    summary: dict[str, Any] = {
        "type": "summary",
        "app": "clarity",
        "mode": run.mode,
        "tag": run.tag,
        "ok": ok,
        "exit": code,
        "step": step,
        "error": error,
        "results": str(run.results),
        "steps": run.steps,
    }
    if run.bounds:
        summary["bounds"] = run.bounds
    (run.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run.results / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    _emit(run, summary)
    if ok and not run.json_mode:
        print(f"OK: {run.results}")
    if not ok:
        print(f"FAILED: step {step}: {error}", file=sys.stderr)
    return code


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Clarity GPU VM test",
        "",
        f"- mode: {summary['mode']}",
        f"- tag: {summary['tag']}",
        f"- ok: {str(summary['ok']).lower()}",
        f"- exit: {summary['exit']}",
        f"- step: {summary['step']}",
        "",
        CLIP_CREDIT,
        "",
        "Case A is 8 s of the pinned low-bitrate H.264 clip, Clean (DRUNet), 2x off.",
        "Case B is 8 s darkened (gamma 0.6, brightness -0.25) with temporal noise, Strong (SCUNet), 2x off.",
        "Case C is 10 s padded to 1280x720, Strong, 2x on.",
        "",
        "| Step | Status | Seconds |",
        "|---|---|---|",
    ]
    steps = summary.get("steps")
    if isinstance(steps, list):
        for item in steps:
            if isinstance(item, dict):
                lines.append(f"| {item.get('name')} | {item.get('status')} | {item.get('seconds')} |")
    if summary.get("error"):
        lines.extend(["", f"Error: {summary['error']}"])
    bounds = summary.get("bounds")
    if isinstance(bounds, dict):
        lines.extend(["", f"Peak RAM {bounds.get('ram_mib')} MiB (limit {bounds.get('ram_limit_mib')}).", f"Peak VRAM {bounds.get('vram_mib')} MiB (limit {bounds.get('vram_limit_mib')})."])
    lines.append("")
    return "\n".join(lines)


def _emit(run: _Run, payload: dict[str, Any]) -> None:
    if run.json_mode:
        print(json.dumps(payload), flush=True)


def _json_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(body, dict):
        return body
    return None


def _last_result(stdout: str) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for line in stdout.splitlines():
        body = _json_line(line)
        if body is not None and body.get("type") == "result":
            found = body
    return found


def _leading_int(text: str) -> int:
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else 0


def _compute_pair(text: str) -> tuple[int, int]:
    if "." not in text:
        return _leading_int(text), 0
    major, _, minor = text.partition(".")
    return _leading_int(major), _leading_int(minor)


def _status_int(text: str, field_name: str) -> int:
    prefix = field_name + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return _leading_int(line.split(":", 1)[1])
    return 0


def _install_trust() -> None:
    global _TRUST_READY
    if _TRUST_READY or truststore is None:
        return
    truststore.inject_into_ssl()
    _TRUST_READY = True


if __name__ == "__main__":
    raise SystemExit(main())
