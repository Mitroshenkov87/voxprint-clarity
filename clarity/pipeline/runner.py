"""Run one video through decode, restore, and encode.

The GPU lock, VRAM budget, thermal rule, and weight fetch come from
``voxprint_engine``. Tests pass a ``Control`` so they can run without a GPU,
ffmpeg, or the private engine.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, IO

import numpy as np

from clarity.engine import (
    choose_tile,
    empty_cuda_cache,
    ensure_model_onnx,
    ensure_weight,
    gpu_busy_text,
    halve_tile,
    hold_gpu,
    measure,
    note_spill,
    open_restore_session,
    read_shared_usage,
    run_onnx_batch,
    sysmem_spill,
    thermal_decide,
)
from clarity.errors import Cancelled, ClarityError
from clarity.models import ClarityModel
from clarity.pipeline.colour import apply_colour_frame, uint8_colour_stats
from clarity.pipeline.encode import (
    concat_command,
    concat_list,
    decode_range_command,
    encode_chunk_command,
    encoder_pix_fmt,
    parse_encoder_list,
    planned_encoder,
    select_encoder,
)
from clarity.pipeline.sharpen import unsharp
from clarity.pipeline.job import Hooks, Settings, models_for, process_frames, wait_for_thermal
from clarity.pipeline.probe import parse_probe, probe_command, require_sdr
from clarity.pipeline.resume import (
    ChunkRecord,
    append_journal,
    chunk_path,
    file_sha256,
    load_done,
    load_plan,
    plan_matches,
    publish_chunk,
    reset_chunks,
    settings_digest,
    write_plan,
)
from clarity.pipeline.scenes import parse_scene_cuts, plan_chunks, scene_command

Proc = Any
Progress = Callable[[float, str], None]


class CancelToken:
    """Set from the UI thread. ``cancel`` also kills ffmpeg so a blocked read returns."""

    def __init__(self) -> None:
        """Start unset, with no ffmpeg processes to kill."""
        self._flag = threading.Event()
        self._procs: list[Proc] = []
        self._guard = threading.Lock()

    def is_set(self) -> bool:
        """Return whether ``cancel`` has been called."""
        return self._flag.is_set()

    def watch(self, proc: Proc) -> None:
        """Remember an ffmpeg process so a later cancel can kill it.

        Args:
            proc: A process with ``kill``. Already-cancelled tokens kill it now.
        """
        with self._guard:
            self._procs.append(proc)
        if self._flag.is_set():
            _kill(proc)

    def cancel(self) -> None:
        """Set the flag and kill every watched ffmpeg process."""
        self._flag.set()
        with self._guard:
            procs = list(self._procs)
        for proc in procs:
            _kill(proc)


@dataclass
class JobSpec:
    """One restore: source, destination, settings, and the ffmpeg binaries."""

    source: Path
    destination: Path
    settings: Settings
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    work: Path | None = None
    scene_threshold: float = 0.4


@dataclass
class Control:
    """The side effects a job needs. Tests substitute every field."""

    infer: Callable[[ClarityModel, np.ndarray], np.ndarray]
    lock: Callable[[str, float], AbstractContextManager[None]]
    spawn: Callable[[list[str]], Proc]
    run_text: Callable[[list[str]], str]
    token: CancelToken = field(default_factory=CancelToken)
    progress: Progress = lambda _fraction, _text: None
    device: bool = False


def _kill(proc: Proc) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except OSError:
        return


def _spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    kwargs: dict[str, Any] = {"stderr": subprocess.PIPE}
    if "pipe:1" in argv:
        kwargs["stdout"] = subprocess.PIPE
    else:
        kwargs["stdin"] = subprocess.PIPE
        kwargs["stdout"] = subprocess.DEVNULL
    if os.name == "nt":
        kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return subprocess.Popen(argv, **kwargs)


def _run_text(argv: list[str]) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClarityError(f"{argv[0]} failed to start: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "command failed").strip()
        raise ClarityError(detail)
    return proc.stdout


def _prepare_infer(settings: Settings, progress: Progress) -> Callable[[ClarityModel, np.ndarray], np.ndarray]:
    """Fetch weights, convert them with the engine, and run the ONNX session.

    DRUNet's manifest sets ``in_channels`` to 4. The noise-level map is that
    fourth channel, and the export example is ``1x4xHxW``. Catalogue precision
    stays fp32 for DRUNet: the engine runs the module in float, then writes
    the checked fp16 graph.
    """
    paths = {model.id: ensure_weight(model.manifest()) for model in models_for(settings)}
    sessions: dict[str, Any] = {}

    def infer(model: ClarityModel, nchw: np.ndarray) -> np.ndarray:
        path = paths.get(model.id)
        if path is None:
            raise ClarityError(f"{model.id} was not fetched before the job")
        session = sessions.get(model.id)
        if session is None:
            onnx_path = ensure_model_onnx(model.manifest(), path, progress)
            session = open_restore_session(onnx_path, progress)
            sessions[model.id] = session
        return np.asarray(run_onnx_batch(session, nchw), dtype=np.float32)

    return infer


def live_control(settings: Settings, token: CancelToken, progress: Progress | None = None) -> Control:
    """The path the Start button uses. The lock owner matches the restore worker."""
    report = progress or (lambda _fraction, _text: None)

    def on_wait(holder: dict[str, Any], waited: float) -> None:
        del waited
        who = holder.get("owner", "?")
        what = holder.get("job", "?")
        report(0.0, gpu_busy_text(f"{who} / {what}"))

    def lock(job: str, eta_s: float) -> AbstractContextManager[None]:
        return hold_gpu(job, eta_s=eta_s, cancel=token.is_set, on_wait=on_wait)

    return Control(
        infer=_prepare_infer(settings, report),
        lock=lock,
        spawn=_spawn,
        run_text=_run_text,
        token=token,
        device=True,
    )


def _read_exact(stream: IO[bytes], size: int, token: CancelToken) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        if token.is_set():
            raise Cancelled("cancelled")
        part = stream.read(size - len(buf))
        if token.is_set():
            raise Cancelled("cancelled")
        if not part:
            break
        buf += part
    if token.is_set():
        raise Cancelled("cancelled")
    return bytes(buf)


def _drain(stream: IO[bytes] | None, sink: list[str]) -> None:
    if stream is None:
        return
    try:
        data = stream.read()
    except OSError:
        return
    if data:
        sink.append(data.decode("utf-8", errors="replace")[-2000:])


def _as_bytes(frames: np.ndarray) -> bytes:
    image = np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image).tobytes()


def _write_all(stream: IO[bytes], payload: bytes) -> None:
    view = payload
    while view:
        written = stream.write(view)
        if written is None:
            return
        if written <= 0:
            raise ClarityError("the encoder stopped reading frames")
        view = view[written:]


def _fail_text(sink: list[str], code: int) -> str:
    detail = " ".join(part.strip() for part in sink if part.strip())
    if detail:
        return f"ffmpeg failed ({code}): {detail}"
    return f"ffmpeg failed ({code})"


def work_dir_for(spec: JobSpec) -> Path:
    """Return the folder that holds ``plan.json``, the journal, and chunk files.

    Args:
        spec: Job whose ``work`` override or destination name is used.

    Returns:
        ``spec.work`` when set, otherwise ``destination`` with a ``.work`` suffix.
    """
    if spec.work is not None:
        return spec.work
    return spec.destination.with_name(spec.destination.name + ".work")


def _finished_records(journal: Path, folder: Path, ranges: list[tuple[int, int]]) -> dict[int, ChunkRecord]:
    """Journal rows whose chunk file is still on disk and hashes to the recorded digest."""
    done = load_done(journal)
    usable: dict[int, ChunkRecord] = {}
    for index, (start, stop) in enumerate(ranges):
        record = done.get(index)
        if record is None or record.start != start or record.stop != stop or not record.sha256:
            continue
        path = chunk_path(folder, index)
        if path.is_file() and file_sha256(path) == record.sha256:
            usable[index] = record
    return usable


def build_restore_plan(
    spec: JobSpec,
    probe: Any,
    *,
    encoder: str,
    cuts: list[int],
    ranges: list[tuple[int, int]],
    work: Path,
    done: dict[int, ChunkRecord],
    encoder_available: bool,
) -> dict[str, Any]:
    """The ffmpeg commands and model list for one restore. Nothing is executed."""
    require_sdr(probe)
    out_h = probe.height * (2 if spec.settings.upscale else 1)
    out_w = probe.width * (2 if spec.settings.upscale else 1)
    chunks_dir = work / "chunks"
    chunk_plans: list[dict[str, Any]] = []
    for index, (start, stop) in enumerate(ranges):
        final = chunk_path(chunks_dir, index)
        partial = final.with_name(final.name + ".partial")
        chunk_plans.append(
            {
                "index": index,
                "start": start,
                "stop": stop,
                "done": index in done,
                "decode": decode_range_command(
                    spec.ffmpeg,
                    str(spec.source),
                    start,
                    stop,
                    deinterlace=probe.interlaced,
                ),
                "encode": encode_chunk_command(
                    spec.ffmpeg,
                    encoder=encoder,
                    destination=str(partial),
                    width=out_w,
                    height=out_h,
                    fps=probe.fps,
                    pix_fmt=encoder_pix_fmt(probe.pix_fmt),
                ),
            }
        )
    list_path = work / "concat.txt"
    partial_out = spec.destination.with_name(spec.destination.name + ".partial")
    return {
        "source": str(spec.source),
        "destination": str(spec.destination),
        "mode": spec.settings.mode,
        "upscale": spec.settings.upscale,
        "container": spec.settings.container.value,
        "codec": spec.settings.codec.value,
        "encoder": encoder,
        "encoder_available": encoder_available,
        "width": int(probe.width),
        "height": int(probe.height),
        "fps": float(probe.fps),
        "frames": int(probe.frames),
        "output_width": out_w,
        "output_height": out_h,
        "models": [model.id for model in models_for(spec.settings)],
        "cuts": list(cuts),
        "probe_command": probe_command(spec.ffprobe, str(spec.source)),
        "scene_command": scene_command(spec.ffmpeg, str(spec.source), spec.scene_threshold),
        "chunks": chunk_plans,
        "concat_list": str(list_path),
        "concat": concat_command(
            spec.ffmpeg,
            str(list_path),
            str(spec.source),
            container=spec.settings.container,
            destination=str(partial_out),
            copy_audio=spec.settings.copy_audio,
        ),
        "executed": False,
        "gpu_used": False,
    }


def dry_run_restore(spec: JobSpec, *, run_text: Callable[[list[str]], str] | None = None) -> dict[str, Any]:
    """Validate the source and build the restore plan without the GPU.

    ffprobe and the CPU scene filter may run. NVDEC, NVENC, CUDA, and the
    weight loader do not.
    """
    text_of = _run_text if run_text is None else run_text
    if not spec.source.is_file():
        raise ClarityError("the source video was not found")
    probe_payload = json.loads(text_of(probe_command(spec.ffprobe, str(spec.source))))
    if not isinstance(probe_payload, dict):
        raise ClarityError("ffprobe did not return an object")
    probe = parse_probe(probe_payload)
    require_sdr(probe)
    if probe.frames < 1:
        raise ClarityError("ffprobe did not report a frame count, so the job cannot resume by chunk")
    scene_text = text_of(scene_command(spec.ffmpeg, str(spec.source), spec.scene_threshold))
    cuts = parse_scene_cuts(scene_text, probe.fps)
    encoders = parse_encoder_list(text_of([spec.ffmpeg, "-hide_banner", "-encoders"]))
    encoder, available = planned_encoder(spec.settings.codec, encoders)
    ranges = plan_chunks(probe.frames, cuts, spec.settings.max_frames)
    digest = settings_digest(spec.settings.digest_payload())
    work = work_dir_for(spec)
    source = str(spec.source.resolve())
    existing = load_plan(work / "plan.json")
    done: dict[int, ChunkRecord] = {}
    if existing is not None and plan_matches(existing, digest=digest, source=source, ranges=ranges):
        done = _finished_records(work / "journal.jsonl", work / "chunks", ranges)
    return build_restore_plan(
        spec,
        probe,
        encoder=encoder,
        cuts=cuts,
        ranges=ranges,
        work=work,
        done=done,
        encoder_available=available,
    )


def _prepare_plan(spec: JobSpec, probe: Any, bound: Control) -> tuple[Path, list[tuple[int, int]], dict[int, ChunkRecord], list[int]]:
    if probe.frames < 1:
        raise ClarityError("ffprobe did not report a frame count, so the job cannot resume by chunk")
    scene_text = bound.run_text(scene_command(spec.ffmpeg, str(spec.source), spec.scene_threshold))
    cuts = parse_scene_cuts(scene_text, probe.fps)
    ranges = plan_chunks(probe.frames, cuts, spec.settings.max_frames)
    digest = settings_digest(spec.settings.digest_payload())
    work = work_dir_for(spec)
    chunks = work / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    journal = work / "journal.jsonl"
    source = str(spec.source.resolve())
    existing = load_plan(plan_path)
    if existing is None or not plan_matches(existing, digest=digest, source=source, ranges=ranges):
        reset_chunks(work)
        write_plan(
            plan_path,
            {
                "version": 1,
                "settings_sha256": digest,
                "source": source,
                "cuts": cuts,
                "ranges": [list(pair) for pair in ranges],
                "width": probe.width,
                "height": probe.height,
                "fps": probe.fps,
            },
        )
        return work, ranges, {}, cuts
    return work, ranges, _finished_records(journal, chunks, ranges), cuts


def host_batch_count(height: int, width: int, model_scale: int) -> int:
    """How many source frames stay in memory before ``iter_tile_frames``.

    The engine iterator keeps one frame in the accumulator. This limit is the
    same 512 MiB budget the three full-chunk float32 buffers used to need, so
    a 720p frame at 4x is still restored alone and a 138-frame chunk is not.

    Args:
        height: Source height in pixels.
        width: Source width in pixels.
        model_scale: 4 while the upscale net runs, otherwise 1.

    Returns:
        At least one.
    """
    pixels = max(1, int(height) * int(model_scale) * int(width) * int(model_scale))
    per_frame = pixels * 3 * 4 * 3
    return max(1, (512 * 1024 * 1024) // per_frame)


def _restore_range(
    spec: JobSpec,
    bound: Control,
    token: CancelToken,
    *,
    probe: Any,
    decode: list[str],
    encode: list[str],
    start: int,
    stop: int,
    partial: Path,
    tile: int,
    batch: int,
) -> tuple[int, int]:
    """Decode one chunk, restore it, and leave the encoded bytes in ``partial``.

    The engine is given at most :func:`host_batch_count` frames, and
    ``iter_tile_frames`` restores those one at a time. Each batch is downscaled
    before the next one starts, stored as uint8, then written to the encoder.
    Colour uses one gain measured on that uint8 chunk, so a scene does not pump
    and the 4x float32 chunk is never allocated.
    """
    out_h = probe.height * (2 if spec.settings.upscale else 1)
    out_w = probe.width * (2 if spec.settings.upscale else 1)
    frame_bytes = probe.width * probe.height * 3
    count = stop - start
    model_scale = 4 if spec.settings.upscale else 1
    limit = host_batch_count(probe.height, probe.width, model_scale)
    plain = replace(spec.settings, white_balance=False, levels=False, sharpen=0.0)
    store_path = partial.with_name(partial.name + ".frames")
    decoder = bound.spawn(decode)
    encoder = bound.spawn(encode)
    token.watch(decoder)
    token.watch(encoder)
    stderr_text: list[str] = []
    drains = [
        threading.Thread(target=_drain, args=(getattr(decoder, "stderr", None), stderr_text), daemon=True),
        threading.Thread(target=_drain, args=(getattr(encoder, "stderr", None), stderr_text), daemon=True),
    ]
    for drain in drains:
        drain.start()
    store: np.memmap | None = None
    try:
        source_out = decoder.stdout
        if source_out is None:
            raise ClarityError("the decoder has no stdout")
        encoder_in = encoder.stdin
        if encoder_in is None:
            raise ClarityError("the encoder has no stdin")
        store = np.memmap(store_path, dtype=np.uint8, mode="w+", shape=(count, out_h, out_w, 3))
        pending: list[np.ndarray] = []
        cursor = 0

        def flush_batch() -> None:
            nonlocal cursor
            if not pending or store is None:
                return
            if token.is_set():
                raise Cancelled("cancelled")
            restored = process_frames(
                np.stack(pending),
                plain,
                bound.infer,
                tile=tile,
                batch=batch,
                use_engine=bound.device,
            )
            expected = (len(pending), out_h, out_w, 3)
            if restored.shape != expected:
                raise ClarityError(f"restored frames are {restored.shape}, expected {expected}")
            stored = np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
            store[cursor : cursor + stored.shape[0]] = stored
            cursor += stored.shape[0]
            pending.clear()

        for _ in range(count):
            if token.is_set():
                raise Cancelled("cancelled")
            raw = _read_exact(source_out, frame_bytes, token)
            if len(raw) != frame_bytes:
                raise ClarityError("the decoder ended in the middle of a frame")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(probe.height, probe.width, 3)
            pending.append(frame.astype(np.float32) / 255.0)
            if len(pending) >= limit:
                flush_batch()
        flush_batch()
        if cursor != count:
            raise ClarityError(f"restored {cursor} frames, expected {count}")
        store.flush()
        means, lo, hi = uint8_colour_stats(store)
        for index in range(count):
            if token.is_set():
                raise Cancelled("cancelled")
            image = store[index].astype(np.float32) / 255.0
            image = apply_colour_frame(
                image,
                means,
                lo,
                hi,
                white_balance=spec.settings.white_balance,
                levels=spec.settings.levels,
            )
            if spec.settings.sharpen > 0:
                image = unsharp(image, spec.settings.sharpen)
            _write_all(encoder_in, _as_bytes(image[None, ...]))
        encoder_in.close()
        code_decode = int(decoder.wait(timeout=120))
        code_encode = int(encoder.wait(timeout=120))
        if code_decode != 0 or code_encode != 0:
            raise ClarityError(_fail_text(stderr_text, code_decode or code_encode))
    except Cancelled:
        _discard(decoder, encoder, partial)
        raise
    except Exception as exc:
        _discard(decoder, encoder, partial)
        for drain in drains:
            drain.join(timeout=1)
        if token.is_set():
            raise Cancelled("cancelled") from None
        detail = " ".join(part.strip() for part in stderr_text if part.strip())
        if isinstance(exc, ClarityError):
            if detail and detail not in str(exc):
                raise ClarityError(f"{exc}\n{detail}") from exc
            raise
        message = f"{type(exc).__name__}: {exc}"
        if detail:
            message = f"{message}\n{detail}"
        raise ClarityError(message) from exc
    finally:
        mapped = store
        store = None
        if mapped is not None:
            mem = getattr(mapped, "_mmap", None)
            del mapped
            if mem is not None:
                try:
                    mem.close()
                except OSError:
                    pass
        try:
            store_path.unlink(missing_ok=True)
        except OSError:
            pass
    return tile, batch


def _device_budget(
    spec: JobSpec,
    bound: Control,
    token: CancelToken,
    *,
    started: float,
    paused: bool,
    tile: int,
    batch: int,
    baseline: int | None,
) -> tuple[bool, int, int, int | None]:
    hooks = Hooks(
        infer=bound.infer,
        decide=thermal_decide,
        elapsed=lambda: time.monotonic() - started,
    )
    paused = _wait(hooks, paused, token)
    budget = float(getattr(measure(), "budget_gb", 0.0) or 0.0)
    if spec.settings.tile <= 0:
        tile = choose_tile(budget, 0, 0)
    usage = read_shared_usage()
    if sysmem_spill(usage, baseline):
        empty_cuda_cache()
        batch = note_spill(batch, True)
        smaller = halve_tile(tile)
        if smaller is not None:
            tile = int(smaller)
        baseline = usage
    elif baseline is None:
        baseline = usage
    return paused, tile, batch, baseline


def run_job(
    spec: JobSpec,
    control: Control | None = None,
    *,
    token: CancelToken | None = None,
    progress: Progress | None = None,
    on_chunk: Callable[[Path], None] | None = None,
) -> None:
    """Decode each unfinished scene chunk, restore it, and join the chunks with ``-c copy``.

    Omitting ``control`` uses the suite GPU lock (owner ``clarity``) and ffmpeg.
    A cancel keeps finished chunks. The next run of the same work folder resumes
    there. The final file is published only after every chunk has been joined.
    """
    if not spec.source.is_file():
        raise ClarityError("the source video was not found")
    if token is None:
        token = control.token if control is not None else CancelToken()
    text_of = control.run_text if control is not None else _run_text
    probe_payload = json.loads(text_of(probe_command(spec.ffprobe, str(spec.source))))
    if not isinstance(probe_payload, dict):
        raise ClarityError("ffprobe did not return an object")
    probe = parse_probe(probe_payload)
    require_sdr(probe)
    if control is None:
        bound = live_control(spec.settings, token, progress)
    else:
        control.token = token
        bound = control
    if progress is not None:
        bound.progress = progress
    if token.is_set():
        raise Cancelled("cancelled")
    encoders = parse_encoder_list(bound.run_text([spec.ffmpeg, "-hide_banner", "-encoders"]))
    encoder_name = select_encoder(spec.settings.codec, encoders)
    work, ranges, done, cuts = _prepare_plan(spec, probe, bound)
    plan = build_restore_plan(
        spec,
        probe,
        encoder=encoder_name,
        cuts=cuts,
        ranges=ranges,
        work=work,
        done=done,
        encoder_available=True,
    )
    chunks = work / "chunks"
    journal = work / "journal.jsonl"
    total = probe.frames
    eta = max(120.0, float(total or 300))
    tile = spec.settings.tile if spec.settings.tile > 0 else 512
    batch = max(1, spec.settings.batch)
    baseline: int | None = None
    paused = False
    started = time.monotonic()
    finished_frames = sum(record.stop - record.start for record in done.values())
    if finished_frames:
        bound.progress(finished_frames / total, f"{finished_frames}/{total}")
    partial = spec.destination.with_name(spec.destination.name + ".partial")
    try:
        with bound.lock("clarity:restore", eta):
            for chunk in plan["chunks"]:
                if chunk["done"]:
                    continue
                index = int(chunk["index"])
                start = int(chunk["start"])
                stop = int(chunk["stop"])
                if token.is_set():
                    raise Cancelled("cancelled")
                if bound.device:
                    paused, tile, batch, baseline = _device_budget(
                        spec,
                        bound,
                        token,
                        started=started,
                        paused=paused,
                        tile=tile,
                        batch=batch,
                        baseline=baseline,
                    )
                chunk_final = chunk_path(chunks, index)
                chunk_partial = Path(str(chunk["encode"][-1]))
                _restore_range(
                    spec,
                    bound,
                    token,
                    probe=probe,
                    decode=list(chunk["decode"]),
                    encode=list(chunk["encode"]),
                    start=start,
                    stop=stop,
                    partial=chunk_partial,
                    tile=tile,
                    batch=batch,
                )
                if token.is_set():
                    _discard(None, None, chunk_partial)
                    raise Cancelled("cancelled")
                publish_chunk(chunk_partial, chunk_final)
                append_journal(journal, ChunkRecord(index, start, stop, file_sha256(chunk_final)))
                if on_chunk is not None:
                    on_chunk(work)
                finished_frames += stop - start
                bound.progress(finished_frames / total, f"{finished_frames}/{total}")
            if token.is_set():
                raise Cancelled("cancelled")
            if finished_frames < 1:
                raise ClarityError("the decoder produced no frames")
            list_path = Path(str(plan["concat_list"]))
            ordered = [str(chunk_path(chunks, index).resolve()) for index in range(len(ranges))]
            list_path.write_text(concat_list(ordered), encoding="utf-8")
            partial.parent.mkdir(parents=True, exist_ok=True)
            joiner = bound.spawn(list(plan["concat"]))
            token.watch(joiner)
            stderr_text: list[str] = []
            threading.Thread(target=_drain, args=(getattr(joiner, "stderr", None), stderr_text), daemon=True).start()
            code = int(joiner.wait(timeout=120))
            if token.is_set():
                _discard(None, joiner, partial)
                raise Cancelled("cancelled")
            if code != 0:
                _discard(None, joiner, partial)
                raise ClarityError(_fail_text(stderr_text, code))
        if token.is_set():
            raise Cancelled("cancelled")
        os.replace(partial, spec.destination)
        bound.progress(1.0, f"{total}/{total}")
    except Cancelled:
        partial.unlink(missing_ok=True)
        raise


def _wait(hooks: Hooks, paused: bool, token: CancelToken) -> bool:
    """Thermal pause, but leave as soon as the user cancels."""

    def sleep(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if token.is_set():
                raise Cancelled("cancelled")
            time.sleep(0.1)

    hooks.sleep = sleep
    return wait_for_thermal(hooks, paused)


def _discard(decoder: Proc | None, encoder: Proc | None, partial: Path) -> None:
    for proc in (decoder, encoder):
        if proc is None:
            continue
        _kill(proc)
        try:
            proc.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
    try:
        partial.unlink(missing_ok=True)
    except OSError:
        return
