"""Benchmark DRUNet, SCUNet, and compact Real-ESRGAN on a rented RTX 4090.

``python -m clarity.bench`` downloads a permissively licensed Blender open-movie
clip and the three weights, exports each graph to ONNX, times PyTorch fp16
against ONNX Runtime TensorRT EP fp16, and writes before/after PNG crops
plus a markdown report and JSON.

This machine must be an NVIDIA GPU. The script refuses to pretend it ran.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import time
import urllib.request

from clarity.net import install_system_trust
from clarity.precision import apply_precision, torch_dtype_name
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from clarity.models import ClarityModel, get_model
from clarity.pipeline.condition import deblock_level, sigma_level

# Torch and spandrel are optional GPU extras. The import stays at module scope
# so the benchmark module still loads in CI without the CUDA wheel.
try:
    import torch
    from spandrel import ModelLoader
except ImportError:
    torch = None
    ModelLoader = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Elephants Dream (Blender Foundation, Creative Commons Attribution).
# Archive.org item ElephantsDream, file ed_1024_512kb.mp4 (about 45 MB).
CLIP_URL = "https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4"
CLIP_SHA256 = "fd6dc0e9da3a2f63487957325983f197f23c6ff1cd2b29bb71e64db567e0994c"
CLIP_NAME = "elephants-dream-ed_1024_512kb.mp4"
CLIP_CREDIT = "Elephants Dream (c) copyright Blender Foundation / Netherlands Media Art Institute. Creative Commons Attribution."

# Three graphs: DRUNet (the deblocking weight; the denoise weight is the same arch),
# SCUNet, and compact Real-ESRGAN.
MODEL_IDS = (
    "drunet-deblocking-color",
    "scunet-color-real-psnr",
    "realesr-general-x4v3",
)
SPATIAL = 256


@dataclass
class Sample:
    """One timed run of a catalogue model on one backend."""

    model_id: str
    backend: str
    fps: float
    vram_peak_mb: float
    frames: int
    note: str
    before_png: str = ""
    after_png: str = ""


def download_clip(folder: Path) -> Path:
    """Fetch the pinned clip. A hash mismatch deletes the file."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / CLIP_NAME
    if target.is_file():
        found = hashlib.sha256(target.read_bytes()).hexdigest()
        if found == CLIP_SHA256:
            return target
        target.unlink()
    if not CLIP_URL.startswith("https://"):
        raise RuntimeError("the clip URL must be https")
    partial = target.with_suffix(target.suffix + ".partial")
    install_system_trust()
    request = urllib.request.Request(CLIP_URL, headers={"User-Agent": "voxprint-clarity-bench"})
    hasher = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - https is checked above
        if not str(response.geturl()).startswith("https://"):
            raise RuntimeError("the clip download left https")
        with partial.open("wb") as handle:
            while True:
                block = response.read(1024 * 256)
                if not block:
                    break
                hasher.update(block)
                handle.write(block)
    if hasher.hexdigest() != CLIP_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("the test clip does not match the pinned sha256")
    partial.replace(target)
    return target


def require_gpu() -> str:
    """Return a short GPU name, or raise when this is not an NVIDIA GPU job."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "clarity.bench needs an NVIDIA GPU (the rented RTX 4090 VM). nvidia-smi was not found."
        )
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        raise RuntimeError("nvidia-smi did not report a GPU. This benchmark does not run on CPU.")
    return (proc.stdout or "").strip().splitlines()[0]


def render_report(gpu: str, samples: list[Sample], clip: str) -> tuple[str, dict[str, Any]]:
    """Build the markdown table and the JSON payload for a benchmark.

    Args:
        gpu: Name reported by nvidia-smi.
        samples: Timed runs, including rows that failed to measure.
        clip: Path or name of the pinned clip.

    Returns:
        Markdown text and the same data as a dictionary.
    """
    payload: dict[str, Any] = {
        "gpu": gpu,
        "clip": clip,
        "clip_sha256": CLIP_SHA256,
        "clip_credit": CLIP_CREDIT,
        "samples": [asdict(sample) for sample in samples],
    }
    lines = [
        "# Clarity benchmark",
        "",
        f"GPU: {gpu}",
        "",
        f"Clip: `{clip}`",
        "",
        CLIP_CREDIT,
        "",
        "| Model | Backend | fps | VRAM peak (MB) | Frames | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for sample in samples:
        lines.append(
            f"| {sample.model_id} | {sample.backend} | {sample.fps:.3f} | {sample.vram_peak_mb:.1f} | {sample.frames} | {sample.note} |"
        )
    if not samples:
        lines.append("| — | — | — | — | — | no sample was measured |")
    crops = [sample for sample in samples if sample.before_png or sample.after_png]
    if crops:
        lines.extend(["", "## Crops", ""])
        for sample in crops:
            lines.append(f"- {sample.model_id} ({sample.backend}): `{sample.before_png}` `{sample.after_png}`")
    lines.append("")
    return "\n".join(lines) + "\n", payload


def tensorrt_providers(cache: Path, *, precision: str = "fp16") -> list[Any]:
    """ORT providers: TensorRT EP, then CUDA EP. There is no CPU provider.

    Args:
        cache: Directory for the TensorRT engine cache.
        precision: ``fp16`` turns on ``trt_fp16_enable``. DRUNet stays ``fp32``.

    Returns:
        The provider list passed to ``InferenceSession``.
    """
    return [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": precision == "fp16",
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
            },
        ),
        "CUDAExecutionProvider",
    ]


def _rgb_u8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        image = frame
    else:
        image = np.clip(np.rint(np.asarray(frame) * 255.0), 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("a crop must be HWC RGB")
    return np.ascontiguousarray(image)


def center_crop(frame: np.ndarray, size: int = SPATIAL) -> np.ndarray:
    """Take a square from the middle of an HWC frame.

    Args:
        frame: Image whose first two axes are height and width.
        size: Side length. A smaller frame is cropped to its own size.

    Returns:
        A contiguous crop.
    """
    height, width = int(frame.shape[0]), int(frame.shape[1])
    crop_h = min(size, height)
    crop_w = min(size, width)
    y = max(0, (height - crop_h) // 2)
    x = max(0, (width - crop_w) // 2)
    return np.ascontiguousarray(frame[y : y + crop_h, x : x + crop_w])


def fit_square(frame: np.ndarray, size: int = SPATIAL) -> np.ndarray:
    """Center crop, then pad with black when the frame is smaller than ``size``."""
    crop = center_crop(frame, size)
    if crop.shape[0] == size and crop.shape[1] == size:
        return crop.astype(np.float32, copy=False)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[: crop.shape[0], : crop.shape[1]] = crop
    return canvas


def save_crops(before: np.ndarray, after: np.ndarray, folder: Path, model_id: str) -> tuple[Path, Path]:
    """Write the before and after PNG pair for one model.

    Args:
        before: Preview frame, HWC RGB.
        after: Restored frame, HWC RGB.
        folder: Directory that receives the files.
        model_id: Used in the file names.

    Returns:
        Paths of the before and after PNGs.
    """
    folder.mkdir(parents=True, exist_ok=True)
    before_path = folder / f"{model_id}-before.png"
    after_path = folder / f"{model_id}-after.png"
    Image.fromarray(_rgb_u8(before), mode="RGB").save(before_path)
    Image.fromarray(_rgb_u8(after), mode="RGB").save(after_path)
    return before_path, after_path


def conditioned_nchw(model: ClarityModel, rgb: np.ndarray) -> np.ndarray:
    """NCHW float32. DRUNet gets its fourth channel; the other bench models stay RGB."""
    image = np.transpose(rgb.astype(np.float32, copy=False), (2, 0, 1))[None]
    if model.channels == 3 or model.condition == "none":
        return image
    if model.condition == "deblock":
        level = deblock_level(40)
    elif model.condition == "sigma":
        level = sigma_level(15)
    else:
        raise ValueError(f"unknown condition {model.condition}")
    extra = np.full((1, 1, rgb.shape[0], rgb.shape[1]), level, dtype=np.float32)
    return np.concatenate([image, extra], axis=1)


def _timed(step: Any, frames: int, sync: Any) -> float:
    step()
    if sync is not None:
        sync()
    started = time.perf_counter()
    for _ in range(frames):
        step()
    if sync is not None:
        sync()
    return frames / max(1e-6, time.perf_counter() - started)


def _vram_mb() -> float:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def export_with_engine(weight: Path, dest: Path) -> str:
    """Write a checked fp16 ONNX file with the engine converter.

    Args:
        weight: The pinned ``.pth`` file.
        dest: Where the ``.onnx`` file is written.

    Returns:
        A note stored on the benchmark row.

    Raises:
        RuntimeError: When the engine converter is missing or the export fails.
    """
    try:
        module = importlib.import_module("voxprint_engine.onnx_export")
    except ImportError as exc:
        raise RuntimeError("voxprint_engine.onnx_export is not installed") from exc
    export = getattr(module, "export_weight", None)
    if not callable(export):
        raise RuntimeError("voxprint_engine.onnx_export.export_weight is missing")
    try:
        export(weight, dest)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    if not dest.is_file():
        raise RuntimeError(f"the engine did not write {dest.name}")
    return "ONNX exported by voxprint_engine.onnx_export.export_weight (fp16)"


def _load_torch(weight: Path, precision: str) -> Any:
    if torch is None or ModelLoader is None or not torch.cuda.is_available():
        raise RuntimeError("PyTorch, spandrel, and CUDA are required. Install requirements-bench.txt.")
    torch.cuda.reset_peak_memory_stats()
    return apply_precision(ModelLoader().load_from_file(str(weight)).eval().cuda(), precision)


def _torch_forward(network: Any, nchw: np.ndarray, precision: str) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is not installed")
    dtype = getattr(torch, torch_dtype_name(precision))
    tensor = torch.from_numpy(np.ascontiguousarray(nchw)).to(device="cuda", dtype=dtype)
    with torch.inference_mode():
        output = network(tensor)
    return np.asarray(output.detach().float().cpu().numpy(), dtype=np.float32)


def time_torch(network: Any, channels: int, frames: int, precision: str) -> tuple[float, float]:
    """Time CUDA inference on a blank tile at the model's declared precision.

    Args:
        network: Loaded PyTorch module already on CUDA.
        channels: Input channel count, including a condition channel.
        frames: Number of forward passes.
        precision: ``fp16``, ``bf16``, or ``fp32``.

    Returns:
        Frames per second and the peak VRAM in megabytes.
    """
    sample = np.zeros((1, channels, SPATIAL, SPATIAL), dtype=np.float32)

    def step() -> None:
        _torch_forward(network, sample, precision)

    fps = _timed(step, frames, torch.cuda.synchronize if torch is not None else None)
    return fps, _vram_mb()


def open_trt_session(onnx_path: Path, cache: Path, *, precision: str = "fp16") -> Any:
    """Open an ONNX session that must include TensorrtExecutionProvider.

    Args:
        onnx_path: Exported graph.
        cache: Directory for the TensorRT engine cache.
        precision: Passed to :func:`tensorrt_providers`. DRUNet is ``fp32``.

    Returns:
        The ONNX Runtime inference session.

    Raises:
        RuntimeError: When ONNX Runtime is missing or TensorRT did not load.
    """
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Install requirements-bench.txt on this VM.")
    session = ort.InferenceSession(str(onnx_path), providers=tensorrt_providers(cache, precision=precision))
    providers = [str(name) for name in session.get_providers()]
    if "TensorrtExecutionProvider" not in providers:
        raise RuntimeError(
            "ONNX Runtime did not load TensorrtExecutionProvider. "
            f"Active providers: {', '.join(providers)}. Install a TensorRT build of onnxruntime-gpu."
        )
    return session


def _session_array(session: Any, channels: int) -> np.ndarray:
    kind = str(session.get_inputs()[0].type)
    dtype = np.float16 if "float16" in kind else np.float32
    return np.zeros((1, channels, SPATIAL, SPATIAL), dtype=dtype)


def time_trt(session: Any, channels: int, frames: int) -> tuple[float, float]:
    """Time TensorRT inference on a blank tile matching the session dtype.

    Args:
        session: Session from ``open_trt_session``.
        channels: Input channel count.
        frames: Number of ``session.run`` calls.

    Returns:
        Frames per second and the peak VRAM in megabytes.
    """
    array = _session_array(session, channels)
    name = str(session.get_inputs()[0].name)

    def step() -> None:
        session.run(None, {name: array})

    return _timed(step, frames, None), _vram_mb()


def rgb_from_nchw(output: np.ndarray) -> np.ndarray:
    """Take the first three channels of an NCHW tensor as HWC RGB in 0..1.

    Args:
        output: Batch of one, NCHW.

    Returns:
        Clipped float32 HWC image.
    """
    rgb = np.transpose(output[0, :3], (1, 2, 0))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def download_weight(model: ClarityModel, folder: Path) -> Path:
    """Fetch one official weight and check the catalogue SHA-256."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / model.filename
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == model.sha256:
        return target
    if target.is_file():
        target.unlink()
    if not model.download_url.startswith("https://"):
        raise RuntimeError(f"{model.id} has no https download")
    partial = target.with_suffix(target.suffix + ".partial")
    install_system_trust()
    request = urllib.request.Request(model.download_url, headers={"User-Agent": "voxprint-clarity-bench"})
    hasher = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - https is checked above
        if not str(response.geturl()).startswith("https://"):
            raise RuntimeError(f"{model.id} left https")
        with partial.open("wb") as handle:
            while True:
                block = response.read(1024 * 256)
                if not block:
                    break
                hasher.update(block)
                handle.write(block)
    if hasher.hexdigest() != model.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{model.id} does not match the pinned sha256")
    partial.replace(target)
    return target


def read_preview_frame(clip: Path) -> np.ndarray:
    """One rgb24 frame from the clip, via ffmpeg. The bench does not invent a picture."""
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("clarity.bench needs ffmpeg and ffprobe on PATH to save before/after crops.")
    probe = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(clip),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(probe.stderr.strip() or "ffprobe failed")
    payload = json.loads(probe.stdout or "{}")
    streams = payload.get("streams") if isinstance(payload, dict) else None
    video = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None) if isinstance(streams, list) else None
    if not isinstance(video, dict):
        raise RuntimeError("the clip has no video stream")
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width < 1 or height < 1:
        raise RuntimeError("the clip size could not be read")
    raw = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if raw.returncode != 0 or len(raw.stdout) < width * height * 3:
        detail = raw.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "ffmpeg did not return a frame")
    frame = np.frombuffer(raw.stdout[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)
    return frame.astype(np.float32) / 255.0


def write_report(folder: Path, gpu: str, samples: list[Sample], clip: str) -> tuple[Path, Path]:
    """Write ``benchmark.md`` and ``benchmark.json``.

    Args:
        folder: Output directory. Created when missing.
        gpu: Name reported by nvidia-smi.
        samples: Timed runs.
        clip: Clip path recorded in the report.

    Returns:
        Paths of the markdown and JSON files.
    """
    folder.mkdir(parents=True, exist_ok=True)
    markdown, payload = render_report(gpu, samples, clip)
    md_path = folder / "benchmark.md"
    json_path = folder / "benchmark.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def _unmeasured(model_id: str, backend: str, note: str) -> Sample:
    return Sample(model_id, backend, 0.0, 0.0, 0, note)


def _reset_vram() -> None:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI.

    Args:
        argv: Arguments without the program name. Defaults to ``sys.argv``.

    Returns:
        ``0`` when every model was timed, ``2`` when setup or a model failed.
    """
    parser = argparse.ArgumentParser(description="Benchmark Clarity models on an NVIDIA GPU.")
    parser.add_argument("--out", type=Path, default=Path("benchmark-out"))
    parser.add_argument("--clip-dir", type=Path, default=Path.home() / ".cache" / "voxprint" / "clarity-bench")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 1:
        print("--frames must be at least 1")
        return 2
    try:
        gpu = require_gpu()
    except RuntimeError as exc:
        print(exc)
        return 2
    try:
        clip_path = args.clip_dir / CLIP_NAME if args.skip_download else download_clip(args.clip_dir)
        if not clip_path.is_file():
            raise RuntimeError("the clip is missing; run without --skip-download")
        preview = fit_square(read_preview_frame(clip_path))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"clip prepare failed: {exc}")
        return 2
    samples: list[Sample] = []
    failed = False
    for model_id in MODEL_IDS:
        model = get_model(model_id)
        trt_backend = f"onnxruntime-tensorrt-{model.precision}"
        weight = args.clip_dir / model.filename
        try:
            if not args.skip_download:
                weight = download_weight(model, args.clip_dir)
            if not weight.is_file():
                raise RuntimeError(f"{model.filename} is missing; run without --skip-download")
            network = _load_torch(weight, model.precision)
            fps, vram = time_torch(network, model.channels, args.frames, model.precision)
            restored = rgb_from_nchw(_torch_forward(network, conditioned_nchw(model, preview), model.precision))
            before_path, after_path = save_crops(preview, restored, args.out, model.id)
        except Exception as exc:
            samples.append(_unmeasured(model.id, f"pytorch-{model.precision}", str(exc)))
            samples.append(_unmeasured(model.id, trt_backend, "not timed; the PyTorch path failed"))
            failed = True
            continue
        samples.append(
            Sample(
                model.id,
                f"pytorch-{model.precision}",
                fps,
                vram,
                args.frames,
                f"{model.precision} CUDA",
                str(before_path),
                str(after_path),
            )
        )
        onnx_path = args.out / f"{model.id}.onnx"
        try:
            if model.channels != 3:
                raise RuntimeError(
                    f"{model.id} takes {model.channels} channels. "
                    "The engine converter writes a 3-channel fp16 graph."
                )
            exported = export_with_engine(weight, onnx_path)
            _reset_vram()
            session = open_trt_session(onnx_path, args.out / "trt-cache" / model.id, precision=model.precision)
            trt_fps, trt_vram = time_trt(session, model.channels, args.frames)
        except Exception as exc:
            samples.append(_unmeasured(model.id, trt_backend, str(exc)))
            failed = True
            continue
        samples.append(Sample(model.id, trt_backend, trt_fps, trt_vram, args.frames, exported))
    md_path, json_path = write_report(args.out, gpu, samples, str(clip_path))
    print(md_path)
    print(json_path)
    if failed or any(sample.fps <= 0 for sample in samples):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
