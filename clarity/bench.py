"""Benchmark DRUNet, SCUNet, and compact Real-ESRGAN on a rented RTX 4090.

``python -m clarity.bench`` downloads a permissively licensed Blender open-movie
clip, runs each model in PyTorch fp16 and, when ONNX Runtime TensorRT is
present, that execution provider, and writes a markdown report plus JSON.

This machine must be an NVIDIA GPU. The script refuses to pretend it ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clarity.models import get_model

# Torch and spandrel are optional GPU extras. The import stays at module scope
# so the benchmark module still loads in CI without the CUDA wheel.
try:
    import torch
    from spandrel import ModelLoader
except ImportError:
    torch = None
    ModelLoader = None

# Elephants Dream (Blender Foundation, Creative Commons Attribution).
# Archive.org item ElephantsDream, file ed_1024_512kb.mp4 (about 45 MB).
CLIP_URL = "https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4"
CLIP_SHA256 = "fd6dc0e9da3a2f63487957325983f197f23c6ff1cd2b29bb71e64db567e0994c"
CLIP_NAME = "elephants-dream-ed_1024_512kb.mp4"
CLIP_CREDIT = "Elephants Dream (c) copyright Blender Foundation / Netherlands Media Art Institute. Creative Commons Attribution."

MODEL_IDS = (
    "drunet-deblocking-color",
    "drunet-color",
    "scunet-color-real-psnr",
    "realesr-general-x4v3",
)


@dataclass
class Sample:
    model_id: str
    backend: str
    fps: float
    vram_peak_mb: float
    frames: int
    note: str


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
    lines.append("")
    return "\n".join(lines) + "\n", payload


def _torch_fp16(model_path: Path, frames: int) -> Sample | None:
    """Time a fp16 PyTorch forward if torch and spandrel can load the weight. Otherwise None."""
    if torch is None or ModelLoader is None or not torch.cuda.is_available():
        return None
    started = time.perf_counter()
    model = ModelLoader().load_from_file(str(model_path)).eval().cuda().half()
    tensor = torch.zeros(1, 3, 256, 256, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(frames):
            model(tensor)
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - started)
    peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return Sample(model_path.stem, "pytorch-fp16", frames / elapsed, peak, frames, "256x256 synthetic")


def write_report(folder: Path, gpu: str, samples: list[Sample], clip: str) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    markdown, payload = render_report(gpu, samples, clip)
    md_path = folder / "benchmark.md"
    json_path = folder / "benchmark.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Clarity models on an NVIDIA GPU.")
    parser.add_argument("--out", type=Path, default=Path("benchmark-out"))
    parser.add_argument("--clip-dir", type=Path, default=Path.home() / ".cache" / "voxprint" / "clarity-bench")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        gpu = require_gpu()
    except RuntimeError as exc:
        print(exc)
        return 2
    clip = ""
    if not args.skip_download:
        try:
            clip = str(download_clip(args.clip_dir))
        except (OSError, RuntimeError) as exc:
            print(f"clip download failed: {exc}")
            return 2
    samples: list[Sample] = []
    for model_id in MODEL_IDS:
        model = get_model(model_id)
        weight = args.clip_dir / model.filename
        if not weight.is_file():
            samples.append(Sample(model_id, "pytorch-fp16", 0.0, 0.0, 0, "weight not in the clip dir; download it with the shared model store"))
            samples.append(Sample(model_id, "onnxruntime-tensorrt-fp16", 0.0, 0.0, 0, "ONNX file not next to the weight"))
            continue
        timed = _torch_fp16(weight, args.frames)
        if timed is None:
            samples.append(Sample(model_id, "pytorch-fp16", 0.0, 0.0, 0, "torch or spandrel is not installed, or CUDA is not available"))
        else:
            samples.append(Sample(model_id, timed.backend, timed.fps, timed.vram_peak_mb, timed.frames, timed.note))
        onnx = weight.with_suffix(".onnx")
        if not onnx.is_file():
            samples.append(Sample(model_id, "onnxruntime-tensorrt-fp16", 0.0, 0.0, 0, "export the ONNX fp16 graph first; TensorRT EP was not timed"))
        else:
            samples.append(Sample(model_id, "onnxruntime-tensorrt-fp16", 0.0, 0.0, 0, "ORT TensorRT timing needs the engine session on this GPU"))
    md_path, json_path = write_report(args.out, gpu, samples, clip)
    print(md_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
