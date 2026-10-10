# Voxprint AI Clarity

Lightweight non-generative video restoration for the Voxprint AI Media Suite.
A 1080p video comes back at the same resolution: deblocking, denoise, classical
sharpening, and a per-scene colour cleanup. Optional 2x upscale is off by
default. There is no frame interpolation, no GAN enhanced mode, no DDColor,
and no NVIDIA RTX Video SDK. Stills belong to Voxprint AI Enlarge.

## Requirements

- Windows 11 24H2 or newer (build 26100), or a 2025-or-newer Linux distribution.
- An NVIDIA GPU from the RTX 40 series (compute capability 8.9, Ada) or newer.
- An NVIDIA driver from the 600 series or newer.
- Python 3.14 and PyTorch 2.14.1 built for CUDA 13.0.
- ffmpeg from an LGPL build, on `PATH`, with NVDEC and NVENC. Clarity does not
  bundle ffmpeg, x264, or x265.

Clarity refuses to start on older GPUs, on a driver below 600, or without an
NVIDIA GPU. There is no CPU-only mode.

## Model licences

The application is [Apache-2.0](LICENSE). Third-party weights and libraries are
listed in [NOTICE](NOTICE). Model downloads are limited to MIT, Apache-2.0, and
BSD. Non-commercial and GPL weights are refused.

| Model | Role | Licence |
|---|---|---|
| DRUNet deblocking colour | default clean, deblock | MIT (KAIR / DPIR) |
| DRUNet colour | default clean, denoise | MIT (KAIR) |
| SCUNet colour real_psnr | optional strong clean | Apache-2.0 (weights hosted by KAIR, MIT) |
| Real-ESRGAN general compact x4v3 | optional 2x | BSD-3-Clause |
| Real-ESRGAN general wdn x4v3 | optional 2x blend | BSD-3-Clause |
| Real-ESRGAN animevideov3 | optional 2x, anime | BSD-3-Clause |

## Build and run

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-gpu.txt
python -m pip install -e ".[engine]"
python -m clarity
```

`requirements-gpu.txt` installs `torch==2.14.1` from the CUDA 13.0 index.
Those wheels were checked on 2026-10-10: the Linux manylinux cp314 wheel and
the Windows win_amd64 cp314 wheel both answered HTTP 200.

The shared engine is a git dependency on
[voxprint-enlarge](https://github.com/Mitroshenkov87/voxprint-enlarge) at
tag `engine-v1.0.0`:

```text
voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0
```

The repository is private. CI installs it over SSH when the
`VOXPRINT_ENGINE_DEPLOY_KEY` secret is set, and skips that install with a
notice when the secret is absent (fork pull requests do not receive it).
Tests that do not need the engine still run. Tests that do need it run when
the package is installed.

```bash
python -m clarity --version
python -m clarity --selftest
python -m pytest
```

The desktop window is PySide6, English with a Russian translation, in the
suite dark theme. Start runs one job on a worker thread: NVDEC decode,
scene-cut chunks, restore, then a lossless join and AV1 encode with audio
stream-copy, under the suite GPU lock, with progress and cancel. Open project
and Save project read and write a `.vxclarity` file. A double-click on that
file starts Clarity and opens it. The Start menu group is **Voxprint**, shared
with the other suite apps.

Windows installer: `installer/VoxprintClarity.iss` (Inno Setup 6). It downloads
Python 3.14 and the CUDA torch wheel. `/RUNTIME=skip` installs the files
without that download and is what CI uses. Linux packaging is experimental:
`bash installer/linux/build-package.sh` writes a tarball and, when
appimagetool runs, an AppImage.

## What the pipeline does

NVDEC decode, scene-cut chunks, a temporal window per chunk, then one encode
per chunk. Default output is AV1 in an MKV (`av1_nvenc`, or SVT-AV1 only when
that encoder is already in the LGPL ffmpeg). MP4 is optional. H.265 is
`hevc_nvenc` only, and only when you ask for it. H.264 output is refused.
Finished chunks stay as Matroska. The final ffmpeg copies those video streams
and stream-copies audio, subtitles, chapters, and metadata from the source.
Opus is used only when the audio cannot be copied. HDR and Dolby Vision are
refused in this version (SDR only). Cancel, or a crash, keeps every finished
chunk and its journal. Starting the same destination again, or opening the
`.vxclarity` project, continues from the last finished chunk. The final file
is written as a `.partial` and replaces the destination only after that copy
exits cleanly. Cancel deletes the unfinished final file and the in-progress
chunk. The project format is [docs/formats/VXCLARITY.md](docs/formats/VXCLARITY.md).

GPU lock, VRAM budget, thermal pauses, the model store, the ONNX session, and
the video helpers come from `voxprint_engine.api`. Clarity does not keep a
second copy of those. The parent lock and the restore worker both use the
owner `clarity`. A restore that finds the GPU lock busy waits and retries,
and the window shows that wait. See
[docs/ENGINE-INTERFACE.md](docs/ENGINE-INTERFACE.md).

## Benchmark

On a machine with an NVIDIA GPU:

```bash
python -m pip install -r requirements.txt -r requirements-bench.txt
python -m clarity.bench
```

`scripts/bench.sh` and `scripts/bench.ps1` call the same module. The script
downloads a short Creative Commons Attribution clip (Elephants Dream) and the
three weights, exports ONNX, times PyTorch fp16 against ONNX Runtime TensorRT
EP fp16, and writes before/after PNG crops. See
[docs/BENCHMARK.md](docs/BENCHMARK.md). A rented Windows GPU VM uses
[scripts/test-gpu.ps1](scripts/test-gpu.ps1) ([docs/TESTING-GPU.md](docs/TESTING-GPU.md)).

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) — platforms, runtime, ffmpeg
- [docs/ENGINE-INTERFACE.md](docs/ENGINE-INTERFACE.md) — what Clarity needs from Enlarge
- [docs/BENCHMARK.md](docs/BENCHMARK.md) — how to time the models
- [docs/TESTING-GPU.md](docs/TESTING-GPU.md) — the rented RTX 4090 script
- [docs/formats/VXCLARITY.md](docs/formats/VXCLARITY.md) — the `.vxclarity` project file
- [docs/DECISIONS.md](docs/DECISIONS.md) — product decisions
- [docs/RESEARCH.md](docs/RESEARCH.md) — model and pipeline notes

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.

## Licence

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
