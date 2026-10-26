# VOXPrint AI Clarity

> [!WARNING]
> **⚠️ Deep alpha.** This software is experimental; nothing is guaranteed to work. Use at your own risk.
>
> **⚠️ Глубокая альфа.** Программа экспериментальная, работа не гарантируется.

Lightweight non-generative restoration for an SDR video, part of the VOXPrint
AI Media Suite. A clip comes back deblocked, denoised, lightly sharpened, and
colour-cleaned per scene. Optional 2x is off by default.

Clarity does not interpolate frames, does not add a GAN or “enhanced” look,
does not use DDColor, and does not use the NVIDIA RTX Video SDK. HDR and
Dolby Vision are refused. There is no CPU mode.

The longer guide is [docs/USER-GUIDE.md](docs/USER-GUIDE.md).

## Requirements

- Windows 11 24H2 or newer, or a 2025-or-newer Linux distribution.
- An NVIDIA RTX 40 series GPU or newer (compute capability 8.9).
- An NVIDIA driver from the 600 series or newer.
- An LGPL ffmpeg on `PATH`, with NVDEC and NVENC.

## Install

Download the release for the version recorded in `BUILD.json`. A tag that
contains `-rc` is a pre-release. A plain `vX.Y.Z` tag is the stable release.
The version, build number, and codename in that file are
the source of truth. A codename is a Hebrew word from the Torah or Tanakh.
The Windows file is `voxprint-clarity-setup-<version>-win-x64.exe`
(tag `v1.0.0` publishes `voxprint-clarity-setup-1.0.0-win-x64.exe`).
Check `SHA256SUMS` in that release. Setup downloads Python 3.14, the CUDA 13.0
PyTorch wheel, the restore libraries, `voxprint-common` 0.1.1 "Menuchah"
from its release wheel (checked against the published `.sha256` file), and
the public engine at tag `engine-v1.4.0`. The installer is not code-signed.
The shared library is not a git clone.

Windows installs into `C:\Program Files\VOXPrint AI Suite\Clarity`. The Start
menu folder is **VOXPrint AI Suite** and the shortcut is **VOXPrint AI Clarity**.
The window program is `voxprint-clarity.exe`. The command line is
`voxprint-clarity-cli.exe`. An older install under `C:\Program Files\VoxprintClarity`
is removed on upgrade. Your files in `%LOCALAPPDATA%\Voxprint` stay where they are.
Put ffmpeg on `PATH` before the first restore. Linux is the experimental
tarball `voxprint-clarity_<version>_linux-x64.tar.gz` in the same release.
Tag `v1.0.0` publishes `voxprint-clarity_1.0.0_linux-x64.tar.gz`. It installs
into `/opt/voxprint-ai-suite/clarity`. Details:
[docs/USER-GUIDE.md](docs/USER-GUIDE.md) and [docs/INSTALL.md](docs/INSTALL.md).

## First run and a basic restore

The first Start downloads the weights for the mode you chose into the shared
models folder and checks their SHA-256.

1. Open a video.
2. Choose **Clean** or **Strong clean**. Leave 2x off unless you want it.
3. Start. The default output is an AV1 MKV beside the source, with audio copied.

**Save project** writes a `.vxclarity` file. Open it again, or start the same
destination, and finished scene chunks are kept. Cancel does not throw those
chunks away.

## Where files live

- Models: `%LOCALAPPDATA%\Voxprint\models` (Linux: `~/.local/share/voxprint/models`).
- Resume data: a `.work` folder beside the project, plus the `.vxclarity` file.
- Windows setup log: `%TEMP%\voxprint-clarity-setup.log`.

Clarity does not write a separate application log. The window status line is
the job’s progress.

## Troubleshooting

- **A message and an exit at launch.** The GPU gate refused the machine. You
  need an RTX 40 series GPU or newer and a 600-series driver. There is no CPU
  fallback.
- **“The GPU is busy”.** Another suite job holds the lock. Clarity waits and
  retries.
- **A pause in the middle of a long job.** The thermal rule paused between
  scene chunks. Finished chunks stay, and the job continues when the GPU cools.

More cases are in [docs/USER-GUIDE.md](docs/USER-GUIDE.md).

## Roadmap

What is next, what can wait, and what Clarity will not add: [ROADMAP.md](ROADMAP.md).

## Documentation

The documentation site is published from `main`:
<https://mitroshenkov87.github.io/voxprint-clarity/>.

- [docs/USER-GUIDE.md](docs/USER-GUIDE.md) — install, first run, restore, resume
- [docs/INSTALL.md](docs/INSTALL.md) — platforms, runtime, ffmpeg
- [docs/ENGINE-INTERFACE.md](docs/ENGINE-INTERFACE.md) — the shared inference engine
- [docs/BENCHMARK.md](docs/BENCHMARK.md) — how to time the models
- [docs/TESTING-GPU.md](docs/TESTING-GPU.md) — the rented RTX 4090 script
- [docs/formats/VXCLARITY.md](docs/formats/VXCLARITY.md) — the `.vxclarity` project file
- [docs/CLI.md](docs/CLI.md) — headless `python -m clarity` / `voxprint-clarity-cli`
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, data flow, and the GPU rules
- [docs/RELEASES-CODENAMES.md](docs/RELEASES-CODENAMES.md) — suite codenames
- [ROADMAP.md](ROADMAP.md) — next version, later work, and what is not planned
- [docs/DECISIONS.md](docs/DECISIONS.md) — product decisions
- [docs/RESEARCH.md](docs/RESEARCH.md) — model and pipeline notes

From a checkout, developers install with `requirements-dev.txt`,
`requirements-gpu.txt`, and
`pip install --extra-index-url https://pypi.nvidia.com -e ".[engine]"`.
The engine line is
`voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.4.0.tar.gz`.
Model weights are MIT, Apache-2.0, or BSD only. The table is in
[docs/USER-GUIDE.md](docs/USER-GUIDE.md).

## Example

Strong (SCUNet) on a darkened frame from Tears of Steel, restored on an RTX 4090.
Tears of Steel (c) Blender Foundation, CC BY 3.0. 2x was off.

Before:

![Case B before Strong (SCUNet)](docs/examples/crop-caseB-scunet-strong-before.png)

After:

![Case B after Strong (SCUNet)](docs/examples/crop-caseB-scunet-strong-after.png)

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.

## Licence

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
