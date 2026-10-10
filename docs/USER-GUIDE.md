# Clarity user guide

Voxprint AI Clarity restores an SDR video without generating new picture.
Deblocking, denoise, classical sharpening, and a per-scene colour cleanup run
on the frames you already have. Optional 2x is off unless you turn it on.

Clarity does not interpolate frames, does not offer a GAN or “enhanced” look,
does not colourise with DDColor, and does not use the NVIDIA RTX Video SDK.
Stills belong to Voxprint AI Enlarge. HDR and Dolby Vision are refused. There
is no CPU mode and no H.264 output.

## Requirements

- Windows 11 24H2 or newer (build 26100), or a 2025-or-newer Linux distribution.
- An NVIDIA GPU from the RTX 40 series (compute capability 8.9) or newer.
- An NVIDIA driver from the 600 series or newer.
- An LGPL ffmpeg on `PATH`, with NVDEC and NVENC. Clarity does not ship ffmpeg,
  x264, or x265.
- A network connection the first time a model is needed, and during setup on
  Windows, because the installer downloads Python and PyTorch.

Older Windows, older Linux, a GPU below 8.9, a driver below 600, or no NVIDIA
GPU: Clarity prints the reason and exits.

## Install

### Windows

Download `VoxprintClarity-Setup.exe` from the GitHub pre-release
**1.0.0 RC · build 999 "Bereshit"** (`v1.0.0-rc.999`). Check it against
`SHA256SUMS` in that same release. The file is not code-signed, so SmartScreen
can warn.

Run the installer. It downloads Python 3.14, PyTorch 2.14.1 for CUDA 13.0, the
restore libraries, and the public engine
`voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0`.
The Start menu folder is **Voxprint**, shared with the other suite apps. The
shortcut is **Voxprint AI Clarity**.

Put an LGPL ffmpeg on `PATH` before the first restore.

### Linux

The Linux package in the same release, `VoxprintClarity-linux.tar.gz`, is
experimental. Extract it, install Python 3.14, PyTorch for CUDA 13.0, and the
engine line above, and put ffmpeg on `PATH`. See [INSTALL.md](INSTALL.md).

## First run

Open Clarity. The window does not download weights until you press Start.
The first Start for a mode fetches that mode’s weights into the shared models
folder and checks each file’s SHA-256. Later runs reuse the files.

Weights are only MIT, Apache-2.0, or BSD. The catalogue is in the table at
the end of this guide.

## A restore

1. Choose the source video.
2. Leave the mode on **Clean**, or switch to **Strong clean** when the picture
   needs the slower blind model.
3. Leave **2x** off unless you want a larger frame. It is off by default.
   Anime is available only when 2x is on.
4. The output path defaults to an AV1 MKV beside the source
   (`name.clarity.mkv`). Audio, subtitles, chapters, and metadata are copied.
   MP4 is optional. HEVC is available only as `hevc_nvenc`, and only if you
   choose it.
5. Press Start. Progress is the finished frames over the whole clip. Cancel
   keeps every finished scene chunk.

The final file is written as a `.partial` and replaces the destination only
after the join finishes. Cancel removes the unfinished final file and the
chunk that was in progress.

## Projects and resume

**Save project** writes a `.vxclarity` file. **Open project** reads it back.
Double-clicking that file starts Clarity and opens it. The recent list holds
the last few projects.

A project remembers the source, the settings, the output path, and which scene
chunks already finished. Opening it, or starting the same destination again,
continues from the last chunk whose file still matches the journal. Changing
the settings or the source starts the chunks over.

Chunk media stays in a work folder beside the project
(`name.vxclarity.work`), not inside the `.vxclarity` file. The format is
[formats/VXCLARITY.md](formats/VXCLARITY.md).

## Where files live

| What | Where |
|---|---|
| Shared models | `%LOCALAPPDATA%\Voxprint\models` on Windows. On Linux, `$XDG_DATA_HOME/voxprint/models` or `~/.local/share/voxprint/models`. The engine owns the folder, the hashes, and `.users.json`. |
| Recent projects | `%LOCALAPPDATA%\Voxprint\Clarity\recent.json` on Windows. On Linux, `$XDG_DATA_HOME/voxprint/clarity/recent.json` or `~/.local/share/voxprint/clarity/recent.json`. |
| Scene chunks and the journal | `<project>.vxclarity.work` beside the project (`plan.json`, `journal.jsonl`, `chunks/`). |
| GPU lock | `%TEMP%\voxprint-gpu.lock` (the system temp directory on Linux). |
| Windows setup log | `%TEMP%\VoxprintClarity-setup.log` |

Clarity does not write its own application log. The status line in the window
is the record of the current job. The setup log is the record of the install.

## Troubleshooting

**The program prints a GPU or system message and exits.** The machine failed
the gate. The text names the cause: the Windows or Linux version is too old,
there is no NVIDIA GPU, compute capability is below 8.9, or the driver is
below the 600 series. Clarity will not start a CPU job. Update the driver or
use an RTX 40 series GPU or newer. A missing engine package prints the
`voxprint-engine @ git+https://...@engine-v1.0.0` install line instead.

**The status says the GPU is busy.** Another suite job holds the shared lock.
Clarity waits and retries. It does not cancel your restore for that wait.
Cancel still stops the job when you ask it to.

**The job pauses and then continues.** Between scene chunks, Clarity follows
the suite thermal rule: after a long run it pauses while the GPU stays hot or
throttled, then continues when the card has cooled. The finished chunks stay
on disk. You do not need to press Start again.

**A chunk is redone.** The chunk file is missing, or its bytes no longer match
the journal. That chunk runs again. The others stay.

**ffmpeg is missing, or AV1 encode fails.** Install an LGPL ffmpeg that
includes `av1_nvenc` and put it on `PATH`. A GPL build is not what this
project ships or asks for.

## Model licences

The application is Apache-2.0. Third-party weights and libraries are listed in
[NOTICE](../NOTICE).

| Model | Role | Licence |
|---|---|---|
| DRUNet deblocking colour | Clean, deblock | MIT (KAIR / DPIR) |
| DRUNet colour | Clean, denoise | MIT (KAIR) |
| SCUNet colour real_psnr | Strong clean | Apache-2.0 (weights hosted by KAIR, MIT) |
| Real-ESRGAN general compact x4v3 | Optional 2x | BSD-3-Clause |
| Real-ESRGAN general wdn x4v3 | Optional 2x blend | BSD-3-Clause |
| Real-ESRGAN animevideov3 | Optional 2x, anime | BSD-3-Clause |
