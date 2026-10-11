# Clarity user guide

VOXPrint AI Clarity restores an SDR video without generating new picture.
Deblocking, denoise, classical sharpening, and a per-scene colour cleanup run
on the frames you already have. Optional 2x is off unless you turn it on.

Clarity does not interpolate frames, does not offer a GAN or “enhanced” look,
does not colourise with DDColor, and does not use the NVIDIA RTX Video SDK.
Stills belong to VOXPrint AI Enlarge. HDR and Dolby Vision are refused. There
is no CPU mode and no H.264 output.

## Requirements

- Windows 11 24H2 or newer (build 26100), or a 2025-or-newer Linux distribution.
- An NVIDIA GPU from the RTX 40 series (compute capability 8.9) or newer.
- An NVIDIA driver from the 600 series or newer.
- The LGPL ffmpeg build the installer places in the shared suite folder, with
  NVDEC and NVENC. Setup downloads that build.
- A network connection during setup. The installer downloads Python 3.14,
  PyTorch, the LGPL ffmpeg build, the inference engine, and Clarity's models.

Older Windows, older Linux, a GPU below 8.9, a driver below 600, or no NVIDIA
GPU: Clarity prints the reason and exits.

## Install

### Windows

Download `VoxprintClarity-Setup.exe` from the GitHub pre-release recorded in
`BUILD.json` (published tag `v1.0.0-rc.999`). The version, the build number,
and the codename all come from that file. A codename is a Hebrew word from
the Torah or Tanakh. Check the download against
`SHA256SUMS` in that same release. The file is not code-signed, so SmartScreen
can warn.

Run the installer. During setup it installs Python 3.14 with uv and calls
`ensure_shared_runtime` with app id `clarity`. That installs the CUDA 13
wheels into `%LOCALAPPDATA%\Voxprint\shared` (or `%VOXPRINT_HOME%\shared`
when that variable is set). It does not download Enlarge's default model.
Clarity then adds the LGPL ffmpeg build and the catalogue weights it
uses. Pillow stays at 12.3.0. The engine pin is tag `engine-v1.4.0` with the
`gpu` extra, fetched as a GitHub archive so setup does not need git. Program
files go in `C:\Program Files\VOXPrint AI Suite\Clarity`. The Start menu folder
is **VOXPrint AI Suite**. The shortcut is **VOXPrint AI Clarity** and opens
`voxprint-clarity.exe`. The command line is `voxprint-clarity-cli.exe`.
Setup replaces an older install in `C:\Program Files\VoxprintClarity`, including
its Start menu shortcut and uninstall entry. It leaves
`%LOCALAPPDATA%\Voxprint` in place.

### Linux

The Linux package in the same release, `voxprint-clarity-linux.tar.gz`, is
experimental. Extract it and run `voxprint-clarity-install` once, with
permission to write `/opt`. That copies the app to
`/opt/voxprint-ai-suite/clarity` and links `voxprint-clarity` and
`voxprint-clarity-cli` onto `PATH`. If uv is
missing, that script downloads it and runs `uv python install 3.14`. It then
calls the same `ensure_shared_runtime` into
`$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared`. An older
launcher or desktop file is removed. The shared data directory is not. See
[INSTALL.md](INSTALL.md).

### Version and codename

`BUILD.json` is the only place the version, build number, and codename are
written down. The window footer and `voxprint-clarity version` read that
file. A codename is a Hebrew word from the Torah or Tanakh. The vendored
table is [RELEASES-CODENAMES.md](RELEASES-CODENAMES.md). CI checks that
Clarity's codename is not used or reserved by another app in that table.
Before a tag, update the suite table and this copy in the same commit as
`BUILD.json`. CI does not query the other repositories.

### Command line

`python -m clarity`, `voxprint-clarity`, and `voxprint-clarity-cli` take the same commands: `restore`,
`project open`, `project resume`, `info`, `version`, `models list`,
`models fetch`, and `selftest`. `--dry-run` prints the ffmpeg plan and does
not use the GPU. `--json` prints result lines on stdout. Exit codes and the
flags are in [CLI.md](CLI.md).

## First run

Open Clarity. Setup has already prepared the shared runtime, so the window
reuses the shared Python, CUDA wheels, and Clarity's models. Each launch, and
`selftest`, calls that API again. When something is missing, the engine
repairs it and Clarity shows the progress: a dialog in the window, or
`{"type": "progress", ...}` lines when you pass `--json`. Uninstall calls
`release_runtime("clarity")`.

A driver older than the R600 series is refused. The message names the driver
that was found and links to NVIDIA's driver download page,
<https://www.nvidia.com/Download/index.aspx>.

Optional 2x weights are still fetched the first time you turn 2x on. The
shared models folder checks each file’s SHA-256. Later runs reuse the files.

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

### Example: Strong on a darkened frame

Case B from the RTX 4090 run. The source is a darkened frame of Tears of
Steel (c) Blender Foundation, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
Strong (SCUNet), 2x off.

![Case B before Strong (SCUNet)](examples/crop-caseB-scunet-strong-before.png)

![Case B after Strong (SCUNet)](examples/crop-caseB-scunet-strong-after.png)

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
| Shared suite | `%LOCALAPPDATA%\Voxprint\shared` on Windows. On Linux, `$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared`. `VOXPRINT_HOME` replaces the Voxprint directory. The shared folder keeps Python 3.14, the CUDA 13 wheels, and the ref list in `manifest.json`. Clarity registers its own weights there and adds the LGPL ffmpeg binary under `shared/ffmpeg`. |
| Shared models | `%LOCALAPPDATA%\Voxprint\models` on Windows. On Linux, `$XDG_DATA_HOME/voxprint/models` or `~/.local/share/voxprint/models`. `voxprint_common.paths` owns the folder. The engine store owns the hashes and `.users.json`. |
| Recent projects | `%LOCALAPPDATA%\Voxprint\Clarity\recent.json` on Windows. On Linux, `$XDG_DATA_HOME/voxprint/clarity/recent.json` or `~/.local/share/voxprint/clarity/recent.json`. |
| Scene chunks and the journal | `<project>.vxclarity.work` beside the project (`plan.json`, `journal.jsonl`, `chunks/`). |
| GPU lock | `%TEMP%\voxprint-gpu.lock` (the system temp directory on Linux). |
| Program files (Windows) | `C:\Program Files\VOXPrint AI Suite\Clarity` (`voxprint-clarity.exe`, `voxprint-clarity-cli.exe`) |
| Program files (Linux) | `/opt/voxprint-ai-suite/clarity` (`voxprint-clarity`, `voxprint-clarity-cli`) |
| Windows setup log | `%TEMP%\voxprint-clarity-setup.log` |

Clarity does not write its own application log. The status line in the window
is the record of the current job. The setup log is the record of the install.

## Troubleshooting

**The program prints a GPU or system message and exits.** The machine failed
the gate. The text names the cause: the Windows or Linux version is too old,
there is no NVIDIA GPU, compute capability is below 8.9, or the driver is
below the 600 series. Clarity will not start a CPU job. A driver older than
the R600 series includes a link to
<https://www.nvidia.com/Download/index.aspx>. Update the driver or use an
RTX 40 series GPU or newer. A missing engine package says
`VOXPrint inference engine missing, run repair.`

**The status says the GPU is busy.** Another suite job holds the shared lock.
Clarity waits and retries. It does not cancel your restore for that wait.
Cancel still stops the job when you ask it to.

**The job pauses and then continues.** Between scene chunks, Clarity follows
the suite thermal rule: after a long run it pauses while the GPU stays hot or
throttled, then continues when the card has cooled. The finished chunks stay
on disk. You do not need to press Start again.

**A chunk is redone.** The chunk file is missing, or its bytes no longer match
the journal. That chunk runs again. The others stay.

**ffmpeg is missing, or AV1 encode fails.** Setup places an LGPL ffmpeg build
in the shared suite folder and puts that directory on `PATH`. Launch repairs
the binary when it is missing. The build needs `av1_nvenc`. Clarity downloads
the LGPL build.

## Model licences

The application is Apache-2.0. Third-party weights and libraries are listed in
[NOTICE](https://github.com/Mitroshenkov87/voxprint-clarity/blob/main/NOTICE).

| Model | Role | Licence |
|---|---|---|
| DRUNet deblocking colour | Clean, deblock | MIT (KAIR / DPIR) |
| DRUNet colour | Clean, denoise | MIT (KAIR) |
| SCUNet colour real_psnr | Strong clean | Apache-2.0 (weights hosted by KAIR, MIT) |
| Real-ESRGAN general compact x4v3 | Optional 2x | BSD-3-Clause |
| Real-ESRGAN general wdn x4v3 | Optional 2x blend | BSD-3-Clause |
| Real-ESRGAN animevideov3 | Optional 2x, anime | BSD-3-Clause |
