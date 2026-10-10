# Install

## Platforms

- Windows 11 24H2 or newer (build 26100). The Inno Setup script sets
  `MinVersion=10.0.26100`.
- Linux from 2025 onward (Ubuntu-style YY.MM at least 25.04, Debian 13,
  Fedora 42, or a rolling Arch). The Linux package is experimental.

Older Windows and older Linux builds are refused. There is no CPU runtime and
no fallback for a GPU below compute capability 8.9 or a driver below 600.

## Windows

Compile `installer/VoxprintClarity.iss` with Inno Setup 6. A normal install
runs `installer/install-runtime.ps1`, which:

1. Reads `installer/runtime_lock.json` (Python 3.14, torch 2.14.1, CUDA 13.0,
   driver 600, compute capability 8.9).
2. Creates a venv under `%LOCALAPPDATA%\Voxprint\runtime-<lock hash>`.
3. Installs torch from `https://download.pytorch.org/whl/cu130`.
4. Installs `requirements.txt`.
5. Junctions that venv to `{app}\runtime`.

The Start menu folder is `Voxprint`. The shortcut is **Voxprint AI Clarity**.

Silent install without the GPU runtime (CI, and any machine that should only
receive the files):

```text
VoxprintClarity-Setup.exe /VERYSILENT /RUNTIME=skip
```

A real install omits `/RUNTIME=skip`. After a runtime install the setup runs
`python main.py --register-models-user`. Uninstall runs
`--unregister-models-user`. Both need the engine package. The private engine
is not cloned by the installer; until `engine-v1.0.0` is published, register
fails with a clear message and the app files are still installed.

ffmpeg must be an LGPL build on `PATH` (NVDEC and NVENC). Clarity does not
ship ffmpeg.

## Linux

```bash
python3.14 -m pip install -r requirements.txt -r requirements-gpu.txt
python3.14 -m pip install -e ".[engine]"
python3.14 -m clarity
```

Experimental package:

```bash
bash installer/linux/build-package.sh
```

That writes `dist/VoxprintClarity-linux.tar.gz`. An AppImage is attempted when
appimagetool can be downloaded.

## Shared folders

- Models: `%LOCALAPPDATA%\Voxprint\models` on Windows,
  `$XDG_DATA_HOME/voxprint/models` or `~/.local/share/voxprint/models` on Linux.
  The engine owns the path, the SHA-256 manifests, and `.users.json`.
- GPU lock: `%TEMP%\voxprint-gpu.lock` (the system temp dir on Linux).

## Tests without a GPU

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check clarity main.py tests
python -m mypy
python -m pytest
QT_QPA_PLATFORM=offscreen python -m clarity --selftest
```

CI is GitHub Actions on `ubuntu-latest` and `windows-latest` (lint and unit
tests), plus an installer job and an experimental Linux package job.
