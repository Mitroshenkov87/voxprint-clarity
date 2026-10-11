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

1. Reads `installer/runtime_lock.json` (Pillow 12.3.0, `onnxruntime-gpu`
   1.31.0, TensorRT 10.14.1.48).
2. Creates a venv under `%LOCALAPPDATA%\Voxprint\shared\venv`
   (`%VOXPRINT_HOME%\shared\venv` when `VOXPRINT_HOME` is set). An existing
   venv is reused.
3. Installs `requirements.txt` and the engine at commit `d11d609`
   (`voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@d11d60921448b697d832b2699e6e3e04c62c6dfd`).
   The pin moves to tag `engine-v1.1.0` when that tag is published.
4. Calls `ensure_runtime("clarity")`. The engine installs Python 3.14, the
   CUDA 13 wheels, and its default model into the shared folder.
5. Uninstalls the CPU `onnxruntime` wheel and TensorRT 11, then installs the
   locked `onnxruntime-gpu` and `tensorrt-cu13-libs` pair. `tensorrt-cu13`
   10.14.1.48 is installed with `--no-deps` so the engine probe does not pull
   TensorRT 11 on the next repair. TensorRT 10.14 bindings have no cp314 wheel.
6. Installs `Pillow==$pillow` from the lock after that step.
7. Creates `{app}` if needed and junctions the venv to `{app}\runtime`. The
   link is checked: a failed `mklink` stops the install, and a `runtime` path
   that is not a junction is refused.

`PrepareToInstall` runs before `{app}` exists, so the setup runs the same
script again with `-LinkOnly` after the files are copied. That second step
only creates the junction. The shortcut and `--register-models-user` use
`{app}\runtime\Scripts\python.exe`.

The Start menu folder is `Voxprint`. The shortcut is **Voxprint AI Clarity**.

The installer registers `.vxclarity` for the current user (HKCU). The ProgID is
`Voxprint.Clarity.Project`. The document icon is `branding\clarity.ico`. Open
runs `pythonw main.py` with the project path. Uninstall deletes that ProgID
and the extension entry.

Silent install without the GPU runtime (CI, and any machine that should only
receive the files):

```text
VoxprintClarity-Setup.exe /VERYSILENT /RUNTIME=skip
```

A real install omits `/RUNTIME=skip`. After the runtime step the setup runs
`python main.py --install-suite`, which calls `ensure_runtime("clarity")`
again, downloads the LGPL ffmpeg build when one is missing, and fetches
Clarity's Clean and Strong weights through the engine registry. It then runs
`python main.py --register-models-user`. Uninstall runs
`--unregister-models-user`, which calls `release_runtime("clarity")` and
unregisters the models user.

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
appimagetool can be downloaded. The package includes
`usr/share/mime/packages/voxprint-clarity.xml`, a `.desktop` file whose
`MimeType` is `application/vnd.voxprint.clarity+zip`,
`usr/bin/voxprint-clarity`, and `usr/bin/voxprint-clarity-install`.

Run `voxprint-clarity-install` once after extracting the package, and once
for an AppImage before the first restore. The script needs Python 3.14,
installs the engine at commit `d11d609` and Pillow 12.3.0, and runs
`--install-suite`. That calls `ensure_runtime("clarity")`, repairs ffmpeg,
and fetches Clarity's Clean and Strong weights. It then applies the same
ONNX Runtime 1.31 and TensorRT 10.14 pin as the Windows installer, read from
`runtime_lock.json`. A second run reuses the venv.
The launcher and `AppRun` prepend `shared/ffmpeg` to `PATH`.

After extracting onto a prefix, update the MIME and desktop databases.
`voxprint-clarity-uninstall` drops Clarity's suite references when `main.py`
is beside the script, and removes the association, the desktop file, and the
icon from that prefix (default `/usr`).

## Shared folders

- Suite runtime: `%LOCALAPPDATA%\Voxprint\shared` on Windows,
  `$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared` on Linux.
  `VOXPRINT_HOME` replaces the Voxprint directory. The engine owns
  `manifest.json` and the holder list.
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
VOXPRINT_CI_LANCZOS=1 QT_QPA_PLATFORM=offscreen python -m clarity --selftest
```

`VOXPRINT_CI_LANCZOS=1` makes `ensure_runtime` skip the network, which is how
CI runs `--selftest`. `assess_launch` still refuses that flag for a restore.

CI is GitHub Actions on `ubuntu-latest` and `windows-latest` (lint and unit
tests, with the public engine installed), plus an installer job and an
experimental Linux package job. A tag `v1.0.0-rc.999` runs
`.github/workflows/release.yml`, which publishes the full Windows installer
and the Linux package as a pre-release with `SHA256SUMS`.
