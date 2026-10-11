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
   Pillow 12.3.0, driver 600, compute capability 8.9).
2. Creates a venv under `%LOCALAPPDATA%\Voxprint\shared\runtime\py3.14-torch2.14.1-cu130`
   (`%VOXPRINT_HOME%\shared\...` when `VOXPRINT_HOME` is set). An existing venv
   with the same pins is reused.
3. Installs torch from `https://download.pytorch.org/whl/cu130`.
4. Installs `requirements.txt`, then Pillow at the locked version, then
   `requirements-bench.txt` (spandrel and `onnxruntime-gpu`).
5. Installs the public engine
   `voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0`,
   then installs `onnxruntime-gpu` again so the engine's CPU wheel does not
   replace it.
6. Downloads the LGPL ffmpeg build into `shared\ffmpeg` when `ffmpeg.exe` is
   absent.
7. Writes ready markers and increments Clarity's count in `shared\manifest.json`.
8. Junctions that venv to `{app}\runtime`.

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

A real install omits `/RUNTIME=skip`. After a runtime install the setup runs
`python main.py --install-suite`, which fetches the default models into the
engine models folder, then `python main.py --register-models-user`. Uninstall
runs `--unregister-models-user`, which also drops Clarity's shared-suite
counts. Both use the engine package at tag `engine-v1.0.0`, which the runtime
step installs from the public Enlarge repository.

The LGPL ffmpeg build is installed into the shared folder during setup. Launch
puts that directory on `PATH`.

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
for an AppImage before the first restore. The script installs Python 3.14,
torch 2.14.1+cu130, the LGPL ffmpeg build, and the engine into the shared
suite folder, then runs `--install-suite` for the default models. A second
run reuses the venv and the ffmpeg binary. The launcher and `AppRun` prepend
`shared/ffmpeg` to `PATH` and do not download.

After extracting onto a prefix, update the MIME and desktop databases.
`voxprint-clarity-uninstall` drops Clarity's suite references when `main.py`
is beside the script, and removes the association, the desktop file, and the
icon from that prefix (default `/usr`).

## Shared folders

- Suite runtime: `%LOCALAPPDATA%\Voxprint\shared` on Windows,
  `$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared` on Linux.
  `VOXPRINT_HOME` replaces the Voxprint directory. `manifest.json` holds the
  per-app refcount. See [UPSTREAM-SHARED.md](UPSTREAM-SHARED.md) for the engine
  addition this layout is waiting on.
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
QT_QPA_PLATFORM=offscreen python -c "from clarity.suite_repair import write_ready_markers; write_ready_markers()"
QT_QPA_PLATFORM=offscreen python -m clarity --selftest
```

`write_ready_markers` is the CI stand-in for a finished install. It lets
`--selftest` re-check the suite without downloading torch or the catalogue.
A real setup does not call it.

CI is GitHub Actions on `ubuntu-latest` and `windows-latest` (lint and unit
tests, with the public engine installed), plus an installer job and an
experimental Linux package job. A tag `v1.0.0-rc.999` runs
`.github/workflows/release.yml`, which publishes the full Windows installer
and the Linux package as a pre-release with `SHA256SUMS`.
