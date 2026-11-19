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

1. Reads `installer/runtime_lock.json` (Pillow 12.3.0).
2. Creates a venv under `%LOCALAPPDATA%\Voxprint\shared\venv`
   (`%VOXPRINT_HOME%\shared\venv` when `VOXPRINT_HOME` is set). An existing
   venv is reused. Reuse still installs Clarity's torch and spandrel into
   that venv, and it installs `voxprint-common` and the engine extra again.
3. Downloads `voxprint-common` 0.1.1 "Menuchah" from the release wheel
   (`voxprint-common @ https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.1.1/voxprint_common-0.1.1-py3-none-any.whl`),
   checks it against the published `.sha256` file and `runtime_lock.json`,
   and installs that verified wheel. It then installs `requirements.txt` when
   the venv is new, omitting the `voxprint-common` line so pip does not fetch
   the wheel a second time. The engine at tag `engine-v1.4.0`
   (`voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.4.0.tar.gz`)
   is installed for a new venv and a reused one. The archive has no
   subdirectory, and the install does not need git. The `gpu` extra is
   installed with `--extra-index-url https://pypi.nvidia.com`. The engine's
   base dependencies do not include ONNX Runtime. The GPU pins are that extra.
   `installer/stage_suite_runtime.py` then copies `enlarge` from the same tag
   into site-packages, because the wheel ships `voxprint_engine` only.
4. Calls `enlarge.suite_runtime.ensure_runtime("clarity")`. That writes the
   app id into the component refs and does not install Enlarge's default
   model. This step runs before `{app}` exists, so it does not import the
   Clarity package. `--install-suite`, after the files are copied, repeats
   that call and fetches the catalogue weights Clarity uses.
5. Runs `installer/pin_runtime.py` with the venv interpreter, including when
   the venv already existed. That installs `torch==2.14.1` from
   `requirements-gpu.txt` and `spandrel`. If the CPU `onnxruntime`
   distribution is present, the pin uninstalls it. The pin then opens a real
   TensorRT session and a real CUDA session and fails when either one does
   not stay active. `Pillow==$pillow` from the lock is
   installed after that.
6. Creates `{app}` if needed and junctions the venv to `{app}\runtime`. The
   link is checked: a failed `mklink` stops the install, and a `runtime` path
   that is not a junction is refused.

`PrepareToInstall` runs before `{app}` exists, so the setup runs the same
script again with `-LinkOnly` after the files are copied. That second step
only creates the junction. `--register-models-user` uses
`{app}\runtime\Scripts\python.exe`. The shortcut uses `voxprint-clarity.exe`,
which starts `pythonw.exe` from that junction. Compile
`installer/launchers/build.ps1` before Inno Setup so those executables exist.

The install directory is `C:\Program Files\VOXPrint AI Suite\Clarity`
(`{autopf}\VOXPrint AI Suite\Clarity`). The directory page is off, so a normal
install uses that folder. `/DIR` still overrides it for the CI file check.
The Start menu folder is `VOXPrint AI Suite`. The shortcut is **VOXPrint AI Clarity**
and runs `voxprint-clarity.exe`. `voxprint-clarity-cli.exe` is the command-line
program. Both are added to the user `PATH`. Peers find Clarity at
`HKCU\Software\Voxprint\Clarity` (`InstallPath`, `Exe`, `Version`, an empty
`IpcName`, `ProtocolVersion` 0) and at
`HKCU\Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-clarity.exe`.
`voxprint-common` 0.1.1 does not ship a registry helper, so the installer writes
those keys. Clarity has no IPC server, so `IpcName` is empty and
`ProtocolVersion` is 0.

An upgrade keeps the same AppId and does not reuse the previous directory.
It removes `C:\Program Files\VoxprintClarity` when that folder is still present,
and it removes any other install directory recorded on the old uninstall entry
when that path is not the new folder and is not under `%LOCALAPPDATA%`. The
`runtime` junction is unlinked before the old tree is deleted, so the shared
venv in `%LOCALAPPDATA%\Voxprint` stays. Old Start menu shortcuts in the
`Voxprint` folder are removed. The folder itself is removed only when it is
empty. The user `PATH` drops the old directory and gains the new one.

The installer registers `.vxclarity` for the current user (HKCU). The ProgID is
`Voxprint.Clarity.Project`. The document icon is `branding\clarity.ico`. Open
runs `voxprint-clarity.exe` with the project path. Uninstall deletes that ProgID
and the extension entry. The visible file type is **VOXPrint AI Clarity project**.

Silent install without the GPU runtime (CI, and any machine that should only
receive the files):

```text
VoxprintClarity-Setup.exe /VERYSILENT /RUNTIME=skip
```

A real install omits `/RUNTIME=skip`. After the runtime step the setup runs
`python main.py --install-suite`, which calls `ensure_shared_runtime`
again, downloads the LGPL ffmpeg build when one is missing, and fetches
every catalogue weight Clarity uses. `ensure_runtime("clarity")` does not
install Enlarge's default model. It then runs
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

That writes `dist/voxprint-clarity-linux.tar.gz`. An AppImage is attempted when
appimagetool can be downloaded (`dist/voxprint-clarity-x86_64.AppImage`). The
package includes `share/mime/packages/voxprint-clarity.xml`, the discovery
desktop file `ai.voxprint.Clarity.desktop` (`Name=VOXPrint AI Clarity`,
`X-Voxprint-Ipc`, `X-Voxprint-Protocol=0`, `X-Voxprint-Version`),
`voxprint-clarity`, `voxprint-clarity-cli`, and `voxprint-clarity-install`.

Run `voxprint-clarity-install` once after extracting the package. It copies
the tree to `/opt/voxprint-ai-suite/clarity`, links both programs into
`/usr/local/bin`, and writes the desktop file for the current user and under
`/usr/local/share/applications`. Writing `/opt` needs root; a normal account
is asked for `sudo` for that step only. The shared runtime still goes to the
invoking user's `~/.local/share/voxprint`. Run it once
for an AppImage before the first restore. When Python 3.14 or uv is missing,
the script downloads uv and runs `uv python install 3.14`. It installs the
engine at tag `engine-v1.4.0`
(`voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.4.0.tar.gz`)
with `--extra-index-url https://pypi.nvidia.com`, verifies the
`voxprint-common` 0.1.1 wheel against its `.sha256` file, copies `enlarge`
from the engine archive, and installs Pillow 12.3.0, then runs
`--install-suite`. That calls `ensure_shared_runtime`, which records app id
`clarity` and does not install Enlarge's default model, repairs ffmpeg, and fetches every catalogue
weight Clarity uses. It then runs `pin_runtime.py`, so an existing venv still
receives `torch==2.14.1` and spandrel. If the CPU `onnxruntime` distribution
is present, the pin uninstalls it. The pin then opens a real TensorRT session
and a real CUDA session and fails when either one does not stay active.
A second run reuses the venv, installs the engine again, and pins torch again.

The launcher and `AppRun` prepend `shared/ffmpeg` to `PATH` and prefer the
shared venv's Python when that interpreter exists.

`voxprint-clarity-install` removes an older `/usr/bin/voxprint-clarity`, the
old `voxprint-clarity.desktop` shortcut, and a leftover `/opt/VoxprintClarity`
directory. It does not delete `~/.local/share/voxprint`.
`voxprint-clarity-uninstall` drops Clarity's suite references when `main.py`
is beside the script, removes `/opt/voxprint-ai-suite/clarity`, the PATH
links, the association, and the desktop file. User data stays.

## Shared folders

- Suite runtime: `%LOCALAPPDATA%\Voxprint\shared` on Windows,
  `$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared` on Linux.
  `VOXPRINT_HOME` replaces the Voxprint directory. `enlarge.suite_runtime`
  owns `manifest.json` and the ref list.
- Models: `%LOCALAPPDATA%\Voxprint\models` on Windows,
  `$XDG_DATA_HOME/voxprint/models` or `~/.local/share/voxprint/models` on Linux.
  `voxprint_common.paths` owns the directory. The engine store owns the SHA-256 manifests and `.users.json`.
- GPU lock: `%TEMP%\voxprint-gpu.lock` (the system temp dir on Linux).

## Tests without a GPU

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check clarity main.py tests scripts
python -m mypy
python -m pytest
QT_QPA_PLATFORM=offscreen python -m clarity --selftest --no-gpu
```

`--no-gpu` is the selftest CI and the release workflow both run. It checks
the window, the engine import, and the restore plan. For that process the
engine skips the suite download. `assess_launch` still refuses
`VOXPRINT_CI_LANCZOS` for a restore. A GPU selftest also refuses a venv that
cannot import torch and spandrel, then opens a real TensorRT session and a
real CUDA session. `--no-gpu` does not.

CI is GitHub Actions on `ubuntu-latest` and `windows-latest` (lint and unit
tests, with the public engine installed), plus an installer job and an
experimental Linux package job. A tag that starts with `v` runs
`.github/workflows/release.yml`. The publish step copies the built installer
to `voxprint-clarity-setup-<version>-win-x64.exe` and the Linux tarball to
`voxprint-clarity_<version>_linux-x64.tar.gz` (a `.deb`, when the job built
one, is `voxprint-clarity_<version>_amd64.deb`). `SHA256SUMS` lists those
names. A tag that contains `-rc` is a pre-release. A plain `vX.Y.Z` tag is a
normal release. The Inno output file stays `VoxprintClarity-Setup.exe`.
