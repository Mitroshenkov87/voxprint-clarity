# Architecture

Clarity is one Python package, `clarity`, plus the window entry `main.py`.
Video frames stay the frames you started with: the models denoise, deblock,
and optionally upscale. They do not invent a new picture.

## Components

| Piece | Role |
|---|---|
| `clarity.cli` / `clarity.ui.main_window` | The command line and the Qt window. Both call `run_job`. |
| `clarity.pipeline.runner` | One restore: probe, plan, chunk loop, concat. `dry_run_restore` prints the plan and stops. |
| `clarity.pipeline.job` | Settings, the model list, and the per-chunk pixel steps. |
| `clarity.pipeline.probe` | ffprobe JSON, and the SDR check. |
| `clarity.pipeline.scenes` | Scene cuts, then a split when a scene is longer than `max_frames` (default 1000). |
| `clarity.pipeline.encode` | NVDEC decode, NVENC encode, and the concat that joins chunks. |
| `clarity.pipeline.resume` | `plan.json`, `journal.jsonl`, and the published chunk files. |
| `clarity.project` | Read and write a `.vxclarity` file. |
| `clarity.models` | The six permissive catalogue weights. |
| `clarity.engine` | Imports `voxprint_engine` for video, VRAM, and the model store. Paths, the GPU lock, thermals, and the GPU probe come from `voxprint_common`. |
| `clarity.platform_gate` | Windows 11 24H2+ and the 2025 Linux releases. |
| `clarity.ui.theme` | Re-exports the shared theme and holds Clarity's splash identity colour. |
| `clarity.bench` | Times the catalogue on CUDA and TensorRT. |

## Data flow

A live restore runs these steps in order:

1. **Decode the facts.** ffprobe reads streams, format, and chapters. HDR and Dolby Vision stop here. The scene filter then runs on the CPU (`-vf select=gt(scene,…)`), with no CUDA decode, and the cuts become chunks.
2. **Scene chunks.** Each chunk is a half-open frame range. A finished chunk is skipped when the journal still matches the file on disk.
3. **Engine.** For each unfinished chunk the job holds the suite GPU lock, asks the engine for a tile and a thermal decision, and restores the frames. `host_batch_count` keeps the source window inside 512 MiB. `voxprint_engine.api.iter_tile_frames` restores those frames one at a time, so the engine accumulator is one frame. Each batch is reduced to uint8 before the next one starts. Every catalogue model, including default Clean (DRUNet), is converted to a cached fp16 ONNX file and run on TensorRT, retrying on CUDA when the engine build fails. DRUNet's manifest sets `in_channels` to 4 so the export example is `1x4xHxW`; the noise-level map is that fourth channel, and the catalogue precision stays fp32.
4. **Encode.** The chunk is decoded with NVDEC (`-hwaccel cuda`) to raw rgb24, restored, and encoded with `av1_nvenc` (or `hevc_nvenc` when HEVC was requested). The output pixel format is `yuv420p` for an 8-bit source and `p010le` when the source is 10-bit or deeper. `libsvtav1` is used only when this ffmpeg has it and `av1_nvenc` is absent. The chunk is renamed into place and one journal line records its SHA-256.
5. **Join.** ffmpeg's concat demuxer copies the chunk video (`-c copy`) and copies the source audio into the destination. The file is written to a `.partial` path and renamed into place only after every chunk is in that file.

`--dry-run` runs the probe and the CPU scene filter, then prints the same command lists. It does not take the GPU lock, load a weight, or start NVDEC or NVENC.

## Engine

Clarity does not reimplement the VRAM budget, the model store, or the ONNX session. Those stay in `voxprint_engine` at tag `engine-v1.4.0`. Paths, the GPU lock, the thermal rule, the GPU probe, and the licence allow-list live in `voxprint_common` 0.1.1. The shared runtime installer is `enlarge.suite_runtime` in that same tag. The engine wheel packages `voxprint_engine` only, so setup copies the `enlarge` package from the tag archive into site-packages. The package metadata is at the repository root, so the archive URL has no subdirectory. The `gpu` extra holds the ONNX Runtime GPU and TensorRT pins. Installing it needs `--extra-index-url https://pypi.nvidia.com`.

```text
voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.4.0.tar.gz
```

`clarity.engine` loads the engine with `importlib` and forwards the calls. The requirement string is `ENGINE_REQUIREMENT`. The lock app id and the runtime app id are `OWNER` (`clarity`). A `restore_batch` message still sends `owner` `clarity`. API stays 1. Launch calls `voxprint_common.gpu_probe.assess` with the display name `Voxprint AI Clarity` and `refuse_below=(8, 9)`.

## Shared dependencies

The installer, every launch, and `selftest` call `ensure_runtime("clarity")` through `clarity.engine.ensure_shared_runtime`. Uninstall calls `release_runtime("clarity")`. `ensure_runtime` writes `clarity` into the component refs and installs Python 3.14, the CUDA 13 torch, ONNX Runtime GPU, and TensorRT wheels. It does not download Enlarge's default model. `ensure_default_model` installs that weight only for app id `enlarge`, and Clarity does not call it. Catalogue weights are fetched separately, under `clarity` in the shared models folder. A payload is removed when its ref list is empty.

The folder is `%LOCALAPPDATA%\Voxprint\shared` on Windows and `$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared` on Linux. `VOXPRINT_HOME` replaces the Voxprint directory.

Clarity's Clean and Strong weights go through the engine registry (`ensure_weight`, `register`, `unregister`). ffmpeg is the one suite piece the runtime installer does not install: Clarity downloads the LGPL build into `shared/ffmpeg` when the machine does not already have one. A repair shows a Qt dialog. `--selftest --json` prints the engine report (`ok`, `skipped`, `components`) and `{"type": "progress"}` lines for each step.

A driver older than the R600 series is refused by `assess_launch`, which calls `voxprint_common.gpu_probe.assess("Voxprint AI Clarity", refuse_below=(8, 9))`. The probe message does not include a download page, so Clarity adds <https://www.nvidia.com/Download/index.aspx>. `VOXPRINT_CI_LANCZOS` is still refused before the probe runs.

## Shared library

`voxprint-common` 0.1.1 "Menuchah" (import `voxprint_common`, API 1) supplies the shared suite pieces:

| Piece | Module |
|---|---|
| SHA-256 of files and byte strings | `voxprint_common.hashing` |
| `BUILD.json` | `voxprint_common.buildinfo`. A missing build stays null. The label says `dev`. |
| Licence gate and the permissive allow-list | `voxprint_common.licence` |
| Theme tokens and stylesheet | `voxprint_common.ui.theme` |
| Splash window | `voxprint_common.ui.splash` |
| `--json` progress and result lines | `voxprint_common.cli_protocol` |
| NVIDIA probe and VRAM tier | `voxprint_common.gpu_probe.assess(app, ...)` |
| Shared and models directories | `voxprint_common.paths` |
| GPU lock | `voxprint_common.gpu_lock.gpu_job(app, stage, ...)` |
| Thermal rule | `voxprint_common.thermal.decide` |
| Runtime manifest | `voxprint_common.runtime_manifest` |

The install is the v0.1.1 release wheel. The installer downloads the `.sha256` file published beside that wheel and refuses a digest that does not match `installer/runtime_lock.json`. pip does not need git, and Clarity does not clone the repository:

```text
voxprint-common @ https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.1.1/voxprint_common-0.1.1-py3-none-any.whl
```

Project files, the journal, and the catalogue stay in Clarity. The fetch gate still accepts MIT, Apache-2.0, and BSD after the shared check. ISC, CC-BY, and CC0 pass the shared release list and are not fetched here.

The GUI opens `voxprint_common.ui.splash.Splash` on `assets/splash/clarity.jpg`. The progress bar uses Clarity's identity colour `#2dd4bf` (teal, registered in the suite palette). It is not the `GREEN` success token. The rest of the window uses the shared yellow primary and soft blue accents. A missing art file paints `#0b1020`. The splash stays up at least 0.6 seconds, then fades for 200 ms.

## GPU lock, VRAM, and thermals

The live path (`live_control`) sets `device=True` and takes the lock with `hold_gpu`, which calls `gpu_job("clarity", stage, ...)`. The lock file records `app` and `stage`. An older file that says `owner` and `job` is read as those fields. A parent process with the same app id borrows the lock and does not delete it. The `restore_batch` message still sends `owner` `clarity`.

If the lock is already held, the lock reports a busy wait. The window shows that status and retries. `submit_restore` does the same for a worker line: a `GpuLockTimeout` comes back as an error line, the worker process stays up, and Clarity sends the request again. Cancel ends the retries.

Between chunks, while `device` is set:

- `measure` reads the VRAM budget, and `choose_tile` picks a tile when settings left the tile at 0.
- `read_shared_usage` and `sysmem_spill` watch shared GPU memory. A spill empties the CUDA cache, asks `note_spill` for a smaller batch, and may `halve_tile`.
- `thermal_decide` calls `voxprint_common.thermal.decide`. `wait_for_thermal` sleeps between chunks while that decision says to pause. Finished chunks stay on disk.

There is no CPU restore path. `assess_launch` refuses `VOXPRINT_CI_LANCZOS`. `enlarge.suite_runtime.ensure_runtime` treats that same flag as a skip so a test run does not download the suite. CI and the release workflow run `python -m clarity --selftest --no-gpu`, which sets the flag for that process only. The engine selftest asserts the CUDA and TensorRT providers. Clarity's selftest checks the window, the import, and the plan.

## `.vxclarity`

A project is a ZIP named `.vxclarity`. The layout matches the suite document package: `mimetype` is stored first, then `manifest.json`, `settings.json`, `resume.json`, and an optional `thumb.png`.

`format_version` is `"1.0"`. `created_by` and `generator` are `{app, version}`. `source` records the video's `format` and file name, and a SHA-256 when one was computed. `files` maps every other entry to `{sha256, size}`. Clarity-only data is under `x_clarity`: the product name, the modified time, and the video fingerprint (absolute path, relative path, size, and the head, tail, and optional full hashes).

Readers refuse more than 1000 entries, more than 512 MiB uncompressed, a JSON member over 4 MiB, an expansion above 100:1, or JSON nested deeper than 32. The full field list is in [formats/VXCLARITY.md](formats/VXCLARITY.md).

## Threads and processes

Clarity stays in one process.

The window builds on the Qt UI thread. Start runs `run_job` on a `QThread` (`_JobThread`). Progress and the finish signal cross back to the UI thread. Closing the window calls `CancelToken.cancel`, which sets a thread-safe flag and kills the ffmpeg processes that token is watching, then waits for the thread.

The CLI calls `run_job` on the process main thread. The same token cancels that run.

The GPU lock, the tiling call, and the CUDA network all run inside that process. A job does not fork a worker process of its own.
