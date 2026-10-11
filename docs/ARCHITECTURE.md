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
| `clarity.engine` | The only module that imports `voxprint_engine`. |
| `clarity.platform_gate` | Windows 11 24H2+ and the 2025 Linux releases. |
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

Clarity does not reimplement the GPU lock, the VRAM budget, the thermal rule, the model store, the hardware query, or the shared runtime. Those live in `voxprint_engine` at tag `engine-v1.2.1`. The package metadata is at the repository root, so the archive URL has no subdirectory.

```text
voxprint-engine @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.2.1.tar.gz
```

`clarity.engine` loads that package with `importlib` and forwards the calls. The requirement string is `ENGINE_REQUIREMENT`. The lock owner and the runtime holder are `OWNER` (`clarity`). API stays 1.

## Shared dependencies

The installer, every launch, and `selftest` call `ensure_runtime("clarity")` through `clarity.engine.ensure_shared_runtime`. Uninstall calls `release_runtime("clarity")`. The engine installs Python 3.14, the CUDA 13 torch, ONNX Runtime GPU, and TensorRT wheels. Clarity tells that call to skip the engine's default upscale weight, then drops its own holder on that component. When Clarity is the only holder, the weight file is deleted. Other holders keep the file. Clarity's own weights are registered under `clarity` in the shared models folder. Holders are a list in `shared/manifest.json`. A payload is removed when that list is empty.

The folder is `%LOCALAPPDATA%\Voxprint\shared` on Windows and `$XDG_DATA_HOME/voxprint/shared` or `~/.local/share/voxprint/shared` on Linux. `VOXPRINT_HOME` replaces the Voxprint directory.

Clarity's Clean and Strong weights go through the engine registry (`ensure_weight`, `register`, `unregister`). ffmpeg is the one suite piece the engine does not install: Clarity downloads the LGPL build into `shared/ffmpeg` when the machine does not already have one. A repair shows a Qt dialog. `--selftest --json` prints the engine report (`ok`, `skipped`, `components`) and `{"type": "progress"}` lines for each step.

A driver older than the R600 series is refused by `assess_launch`. The engine message does not include a download page, so Clarity adds <https://www.nvidia.com/Download/index.aspx>.

## GPU lock, VRAM, and thermals

The live path (`live_control`) sets `device=True` and takes the lock with `hold_gpu`, which calls `gpu_job(..., owner="clarity")`. The same owner is written on a `restore_batch` message. The engine shares a parent lock with a worker only when both sides pass that owner.

If the lock is already held, the engine reports a busy wait. The window shows that status and retries. `submit_restore` does the same for a worker line: a `GpuLockTimeout` comes back as an error line, the worker process stays up, and Clarity sends the request again. Cancel ends the retries.

Between chunks, while `device` is set:

- `measure` reads the VRAM budget, and `choose_tile` picks a tile when settings left the tile at 0.
- `read_shared_usage` and `sysmem_spill` watch shared GPU memory. A spill empties the CUDA cache, asks `note_spill` for a smaller batch, and may `halve_tile`.
- `thermal_decide` (`api.decide`) says whether to pause. `wait_for_thermal` sleeps between chunks while the engine says to pause. Finished chunks stay on disk.

There is no CPU restore path. `assess_launch` refuses `VOXPRINT_CI_LANCZOS`. The engine's `ensure_runtime` treats that same flag as a skip so a test run does not download the suite. CI and the release workflow run `python -m clarity --selftest --no-gpu`, which sets the flag for that process only. The engine selftest asserts the CUDA and TensorRT providers. Clarity's selftest checks the window, the import, and the plan.

## `.vxclarity`

A project is a ZIP named `.vxclarity`. The layout matches the suite document package: `mimetype` is stored first, then `manifest.json`, `settings.json`, `resume.json`, and an optional `thumb.png`.

`format_version` is `"1.0"`. `created_by` and `generator` are `{app, version}`. `source` records the video's `format` and file name, and a SHA-256 when one was computed. `files` maps every other entry to `{sha256, size}`. Clarity-only data is under `x_clarity`: the product name, the modified time, and the video fingerprint (absolute path, relative path, size, and the head, tail, and optional full hashes).

Readers refuse more than 1000 entries, more than 512 MiB uncompressed, a JSON member over 4 MiB, an expansion above 100:1, or JSON nested deeper than 32. The full field list is in [formats/VXCLARITY.md](formats/VXCLARITY.md).

## Threads and processes

Clarity stays in one process.

The window builds on the Qt UI thread. Start runs `run_job` on a `QThread` (`_JobThread`). Progress and the finish signal cross back to the UI thread. Closing the window calls `CancelToken.cancel`, which sets a thread-safe flag and kills the ffmpeg processes that token is watching, then waits for the thread.

The CLI calls `run_job` on the process main thread. The same token cancels that run.

The GPU lock, the tiling call, and the CUDA network all run inside that process. A job does not fork a worker process of its own.
