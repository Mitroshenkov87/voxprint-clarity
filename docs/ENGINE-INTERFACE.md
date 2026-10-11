# Engine interface

Clarity uses the shared package `voxprint_engine` (distribution name
`voxprint-engine`).

| | |
|---|---|
| Tag | `engine-v1.4.0` |
| Requirement | `voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.4.0.tar.gz` |
| Layout | `pyproject.toml` is at the repository root, so the archive URL has no `#subdirectory` fragment. pip does not need git. |
| CI install | the same archive URL, with `--extra-index-url https://pypi.nvidia.com`. The repository is public, so there is no deploy key. |
| Import package | `voxprint_engine` |
| Video names | `voxprint_engine.api` |

The video API is in that tag. Clarity does not vendor a copy of the engine.

`iter_tile_frames` yields one restored frame. The accumulator stays one frame.
Clarity still limits the source window with `host_batch_count` (512 MiB for the
three full-resolution float32 buffers). A 720p frame at 4x is restored alone.
Stacking the yields would allocate the chunk again.

The `gpu` extra pins `onnxruntime-gpu==1.31.0` and
`tensorrt-cu13-libs==10.14.1.48`. The TensorRT libraries are on the NVIDIA
package index, so the install passes `--extra-index-url https://pypi.nvidia.com`.
The engine's base dependencies do not include ONNX Runtime. A CPU
`onnxruntime` distribution shadows the GPU wheel, and the engine selftest
fails when that distribution is present. Clarity's `pin_runtime.py` uninstalls
it only when it is installed, then fails if CUDA or TensorRT providers are
still missing. Torch and spandrel come from `requirements-gpu.txt` on every
install, including a venv that already existed.
The engine export uses a static batch of 1. Height and width are the dynamic
axes. `run_onnx_batch` runs a larger tile batch as slices of that declared
size and concatenates the rows. A GPU `--selftest` refuses a venv that cannot
import torch and spandrel. `--selftest --no-gpu` does not. The CUDA and
TensorRT provider check stays in the engine selftest and in
`scripts/vm_provider_check.py`.

## What Clarity calls

| Need | Import |
|---|---|
| GPU lock | `voxprint_common.gpu_lock.gpu_job("clarity", stage, ...)` |
| VRAM tile, spill, batch | `api.choose_tile`, `api.halve_tile`, `api.note_spill`, `api.sysmem_spill` |
| Shared GPU memory | `api.read_shared_usage` |
| Thermal rule | `voxprint_common.thermal.decide` |
| Weights and users | `api.ensure_weight`, `api.register`, `api.unregister` |
| Models directory | `voxprint_common.paths.models_dir` |
| Frames | `api.iter_tile_frames`, `api.temporal_windows` |
| ONNX | `api.ensure_onnx` (checked fp16 file, cached by sha256) |
| Session | `voxprint_engine.session.open_session`, TensorRT then CUDA |
| Restore worker | `api.RestoreBatchRequest` with `owner="clarity"`, then `api.WorkerClient` |
| Startup gate | `voxprint_common.gpu_probe.assess("Voxprint AI Clarity", refuse_below=(8, 9))`. An old driver message gains NVIDIA's download page in Clarity. |
| Shared runtime | `enlarge.suite_runtime.ensure_runtime("clarity")` and `release_runtime("clarity")`. The call writes the app id and does not install Enlarge's default model. |
| VRAM measurement | `voxprint_engine.vram.measure` |

`clarity/engine.py` is the only application module that imports
`voxprint_engine`. It adds no second implementation of those functions.
`clarity.bench` imports `voxprint_engine.onnx_export.export_weight` for the
TensorRT timing.

## Lock owner

The lock file records `app` and `stage`. A parent process with the same app
id borrows the lock and does not delete it. An older file that says `owner`
and `job` is read as those fields. Clarity's parent uses app id `clarity`.
The `restore_batch` message still sends `owner` `clarity`. An `enlarge` lock
is not reused.

## GPU lock timeout

`restore_batch` catches `GpuLockTimeout` and returns one JSON line,
`{"op":"error","ok":false,"message":"the GPU is busy: ..."}`. The worker
process keeps running. Clarity shows that message and sends the same request
again. It does not kill the worker for that line. Cancel ends the retries.

## ONNX export

`api.ensure_onnx` converts a pinned `.pth` to a checked fp16 ONNX file and
caches it as `<sha256>.onnx`. A second call returns that file.
`python -m clarity.bench` calls `voxprint_engine.onnx_export.export_weight`
for the same conversion.

The example tensor is `1 x in_channels x H x W`. `in_channels` is 1 to 16
and defaults to 3. DRUNet's manifest sets it to 4, so the plan is `1x4xHxW`
and the noise-level map is the fourth channel. Clean, Strong, and the upscale
nets all open a TensorRT session and retry on CUDA when the engine build
fails. DRUNet's catalogue precision stays fp32: the engine runs the module in
float, then writes the fp16 graph, and the bench leaves `trt_fp16_enable` off
for that model.

Before TensorRT loads, Clarity reports `building_engine` or
`loading_cached_engine`. A `--json` progress line adds `"stage"`. The window
status shows the same sentence. The first build can take several minutes.

## CI

CI installs the public tag over HTTPS on every test job, then copies
`enlarge` from that archive into site-packages, and imports `voxprint_engine`
and `enlarge.suite_runtime` before the suite runs. `tests/test_engine.py`
runs in that job. There is no deploy key and no skip when a secret is absent.
