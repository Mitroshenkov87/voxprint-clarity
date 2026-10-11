# Engine interface

Clarity uses the shared package `voxprint_engine` (distribution name
`voxprint-engine`) from Voxprint AI Enlarge.

| | |
|---|---|
| Tag | `engine-v1.1.0` (commit `0f6a84013a72413b1fb53b4b0ae9826c10c54436`). API stays 1. |
| Requirement | `voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.1.0` |
| CI install | the same `git+https` requirement. Enlarge is public, so there is no deploy key. |
| Import package | `voxprint_engine` |
| Video names | `voxprint_engine.api` |

The video API is in that tag. Clarity does not vendor a copy of the engine.

`iter_tile_frames` yields one restored frame. The accumulator stays one frame.
Clarity still limits the source window with `host_batch_count` (512 MiB for the
three full-resolution float32 buffers). A 720p frame at 4x is restored alone.
Stacking the yields would allocate the chunk again.

`install_gpu_wheels` pins `onnxruntime-gpu==1.31.0` and
`tensorrt-cu13-libs==10.14.1.48` and removes the CPU `onnxruntime` wheel.
Clarity's installers leave that pin to the engine. The engine selftest asserts
`CUDAExecutionProvider` and `TensorrtExecutionProvider`. Clarity's `--selftest`
checks the window, the import, and the default plan.

## What Clarity calls

| Need | Import |
|---|---|
| GPU lock | `api.gpu_job(..., owner="clarity")` |
| VRAM tile, spill, batch | `api.choose_tile`, `api.halve_tile`, `api.note_spill`, `api.sysmem_spill` |
| Shared GPU memory | `api.read_shared_usage` |
| Thermal rule | `api.decide` |
| Weights and users | `api.ensure_weight`, `api.register`, `api.unregister`, `api.models_dir` |
| Frames | `api.iter_tile_frames`, `api.temporal_windows` |
| ONNX | `api.ensure_onnx` (checked fp16 file, cached by sha256) |
| Session | `voxprint_engine.session.open_session`, TensorRT then CUDA |
| Restore worker | `api.RestoreBatchRequest` with `owner="clarity"`, then `api.WorkerClient` |
| Startup gate | `voxprint_engine.hardware.assess_gpu(refuse_below=(8, 9))`. An old driver message gains NVIDIA's download page in Clarity. |
| Shared runtime | `suite_runtime.ensure_runtime("clarity")` and `release_runtime("clarity")` |
| VRAM measurement | `voxprint_engine.vram.measure` |

`clarity/engine.py` is the only application module that imports
`voxprint_engine`. It adds no second implementation of those functions.
`clarity.bench` imports `voxprint_engine.onnx_export.export_weight` for the
TensorRT timing.

## Lock owner

The engine shares a parent lock with a worker subprocess only when both sides
pass the same `owner`. Clarity's parent and its `restore_batch` message both
use `clarity`. An `enlarge` lock is not reused.

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

The converter's example tensor is 3 channels. SCUNet and Real-ESRGAN take
that path, then Clarity opens a session with TensorRT and retries on CUDA
when the engine build fails. DRUNet takes a condition channel (4 channels)
and stays on PyTorch. The restore path does not call `torch.onnx.export`.

Before TensorRT loads, Clarity reports `building_engine` or
`loading_cached_engine`. A `--json` progress line adds `"stage"`. The window
status shows the same sentence. The first build can take several minutes.

## CI

CI installs the public tag over HTTPS on every test job and imports
`voxprint_engine` before the suite runs. `tests/test_engine.py` runs in that
job. There is no deploy key and no skip when a secret is absent.
