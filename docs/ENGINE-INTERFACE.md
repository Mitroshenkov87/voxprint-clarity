# Engine interface

Clarity uses the shared package `voxprint_engine` (distribution name
`voxprint-engine`) from Voxprint AI Enlarge.

| | |
|---|---|
| Commit | `d11d609` (`d11d60921448b697d832b2699e6e3e04c62c6dfd`). TODO: tag `engine-v1.1.0` when it is published. |
| Requirement | `voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@d11d60921448b697d832b2699e6e3e04c62c6dfd` |
| CI install | the same `git+https` requirement. Enlarge is public, so there is no deploy key. |
| Import package | `voxprint_engine` |
| Video names | `voxprint_engine.api` |

The video API is in that commit. Clarity does not vendor a copy of the engine.
`patches/` holds two diffs against `d11d609` for the Enlarge repository:

- `patches/engine-d11d609-stream-frames.patch` adds `iter_tile_frames`.
  `tile_frames` keeps the accumulator, the coverage map, and the clip temporary
  for every frame at the model scale. 138 frames of 720p at 4x is about 22.7 GiB
  of float32. Clarity passes at most `host_batch_count` frames (512 MiB for those
  three buffers) so the installed engine stays inside RAM, downscales 4x to 2x
  on that batch, and stores uint8. The iterator is the upstream shape: one frame
  in the accumulator, and stacking the yields brings the whole chunk back.
- `patches/engine-d11d609-ort-trt-pin.patch` changes `install_gpu_wheels` and the
  GPU probe. The engine depends on `onnxruntime>=1.20,<2` (the CPU wheel) and
  then installs unpinned `tensorrt-cu13`, which is TensorRT 11 (`nvinfer_11`).
  ONNX Runtime 1.31's TensorRT EP loads `nvinfer_10`. Clarity's installers
  uninstall the CPU wheel after `ensure_runtime` and install
  `onnxruntime-gpu==1.31.0`, `tensorrt-cu13-libs==10.14.1.48`, and
  `tensorrt-cu13==10.14.1.48` with `--no-deps`. The 10.14 bindings have no
  cp314 wheel. The meta package is recorded so the current probe, which only
  checks that some `tensorrt-cu13` distribution exists, does not install
  TensorRT 11 on the next repair.

## What Clarity calls

| Need | Import |
|---|---|
| GPU lock | `api.gpu_job(..., owner="clarity")` |
| VRAM tile, spill, batch | `api.choose_tile`, `api.halve_tile`, `api.note_spill`, `api.sysmem_spill` |
| Shared GPU memory | `api.read_shared_usage` |
| Thermal rule | `api.decide` |
| Weights and users | `api.ensure_weight`, `api.register`, `api.unregister`, `api.models_dir` |
| Frames | `api.restore_frames`, `api.temporal_windows`, `api.tile_frames` |
| Restore worker | `api.RestoreBatchRequest` with `owner="clarity"`, then `api.WorkerClient` |
| Startup gate | `voxprint_engine.hardware.assess_gpu(refuse_below=(8, 9))`. An old driver message gains NVIDIA's download page in Clarity. |
| Shared runtime | `suite_runtime.ensure_runtime("clarity")` and `release_runtime("clarity")` |
| VRAM measurement | `voxprint_engine.vram.measure` |

`clarity/engine.py` is the only application module that imports
`voxprint_engine`. It adds no second implementation of those functions.

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

Tag `engine-v1.0.0` has `tools/export_onnx.py` (`convert`) in the enlarge
repository. That file is not part of the installed `voxprint_engine` wheel.
`python -m clarity.bench` calls `convert` when the module can be imported, and
otherwise writes the graph with `torch.onnx.export`.

## CI

CI installs the public tag over HTTPS on every test job and imports
`voxprint_engine` before the suite runs. `tests/test_engine.py` runs in that
job. There is no deploy key and no skip when a secret is absent.
