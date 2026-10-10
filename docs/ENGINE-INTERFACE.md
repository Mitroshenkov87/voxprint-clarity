# Engine interface

Clarity uses the shared package `voxprint_engine` (distribution name
`voxprint-engine`) from Voxprint AI Enlarge.

| | |
|---|---|
| Tag | `engine-v1.0.0` (`26b7807`) |
| Requirement | `voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0` |
| CI install | `git+ssh://git@github.com/Mitroshenkov87/voxprint-enlarge.git@engine-v1.0.0` with `VOXPRINT_ENGINE_DEPLOY_KEY` |
| Import package | `voxprint_engine` |
| Video names | `voxprint_engine.api` |

The video API is in the tag. Clarity does not vendor a copy and does not
carry `docs/engine-patches`.

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
| Startup gate | `voxprint_engine.hardware.assess_gpu(refuse_below=(8, 9))` |
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

`VOXPRINT_ENGINE_DEPLOY_KEY` is a read-only deploy key. The workflow starts
`webfactory/ssh-agent@v0.9.0` and installs the tag over SSH. When the secret is
absent, the install step is skipped and the pure tests still run. Tests in
`tests/test_engine.py` run only when the package imports.
