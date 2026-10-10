# Engine interface

Clarity uses the shared package `voxprint_engine` (distribution name
`voxprint-engine`) from Voxprint AI Enlarge. It does not fork that package
and it does not reimplement the GPU lock, the VRAM budget, the thermal rule,
the model store, the hardware query, or the ONNX Runtime session.

## Pin

| | |
|---|---|
| Repository | https://github.com/Mitroshenkov87/voxprint-enlarge |
| Commit Clarity was written against | `6d00d44` |
| Planned tag | `engine-v1.0.0` (not published yet) |
| Import package | `voxprint_engine` |
| API | `voxprint_engine.API == 1` (the video additions keep API 1) |
| CI secret | `VOXPRINT_ENGINE_TOKEN` (read access to the private repo) |

`pyproject.toml` pins
`voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge.git@6d00d44`.
TODO: switch that pin to `@engine-v1.0.0` when the tag and a buildable engine
pyproject exist. At `6d00d44` the project name in that repo is still
`voxprint-enlarge` and the pyproject has no build-system table, so
`pip install` of the pin can fail even when the token is present. CI prints a
notice and continues with the pure Clarity tests.

When `VOXPRINT_ENGINE_TOKEN` is unset, CI skips the engine install and says so.
Jobs that need a live engine are skipped. Unit tests that stub the engine keep
running.

This workspace could not clone Enlarge with the local git credential (the
contents API returned 404). The files at `6d00d44` were read through GitHub
and the patch below was applied, in a temporary tree, with `git apply --check`.

## What Clarity calls

All of these go through `clarity/engine_shim.py`.

| Need | Engine symbol | Notes |
|---|---|---|
| Driver 600+ and compute capability | `hardware.assess_gpu` | Clarity passes `refuse_below=(8, 9)`. Omit that argument and Enlarge's warning-only behaviour stays. |
| No CPU stand-in | — | Clarity refuses to start when `VOXPRINT_CI_LANCZOS=1`. It does not honour that flag. |
| GPU lock | `gpu_lock.gpu_job` | Clarity passes `owner="clarity"`. The default owner remains `enlarge`. |
| VRAM budget | `vram.measure`, `choose_tile`, `note_spill`, `halve_tile` | Budget is `max(0, free_now - max(2 GB, 8% of total))`, re-measured before each batch. |
| System-memory spill | `vram.sysmem_spill` | True when Windows shared GPU memory grew by more than 256 MB. `None` means no spill. |
| Thermal | `thermal.decide` | Full speed for 2.5 h, then pause between chunks while the 5-minute median is at least 83 C or throttling exceeds 60 s. Resume at or below 75 C. |
| Weights | `store.ensure_weight`, `register`, `unregister` | Shared folder. Clarity's refcount key is `clarity`. |
| Session | `session.open_session`, `session.run_batch` | TensorRT EP, then CUDA EP, fp16 when the input type contains `float16`. Cache under the engine trt-cache directory. |
| Worker | op `restore_batch` | One worker process under the GPU lock. |

`formats.py` and `heic_system.py` are not used.

## Video additions (API stays 1)

These are not in Enlarge at `6d00d44`. They are additive:

- `hardware.assess_gpu(..., refuse_below=None)`.
- `gpu_lock.try_acquire`, `release`, and `gpu_job` take `owner=None` (default `enlarge`).
- `session.run_batch(session, nchw)` delegates to `voxprint_engine.video.run_batch`.
- `vram.sysmem_spill(shared_now, shared_baseline, threshold=256*1024*1024)`.
- New module `voxprint_engine.video`: `temporal_windows`, `tile_frames`, `run_batch`, `restore_frames`, `sysmem_spill`, `RestoreBatchRequest`.
- New `voxprint_engine.worker.restore.handle_restore`.
- `worker.server.handle` accepts `op == "restore_batch"`.
- `worker.protocol` re-exports `RestoreBatchRequest`.

Frame-wise models use `length=1` and `overlap=0`. A longer window keeps the
left overlap as context and emits each source frame once. Tiling is per frame,
with the same feather as `tile.py` at `6d00d44`, and the tiles of one step are
batched on N.

## Shim

`clarity/engine_shim.py` is the only import of `voxprint_engine`. If
`voxprint_engine.video` does not yet export the video names, the shim loads
`docs/engine-patches/files/voxprint_engine/video.py`. Delete that fallback
when the same names exist upstream.

`gpu_job(owner=)` falls back to the unpatched signature on `TypeError`.
`assess_gpu(refuse_below=)` does the same, and a non-empty warning from the
old function becomes a refusal. That is how Clarity stays strict on an engine
that only warns for compute capability below 8.9.

## Applying the patch

From a checkout of voxprint-enlarge at `6d00d44`:

```bash
git apply --check /path/to/clarity/docs/engine-patches/0001-clarity-video.patch
git apply /path/to/clarity/docs/engine-patches/0001-clarity-video.patch
```

The patch was produced from the blobs at that commit and checked with
`git apply`. See [engine-patches/README.md](engine-patches/README.md).
