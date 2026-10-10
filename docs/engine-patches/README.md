# Engine patches

`0001-clarity-video.patch` is an additive diff against voxprint-enlarge
commit `6d00d44`. API stays 1. Enlarge's image path is unchanged when the new
arguments are left at their defaults.

Apply it from the Enlarge repository root:

```bash
git checkout 6d00d44
git apply --check docs/engine-patches/0001-clarity-video.patch
git apply docs/engine-patches/0001-clarity-video.patch
```

Use the copy of the patch from the Clarity repository. This directory is not
inside the Enlarge tree.

## Files

| Path | Change |
|---|---|
| `voxprint_engine/hardware.py` | `assess_gpu(..., refuse_below=None)` |
| `voxprint_engine/gpu_lock.py` | `owner=` on acquire, release, and `gpu_job` |
| `voxprint_engine/session.py` | `run_batch` delegates to `video.run_batch` |
| `voxprint_engine/vram.py` | `sysmem_spill` delegates to `video.sysmem_spill` |
| `voxprint_engine/video.py` | new: windows, tiling, NCHW batch, spill reader |
| `voxprint_engine/worker/restore.py` | new: `restore_batch` handler |
| `voxprint_engine/worker/protocol.py` | documents `restore_batch`, re-exports the request |
| `voxprint_engine/worker/server.py` | dispatches `restore_batch` |

`docs/engine-patches/files/` is the same new Python the patch adds, so Clarity
can load `video.py` until Enlarge ships it. Do not edit one without the other.
Regenerate the patch from a `6d00d44` tree plus those files; do not hand-edit
the diff.
