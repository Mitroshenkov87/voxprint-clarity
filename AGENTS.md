# Agents

Clarity is the non-generative video app in the Voxprint AI Media Suite.
English in the README, docs, comments, and commits. Imperative commit subjects.

- Depend on `voxprint_engine` from voxprint-enlarge at tag `engine-v1.1.0`
  (commit `0f6a840`). API stays 1.
  Import the video names from `voxprint_engine.api`. Do not reimplement the
  GPU lock, VRAM budget, thermal rule, model store, hardware query, or ONNX session.
- The parent GPU lock and the restore worker use the owner string `clarity`.
  A `GpuLockTimeout` from `restore_batch` is an error line: wait, tell the
  user, and retry. Do not kill the worker for that line.
- Permissive weights only (MIT, Apache-2.0, BSD). No GPL, no x264, no x265,
  no H.264 output.
- RTX 40 series (compute capability 8.9) and a 600-series driver are required.
  Do not honour `VOXPRINT_CI_LANCZOS`.
- Do not commit weights, ONNX files, or TensorRT engines.
