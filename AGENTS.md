# Agents

Clarity is the non-generative video app in the VOXPrint AI Media Suite.
English in the README, docs, comments, and commits. Imperative commit subjects.

- Depend on `voxprint_engine` at tag `engine-v1.4.0`. The requirement is
  `voxprint-engine[gpu]` from the GitHub archive URL. `pyproject.toml` is at
  the repository root, so there is no `#subdirectory` fragment and pip does
  not need git. Install that extra with
  `--extra-index-url https://pypi.nvidia.com`. The GPU pins live in the extra.
  Import the video names from `voxprint_engine.api`. Do not reimplement the
  VRAM budget, model store, or ONNX session. Paths, the GPU lock, the thermal
  rule, the GPU probe, and the licence allow-list live in `voxprint_common`.
  `ensure_runtime` lives in `enlarge.suite_runtime` from the same tag. The
  wheel does not ship that package, so setup copies it from the archive.
  Pass app id `clarity`. GPU checks use the display name `VOXPrint AI Clarity`.
  Do not install Enlarge's default model. `ensure_default_model` does that
  only for app id `enlarge`.
- Depend on `voxprint-common` 0.1.1 "Menuchah" (API 1) from the v0.1.1
  release wheel, never a git clone. Verify the published `.sha256` where the
  installer downloads it. Hashing, BUILD.json, the licence gate, the theme,
  the splash, the `--json` line, paths, the GPU lock, thermals, the runtime
  manifest, and the GPU probe come from that package. A missing build stays
  null and the label says `dev`.
- The parent GPU lock and the restore worker use the owner string `clarity`.
  A `GpuLockTimeout` from `restore_batch` is an error line: wait, tell the
  user, and retry. Do not kill the worker for that line.
- Permissive weights only (MIT, Apache-2.0, BSD). No GPL, no x264, no x265,
  no H.264 output.
- RTX 40 series (compute capability 8.9) and a 600-series driver are required.
  Do not honour `VOXPRINT_CI_LANCZOS`.
- Do not commit weights, ONNX files, or TensorRT engines.
