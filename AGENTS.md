# Agents

Clarity is the non-generative video app in the Voxprint AI Media Suite.
English in the README, docs, comments, and commits. Imperative commit subjects.

- Depend on `voxprint_engine` from voxprint-enlarge. Pin is commit `6d00d44`
  until tag `engine-v1.0.0`. Do not reimplement the GPU lock, VRAM budget,
  thermal rule, model store, hardware query, or ONNX session.
- Video-only additions belong in `docs/engine-patches` and, until they land
  upstream, behind `clarity/engine_shim.py`.
- Permissive weights only (MIT, Apache-2.0, BSD). No GPL, no x264, no x265,
  no H.264 output.
- RTX 40 series (compute capability 8.9) and a 600-series driver are required.
  Do not honour `VOXPRINT_CI_LANCZOS`.
- Do not commit weights, ONNX files, or TensorRT engines.
