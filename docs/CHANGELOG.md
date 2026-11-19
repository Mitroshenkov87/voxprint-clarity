# Changelog

## Unreleased

- Build 1005 is an RC again. `BUILD.json` version is `1.0.0-rc` while the
  first final release waits for the shared UI. The label is
  `1.0.0 RC build N "…"`. `version --json` reports `1.0.0-rc`.
  Tag `v1.0.0-rc.1005` matches this file. A plain `v1.0.0` tag matches a
  later file whose version is `1.0.0`.
- The installer session probe and the worker share `clarity/native_gap.py`.
  A failed session names which of `cudart64_13`, `cublas64_13`,
  `cublasLt64_13`, `cudnn*`, and `nvinfer_10` are actually missing, and
  appends the ONNX Runtime log line when one was captured.
- The setup log receives the pin script's stdout and stderr after
  `pinning torch and spandrel into the shared venv`.
- The identity probe model is ONNX IR 10 and opset 13. ONNX Runtime 1.31
  rejects an IR version above 13, and the onnx helper default is higher.
- Build 1005. On Windows, register `site-packages/torch/lib` and every directory
  under `site-packages/nvidia` and `site-packages/tensorrt_libs` that contains
  a CUDA 13 or TensorRT library, including a nested `bin/x86_64`. A session
  that stays on `CPUExecutionProvider` lists which of `cudart64_13`,
  `cublas64_13`, `cublasLt64_13`, `cudnn*`, and `nvinfer_10` is missing.
  The engine helper still runs when it is installed.
- The installer publisher is **VOXPrint AI Suite**.
- Build 1004. On Windows, register the CUDA 13 and TensorRT library directories
  before ONNX Runtime is imported. A session that stays on
  `CPUExecutionProvider` fails, and the message names `nvinfer_10.dll`.
  The installer and a GPU selftest open a real TensorRT session and a real
  CUDA session. `--json` flushes phase and frame-batch progress while a job
  is running. Subprocess text is read as UTF-8.
- Install Windows program files in `C:\Program Files\VOXPrint AI Suite\Clarity`.
  The Start menu folder is **VOXPrint AI Suite**. The window is
  `voxprint-clarity.exe` and the command line is `voxprint-clarity-cli.exe`.
  Linux installs in `/opt/voxprint-ai-suite/clarity`. An upgrade removes the
  old install directory and shortcuts and leaves `%LOCALAPPDATA%\Voxprint` in place.
- Register Clarity in `HKCU\Software\Voxprint\Clarity` and the Windows App Paths
  key so other suite apps can find it. The Linux desktop file is
  `ai.voxprint.Clarity.desktop`.
- Spell the product **VOXPrint** in window titles, the splash, the installer,
  and the docs. Package names, the CLI, and the data directory stay `voxprint`.
- Pin `voxprint-engine[gpu]` at tag `engine-v1.4.0`. API stays 1.
- Pin `voxprint-common` 0.1.1 "Menuchah" from the release wheel. The installer
  checks the wheel against the published `.sha256` file.
- Import paths, the GPU lock, the thermal rule, and the GPU probe from
  `voxprint_common`. Call `enlarge.suite_runtime.ensure_runtime("clarity")`.
  That call writes the app id and does not install Enlarge's default model.
- Pass the display name `VOXPrint AI Clarity` to the GPU probe.
- Keep a missing build number as null. The version label says `dev`.
  When `GITHUB_RUN_NUMBER` is set, the build is that number plus `offset`.
