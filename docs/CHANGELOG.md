# Changelog

## Unreleased

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
