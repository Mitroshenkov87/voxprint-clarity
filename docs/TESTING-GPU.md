# GPU test on a rented RTX 4090

A smoke run has to pass before an RC tag. Linux VMs run `scripts/vm_test.sh`.
Windows Server 2025 VMs run `scripts/vm_test.ps1`. Both call `scripts/vm_test.py`.
The flags, the JSON objects, and the exit codes are the same.

CI does not have a GPU. It checks that `--smoke` exits 3, and it runs shellcheck
and PSScriptAnalyzer. It does not install the GPU stack.

```bash
scripts/vm_test.sh --smoke
scripts/vm_test.sh --full --json
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\vm_test.ps1 --smoke
powershell -ExecutionPolicy Bypass -File .\scripts\vm_test.ps1 --full --json
```

`--allow-other-gpu` accepts another RTX 40 series card. `--allow-other-os`
accepts a Windows caption other than Windows Server 2025. Nothing prompts.

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | Every step passed. |
| 1 | A test failed. The message names the step. |
| 2 | Setup failed (install, ffmpeg, arguments, the Windows caption). |
| 3 | No usable GPU. `nvidia-smi` is missing, the driver is older than the 600 series, compute capability is below 8.9, or the card is not a 4090. |

A failed step prints `FAILED: step <name>: <reason>` and stops. With `--json`,
stdout is one JSON object per line. The last line is the summary. The same
summary is always written to `summary.json`.

```json
{"type":"step","name":"restore-clean","status":"pass","seconds":12.4}
{"type":"summary","app":"clarity","mode":"smoke","tag":"main-b01cd6c","ok":false,"exit":1,"step":"restore-clean","error":"...","results":"...","steps":[]}
```

`status` is `pass`, `fail`, or `skip`. `tag` is the git tag when HEAD is
exactly that tag, otherwise `main-` plus `git rev-parse --short=7 HEAD`
(for example `main-b01cd6c`). It is not taken from `BUILD.json`. The `info`
step still compares version and build with `BUILD.json`. The driver always
writes `summary.json`. On Windows, a failed GPU gate prints that summary and
exits 3 before Python is installed.

## Resume

A finished step writes `~/.done/<tag>--<step>`. The next run skips that step.
`gpu` always runs. `install` always runs its check too: the wheel install is
skipped when the install marker exists, and that marker is written only after
`CUDAExecutionProvider` and `TensorrtExecutionProvider` have loaded. Delete
one marker to repeat that step. A new tag does not reuse an older tag's markers.

## Results

The directory is `--results`, or `/workspace/vm-results/clarity/<tag>` when
that parent already exists, or `vm-results/clarity/<tag>` inside the checkout.
Sync that tree to `/workspace/vm-results/clarity/<tag>/`. It contains:

- `summary.json` and `summary.md`
- `logs/` — one log per step, plus `*-peaks.json` for the memory sample
- `crops/` — one before/after frame per restore
- `clips/` and `outputs/` — the short sources and the restored MKV files
- `bench/` — present after `--full`

## Smoke

About 10 minutes once the TensorRT engines are cached. The first engine build
can take several minutes, and each smoke restore is allowed 20 minutes before
that step fails.

1. `gpu` — driver 600+, compute capability 8.9+, RTX 4090 unless `--allow-other-gpu`.
2. `install` — uv, CPython 3.14, this checkout, `torch==2.14.1` from the CUDA 13.0 index, `requirements.txt`, `requirements-bench.txt`, and `voxprint-engine` at tag `engine-v1.2.1` (the GitHub archive, so git is not required). After the engine, `installer/pin_runtime.py` removes the CPU `onnxruntime` wheel, installs `onnxruntime-gpu==1.31.0`, `tensorrt-cu13-libs==10.14.1.48`, and `tensorrt-cu13==10.14.1.48` (`--no-deps`) from the runtime lock, and installs `torch==2.14.1` from `requirements-gpu.txt` plus spandrel. The child process checks the providers before the done marker is written. The step's seconds include that wheel install, not only the re-exec check.
3. `selftest` — `python -m clarity --selftest` with no `--no-gpu`. A missing torch or spandrel fails this step. Then `scripts/vm_provider_check.py`. `CUDAExecutionProvider` and `TensorrtExecutionProvider` both have to load.
4. `info` — `python -m clarity info --json` matches `BUILD.json`.
5. `clip` — 15 seconds of the pinned Elephants Dream file, stream-copied.
6. `restore-clean` — Clean (DRUNet) with 2x off. Progress must name `building_engine` or `loading_cached_engine`.
7. `restore-2x` — the same clip, Clean with 2x on, with the same TensorRT stage. RAM and VRAM are sampled while it runs.
8. `bounds` — peak process-tree RSS at most 8192 MiB, and peak `nvidia-smi` memory used at most 16384 MiB. The 720p 4x path used to allocate about 22.7 GiB for one chunk.
9. `forced-failure` — `restore` of a missing file prints `{"type":"result","ok":false,...}` and exits 3.

`VOXPRINT_CI_LANCZOS` is removed for the run, so the provider check is not skipped.
The script does not pass a multi-line `python -c` string to a native command.

## Full

The lean plan from the first RTX 4090 session, then the benchmark. Case A was
a blocky 8 s H.264 clip restored with Clean. Case B was 8 s darkened
(gamma 0.6, brightness -0.25) with temporal noise, Strong (SCUNet), 2x off.
Case C was 10 s padded to 1280x720, Strong, 2x on. That case is where the
whole chunk was buffered at 4x in float32.

The VM builds those three clips from the pinned Elephants Dream file
(`ed_1024_512kb.mp4`, SHA-256
`fd6dc0e9da3a2f63487957325983f197f23c6ff1cd2b29bb71e64db567e0994c`).
The source is already a low-bitrate H.264 web encode, so case A is a stream
copy and does not use libx264. Cases B and C are re-encoded with the built-in
`mpeg4` encoder. The published case B stills in the user guide are the Tears
of Steel frame from that first run (Blender Foundation, CC BY 3.0).

`bounds` reads the case C sample and uses the same 8192 / 16384 MiB limits.
`bench` runs `python -m clarity.bench` and copies `benchmark.md`,
`benchmark.json`, and the crops.

## What the install rejects

ffmpeg is the BtbN LGPL build (`ffmpeg-master-latest-win64-lgpl.zip` on
Windows, `ffmpeg-master-latest-linux64-lgpl.tar.xz` on Linux). A build whose
configuration contains `--enable-gpl`, `--enable-libx264`, or
`--enable-libx265` is rejected. `--disable-libx264` is the LGPL build and is
kept. `av1_nvenc` must be present. The engine archive does not need git.
The engine line is

`voxprint-engine @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.2.1.tar.gz`

## `scripts/test-gpu.ps1`

This remains the one-archive Windows run: install, `python -m clarity.bench`,
one Clean restore of the first `-SampleFrames` frames (default 48), and
`gpu-results\clarity-gpu-test.tar.gz`. It uses the same helpers as
`scripts/vm_test.ps1` (`scripts/gpu-common.ps1`): the engine URL, the LGPL
check, uv, and the GPU gate. Its own failures still exit 1.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test-gpu.ps1
```

Pass `-AllowOtherOs` when the image caption is not exactly Windows Server 2025,
and `-AllowOtherGpu` when the card is another RTX 40 series GPU.
A second run reuses uv, the virtualenv, ffmpeg, and the clip cache.
