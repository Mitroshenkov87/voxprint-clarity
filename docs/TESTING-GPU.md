# GPU test on a rented RTX 4090

`scripts/test-gpu.ps1` is the one script for a fresh Windows Server 2025 VM
with an RTX 4090 and an NVIDIA driver from the 600 series or newer. It
installs what the run needs, times the models, restores a sample of the test
clip, and packs one archive.

CI does not run this script. The GitHub jobs install the same public engine
URL over HTTPS. Enlarge is public, so the script does not take a deploy key.

## What it installs

- uv, then CPython 3.14, in `gpu-results\venv`
- this checkout (`uv pip install -e .`) so `python -m clarity.bench` imports Clarity
- `torch==2.14.1` from the CUDA 13.0 index
- `requirements.txt` and `requirements-bench.txt` (`spandrel`, `onnxruntime-gpu==1.31.0`, `onnx`)
- an LGPL ffmpeg and ffprobe from
  `ffmpeg-master-latest-win64-lgpl.zip` (BtbN). A build whose configuration
  contains `--enable-gpl`, `--enable-libx264`, or `--enable-libx265` is rejected.
  `--disable-libx264` is the LGPL build and is kept. `av1_nvenc` must be present.
- MinGit, when `git` is not already on `PATH` (the engine install is a git URL)
- `voxprint-engine` from
  `git+https://github.com/Mitroshenkov87/voxprint-enlarge@9183489`

The engine tag installs `onnxruntime-gpu==1.31.0` and TensorRT 10.14.1.48
and removes the CPU wheel. Its selftest asserts `CUDAExecutionProvider` and
`TensorrtExecutionProvider`. The GPU script does not repeat that check.
A multi-line `python -c` string is not used: Windows PowerShell 5.1 strips
quotes from native arguments.

## What it runs

1. `python -m clarity.bench`, which downloads Elephants Dream and the three
   weights into `gpu-results\cache` and writes `benchmark.md`, `benchmark.json`,
   and the before/after crops.
2. One end-to-end restore of that clip through `run_job` (NVDEC, scene chunks,
   the clean models, AV1). `-SampleFrames` defaults to 48 and is the length of
   `sample.mkv`. `-SampleFrames 0` restores the whole clip.

A second run reuses uv, the virtualenv, ffmpeg, git, and the clip cache.
The restore work folder is removed at the start of that step so the sample is
produced again. A failed step stops the script with `FAILED:` and the reason.
It does not write the archive after a failure.

## Output

The archive is `gpu-results\clarity-gpu-test.tar.gz`. Inside
`clarity-gpu-test/`:

- `transcript.log` — the full PowerShell transcript
- `benchmark.md` and `benchmark.json`
- `<model>-before.png` and `<model>-after.png`
- `sample.mkv` — the restored sample
- `sample-source.mp4` — the cut passed to the restore, when `-SampleFrames` is not 0

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test-gpu.ps1
```

Pass `-AllowOtherOs` when the image caption is not exactly Windows Server 2025,
and `-AllowOtherGpu` when the card is another RTX 40 series GPU.
