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
- `requirements.txt` and `requirements-bench.txt` (`spandrel`, `onnxruntime-gpu`)
- the `tensorrt` package
- an LGPL ffmpeg and ffprobe from
  `ffmpeg-master-latest-win64-lgpl.zip` (BtbN). A build that enables GPL,
  libx264, or libx265 is rejected. `av1_nvenc` must be present.
- MinGit, when `git` is not already on `PATH` (the engine install is a git URL)
- `voxprint-engine` from
  `git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0`

After the engine install, the script installs `onnxruntime-gpu` again so the
engine's CPU wheel does not replace it, then stops if
`TensorrtExecutionProvider` is missing.

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
