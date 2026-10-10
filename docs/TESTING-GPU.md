# GPU test on a rented RTX 4090

`scripts/test-gpu.ps1` is the one script for a fresh Windows Server 2025 VM
with an RTX 4090 and an NVIDIA driver from the 600 series or newer. It
installs what the run needs, times the models, restores a sample of the test
clip, and packs one archive.

CI does not run this script. The GitHub job uses the same engine URL with
`VOXPRINT_ENGINE_DEPLOY_KEY` and skips that install when the secret is absent.

## What it installs

- uv, then CPython 3.14, in `gpu-results\venv`
- this checkout (`uv pip install -e .`) so `python -m clarity.bench` imports Clarity
- `torch==2.14.1` from the CUDA 13.0 index
- `requirements.txt` and `requirements-bench.txt` (`spandrel`, `onnxruntime-gpu`)
- the `tensorrt` package
- an LGPL ffmpeg and ffprobe from
  `ffmpeg-master-latest-win64-lgpl.zip` (BtbN). A build that enables GPL,
  libx264, or libx265 is rejected. `av1_nvenc` must be present.
- MinGit, when `git` is not already on `PATH`
- `voxprint-engine` from
  `git+ssh://git@github.com/Mitroshenkov87/voxprint-enlarge.git@engine-v1.0.0`

The engine line is private. Pass the read-only deploy key for voxprint-enlarge
in either of these ways:

- `-DeployKeyPath C:\keys\enlarge`
- `VOXPRINT_ENGINE_DEPLOY_KEY` set to that path, or to the private key text

The script restricts the key file to the current user, runs `ssh-keyscan` for
`github.com`, and points `GIT_SSH_COMMAND` at the key. It does not print the
key. OpenSSH Client has to be installed (`ssh` and `ssh-keyscan`).

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
$env:VOXPRINT_ENGINE_DEPLOY_KEY = "C:\keys\enlarge"
powershell -ExecutionPolicy Bypass -File .\scripts\test-gpu.ps1
```

Pass `-AllowOtherOs` when the image caption is not exactly Windows Server 2025,
and `-AllowOtherGpu` when the card is another RTX 40 series GPU.
