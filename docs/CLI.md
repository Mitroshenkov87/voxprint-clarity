# Command line

`python -m clarity` and `voxprint-clarity` are the same program. With no
subcommand they open the window, after the operating-system and GPU gates.
A positional `.vxclarity` path opens that project in the window.

Headless commands are for other suite apps. They share the restore plan and
`run_job` with the Start button.

The flag list is generated from those argparse parsers when the documentation
site is built, so the reference cannot drift from the program. See the
[command reference](reference/cli.md). That page is written at build time and
is not stored in the repository. The sections below are the narrative.

```text
python -m clarity restore --input CLIP --output OUT --mode clean --format mkv --dry-run --json
python -m clarity project open clip.vxclarity --json
python -m clarity project resume clip.vxclarity --dry-run --json
python -m clarity version --json
python -m clarity models list --json
python -m clarity models fetch --dry-run --json
python -m clarity selftest --json
```

`--json` may also be the first argument (`python -m clarity --json version`).

## Exit codes

| Code | Name | When |
|---|---|---|
| 0 | ok | The command finished. A dry-run that printed a plan is 0. |
| 1 | usage | Bad arguments, a missing file, a project this build cannot read, an HDR source, or a cancel. |
| 2 | environment | The OS or GPU gate refused a live run, the engine is not installed, or ffmpeg/ffprobe failed to start. |
| 3 | job | A live restore or a live model fetch failed after it started. |

With `--json`, a failure still prints one result line on stdout:
`{"type":"result","ok":false,"error":"..."}`. The same message is on stderr.
The line is written by `voxprint_common.cli_protocol`, so it also carries
`code` (the exit code from the table above) and `message` (the same text as
`error` on a failure). The exit codes themselves are unchanged.
Progress, when a live job reports it, is `{"type":"progress","fraction":0.25,"message":"12/48"}`
on stdout. A TensorRT build or cache load adds `"stage":"building_engine"` or
`"stage":"loading_cached_engine"`. The message says the first build can take
several minutes. Other logs stay on stderr.

## restore

| Flag | Meaning |
|---|---|
| `--input` | Source video. Required. |
| `--output` | Destination video. Required. |
| `--mode` | `clean` (default) or `strong`. |
| `--upscale` | Optional 2x. Off unless this flag is present. |
| `--format` | `mkv` (default), `mp4`, `av1`, or `hevc`. `mkv` and `mp4` set the container and leave the codec at AV1. `av1` and `hevc` set the codec and leave the container at MKV. |
| `--codec` | Optional `av1` or `hevc` when the container and the codec both need to be set. |
| `--ffmpeg`, `--ffprobe` | Binaries. Default to the names on `PATH`. |
| `--dry-run` | Validate the file, run ffprobe and a CPU scene pass, and print the full plan. |
| `--json` | Machine-readable stdout. |

A live restore (no `--dry-run`) uses the GPU gate and then `run_job`. There is
no CPU mode.

### Dry-run

`--dry-run` does not lock the GPU, does not load weights, and does not spawn
the NVDEC or NVENC commands. It does read the file with ffprobe and with
ffmpeg's CPU scene filter, which is the same filter the job uses to choose
chunk boundaries. The printed plan then contains:

- the probe command and the scene command
- one decode command and one encode command per chunk (`-hwaccel cuda`, NVENC or the SVT-AV1 fallback when that encoder is installed)
- the concat command that joins the chunks and copies audio
- the model ids for the mode
- `executed: false` and `gpu_used: false`
- `encoder_available`: false when this ffmpeg has neither the NVENC encoder nor `libsvtav1` / `hevc_nvenc`. The plan still names `av1_nvenc` or `hevc_nvenc`, which is the encoder a GPU machine would use. Those commands are not run.

A matching `plan.json` and journal already on disk are read so a resume plan
can mark finished chunks. Dry-run does not write them.

## project

`project open PATH` reads a `.vxclarity` file and prints the source,
destination, mode, and how many journal chunks are recorded. It does not
launch the window and it does not render frames. `--dry-run` is the same read.

`project resume PATH` continues that project through `run_job`, using the
work folder recorded in the project. `--dry-run` prints the restore plan for
that source and those settings, including chunks the journal already finished.

Opening the window on a project is still `python -m clarity PATH.vxclarity`.

## info and version

Both print the label from `BUILD.json`: version, build, and codename.
`--version` on the window launcher prints the same label and does not need
`--json`. `--dry-run` does not change the text.

## models

`models list` prints the catalogue (id, name, role, licence, filename,
sha256, download URL). Nothing is downloaded.

`models fetch` downloads the catalogue, or the ids passed with repeated
`--id`, through the shared engine store and checks the hash. `--dry-run`
lists those files and does not download them. A live fetch needs the engine
package. It does not need a GPU to hash a file the engine stores, but the
engine import itself is required.

The Start button and **Download models** call the same catalogue and the same
`fetch_weights` helper.

## selftest

`selftest` calls `ensure_runtime("clarity")`, repairs ffmpeg when it is
missing, builds the default Clean restore plan, then opens the window
offscreen and checks that Start has a label. A GPU selftest also requires
torch and spandrel. `--no-gpu` does not. The CUDA and TensorRT provider
check belongs to the engine selftest.

`--no-gpu` is the check for a machine with no GPU. It still imports the
engine, builds the plan (CUDA decode, `av1_nvenc`, `yuv420p` after `-c:v`),
and opens the window. For that process it tells the engine to skip the suite
download, then clears the skip. It does not make a restore run on CPU.

`--json` prints each step as a `{"type": "progress"}` line and includes the
engine report (`ok`, `skipped`, `components`) and the plan on the result.
`--dry-run` lists the checks and does not repair or open a window.

`python -m clarity --selftest` is the check a user runs. CI and the release
workflow both run `python -m clarity --selftest --no-gpu`.

## install-suite

`python -m clarity --install-suite` calls `ensure_shared_runtime`, repairs
ffmpeg, and fetches every catalogue weight Clarity uses. It does not download
the engine's default upscale weight. A missing engine package says
`Voxprint inference engine missing, run repair.`
The Windows setup and `voxprint-clarity-install` run it. Uninstall calls
`release_runtime("clarity")` from `--unregister-models-user`.

## What the window still does

Save, Save as, and Save a copy write a `.vxclarity` file. Export chooses the
output video path. Those actions use the suite icons in `assets/icons/suite/`.
The headless `restore` command is the Start button without the window.
