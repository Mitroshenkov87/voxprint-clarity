# Benchmark

`python -m clarity.bench` is the timing run for a rented RTX 4090
(or any RTX 40 series or newer GPU). `scripts/bench.sh` and `scripts/bench.ps1`
call it. `scripts/vm_test.sh --full` (and `scripts/vm_test.ps1 --full` on
Windows Server 2025) runs the lean plan and then this benchmark.
`scripts/test-gpu.ps1` still installs the stack, runs this benchmark, restores
a sample, and packs one archive. See [TESTING-GPU.md](TESTING-GPU.md).

The benchmark downloads the clip and the three weights, exports ONNX,
times both backends, and writes PNG crops. It exits 2 when the GPU, ffmpeg,
a weight, the export, or TensorRT EP is missing, and it does not invent a
frames-per-second number for a backend it did not run.

## Measured on an RTX 4090

Windows Server 2025 desktop VM, NVIDIA driver 610.88, the first GPU run of
this tree's restore path (Strong / SCUNet, 8-bit, 2x off):

| Run | Result |
|---|---|
| End to end, Strong (SCUNet), 192 frames, 77.4 s | 2.5 fps, peak VRAM 2736 MiB (2.7 GB), GPU util 96% |
| `python -m clarity.bench`, SCUNet `pytorch-fp16`, 8 frames of 256×256 | 17.241 fps, 138.8 MB |
| `python -m clarity.bench`, `realesr-general-x4v3` `pytorch-fp16` | 46.190 fps, 48.0 MB |

DRUNet was still being loaded in fp16 on that run, and `onnx` was not installed,
so the DRUNet timing and every TensorRT export were not measured. DRUNet is
fp32. The bench loads each model in the precision declared in the catalogue.

## Install

On the VM, with Python 3.14:

```bash
python -m pip install -r requirements.txt -r requirements-bench.txt
python -m clarity.bench
```

`requirements-bench.txt` pulls the CUDA 13.0 torch wheel, `spandrel`,
`onnx`, and `onnxruntime-gpu==1.31.0`. TensorRT EP has to be in that build.
Engine tag `engine-v1.4.0` is the shared library, installed as
`voxprint-engine[gpu]` with `--extra-index-url https://pypi.nvidia.com`.
The `gpu` extra installs the TensorRT libraries. `installer/pin_runtime.py`
uninstalls the CPU `onnxruntime` distribution only when that distribution is
present.
`--skip-download` reuses files already in `--clip-dir` and still exits 2
when one of them is absent.

## Clip

The script downloads Elephants Dream, file `ed_1024_512kb.mp4`, from
`https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4`.

- SHA-256: `fd6dc0e9da3a2f63487957325983f197f23c6ff1cd2b29bb71e64db567e0994c`
- Size: 47065346 bytes
- Licence: Creative Commons Attribution (Blender Foundation / Netherlands Media Art Institute)

ffmpeg reads one RGB frame. A center square of 256 pixels is the picture
used for the before/after crops. A hash mismatch deletes the download.
The script refuses to run when `nvidia-smi` is missing.

## Models

Three graphs, downloaded from the catalogue URLs and checked against the
pinned SHA-256:

- `drunet-deblocking-color` (DRUNet; the denoise weight is the same architecture and is not a fourth timed export)
- `scunet-color-real-psnr`
- `realesr-general-x4v3`

## Backends

Each model is loaded with spandrel on CUDA at its catalogue precision.
DRUNet is fp32. SCUNet and Real-ESRGAN are fp16. The timed loop is a
256×256 forward, after one warmup call. Peak VRAM is
`torch.cuda.max_memory_allocated`.

ONNX export calls `voxprint_engine.onnx_export.export_weight`. That writes a
checked fp16 graph. The example tensor is `1 x in_channels x H x W`. DRUNet
passes `in_channels` 4, so the plan is `1x4xHxW` and the noise-level map is
the fourth channel. The PyTorch timing still uses the catalogue precision, so
DRUNet stays fp32 and `trt_fp16_enable` stays off for that row. SCUNet and
Real-ESRGAN stay 3-channel fp16.
The ONNX Runtime session is opened with
`TensorrtExecutionProvider` (engine cache under `--out/trt-cache`) and then
`CUDAExecutionProvider`. `trt_fp16_enable` is on for an fp16 model and off
for fp32. There is no CPU provider. If TensorRT EP is not in
`session.get_providers()`, that model is not given a TensorRT timing.

## Output

`--out` (default `benchmark-out`) receives:

- `benchmark.md` and `benchmark.json` (fps, VRAM peak, frame count, notes)
- `<model-id>.onnx`
- `<model-id>-before.png` and `<model-id>-after.png` from the PyTorch forward

## Example

```bash
python -m pip install -r requirements.txt -r requirements-bench.txt
python -m clarity.bench --out ./benchmark-out
```
