# Benchmark

`python -m clarity.bench` is the one-command timing run for a rented RTX 4090
(or any RTX 40 series or newer GPU). `scripts/bench.sh` and `scripts/bench.ps1`
call it. The script downloads the clip and the three weights, exports ONNX,
times both backends, and writes PNG crops. It exits 2 when the GPU, ffmpeg,
a weight, the export, or TensorRT EP is missing, and it does not invent a
frames-per-second number for a backend it did not run.

## Install

On the VM, with Python 3.14:

```bash
python -m pip install -r requirements.txt -r requirements-bench.txt
python -m clarity.bench
```

`requirements-bench.txt` pulls the CUDA 13.0 torch wheel, `spandrel`, and
`onnxruntime-gpu`. TensorRT EP has to be in that ONNX Runtime build.
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

Each model is loaded with spandrel in fp16 on CUDA. The timed loop is a
256×256 forward, after one warmup call. Peak VRAM is
`torch.cuda.max_memory_allocated`.

ONNX export tries `voxprint_engine.export_onnx`, then
`voxprint_engine.session.export_onnx`. Enlarge pin `6d00d44` has neither, so
the script writes the graph with `torch.onnx.export` (fp16, opset 17) and
says so in the report. The ONNX Runtime session is opened with
`TensorrtExecutionProvider` (`trt_fp16_enable`, engine cache under
`--out/trt-cache`) and then `CUDAExecutionProvider`. There is no CPU
provider. If TensorRT EP is not in `session.get_providers()`, that model is
not given a TensorRT timing.

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
