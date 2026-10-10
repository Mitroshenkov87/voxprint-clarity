# Benchmark

`python -m clarity.bench` is the one-command timing run for a rented RTX 4090
(or any RTX 40 series or newer GPU). `scripts/bench.sh` and `scripts/bench.ps1`
call it.

## Clip

The script downloads Elephants Dream, file `ed_1024_512kb.mp4`, from
`https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4`.

- SHA-256: `fd6dc0e9da3a2f63487957325983f197f23c6ff1cd2b29bb71e64db567e0994c`
- Size: 47065346 bytes
- Licence: Creative Commons Attribution (Blender Foundation / Netherlands Media Art Institute)

The file is not 1080p. The report records the resolution ffprobe returns. A
hash mismatch deletes the download. The script refuses to run when `nvidia-smi`
is missing (exit code 2) and does not invent timings.

## Models

DRUNet deblocking colour, DRUNet colour, SCUNet colour real_psnr, and compact
Real-ESRGAN general x4v3. Place the `.pth` files (the names in `clarity/models.py`)
in the directory you pass as `--clip-dir`. The script does not download weights.

## Backends

- PyTorch fp16, when `torch` and `spandrel` can load the weight on CUDA.
- ONNX Runtime TensorRT EP fp16 is recorded as a note until the matching
  `.onnx` is present and the engine session can be opened. That path does not
  yet time a full TensorRT loop, and it does not yet write before/after frames.

The report is `benchmark.md` and `benchmark.json` (fps, VRAM peak, frame count).

## Example

```bash
python -m pip install -r requirements-gpu.txt
python -m clarity.bench --clip-dir ./benchmark-out
```
