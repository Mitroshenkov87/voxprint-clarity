# VOXPrint AI Clarity: Phase 1 research (non-generative video restoration)

Updated: 2026-10-10. Status: research only. No repo, builds or installs until the Audiobook Builder and Movie Dubber RC builds (999) are tested.

Goal: 1080p in -> 1080p out (optional 2x), denoise, compression-artifact removal, sharpening, colour cleanup. **No generative repainting.** Target hardware: RTX 4090 Laptop (16 GB) and RTX 5090 Laptop (24 GB). Platforms: Windows 11 24H2+ and Linux 2025+ only.

**Product decisions (2026-10-10): see [DECISIONS.md](DECISIONS.md).** Summary: lightweight near-real-time "quality leveller"; no heavy/generative models; no frame interpolation; upscale optional and off by default (Compact only); MVP SDR-only, HDR10 later; GAN "Enhanced" undecided, leaning no. Where this file differs, DECISIONS.md wins.

Evidence tags: **[V]** verified by me (repo LICENSE file / GitHub or HF API / file size), **[S]** number reported by the cited source (measured by them, not by us), **[E]** my estimate, **[I]** inferred, **[TODO]** not found yet. Nothing here is legal advice.

Faithfulness classes used below:
- **F (faithful)**: trained with L1/PSNR losses. Removes noise and blocks, does not invent texture. Default for Clarity.
- **P (perceptual/GAN)**: GAN loss. Sharper, but synthesises plausible fine texture (mild hallucination). Allowed only as an opt-in "Enhanced" strength, blended with F output.
- **G (generative)**: diffusion/transformer prior. Can repaint content. Reference only, never default.

## 1. Candidate models

Licences: code from each repo's LICENSE (GitHub API `license` + file read where it says NOASSERTION), weights from the HF model card or, if there is no separate weights licence, the repo licence [I]. Sizes are the real download sizes [V]. VRAM at 1080p is TensorRT fp16, 1 stream, from vs-mlrt where available [S], otherwise [E].

| Model | Task / scale | Code licence | Weights licence | Weights size | VRAM at 1080p in | Temporal stability | Class | Verdict |
|---|---|---|---|---|---|---|---|---|
| Real-ESRGAN x4plus (RRDB) | real-world SR x4 | BSD-3-Clause [V] | no separate licence -> BSD-3 [I] | 63 MB [V] | 4-8 GB with tiling [E] | frame-wise: flicker on fine texture [I] | P | too slow at 1080p input (8K output); no |
| Real-ESRGAN x2plus (RRDB) | SR x2 | BSD-3 [V] | BSD-3 [I] | 63 MB [V] | 3-6 GB [E] | frame-wise | P | candidate for "2x Enhanced", slow |
| RealESRNet x4plus | SR x4, no GAN | BSD-3 [V] | BSD-3 [I] | 63 MB [V] | as x4plus | frame-wise, smoother | F | faithful twin of x4plus; slow |
| realesr-general-x4v3 (Compact, SRVGGNet) + `wdn` variant | SR x4 + denoise strength (blend of the two) | BSD-3 [V] | BSD-3 [I] | 4 MB each [V] | ~1.1 GB (v2/v3 class) [S vs-mlrt] | frame-wise; light model flickers less than RRDB [I] | P | **MVP "2x" path** (run, then downscale 4x->2x) |
| realesr-animevideov3 (Compact) | anime SR x4 (also x2/x3 via downscale) | BSD-3 [V] | BSD-3 [I] | 2 MB [V] | ~1.1 GB [S] | designed for video, frame-wise [S README] | P | anime preset only |
| Real-ESRGAN "1x" variants | 1x cleanup | not official: community 1x models (OpenModelDB) | **per model, many CC-BY-NC-SA** [I] | 0.1-64 MB | small | frame-wise | P mostly | only via an allow-list with verified licences |
| RealBasicVSR x4 | real-world VSR x4 (cleaning module + recurrent) | Apache-2.0 [V] | no separate licence; Dropbox/GDrive hosted [V README] | ~25 MB (6.3 M params) [S paper] | high; grows with sequence; OOM on 8 GB at 1080p [S issue #49] | **good (recurrent, long-range)** | P | optional "Temporal" tier on 540p->2160p; old mmedit/mmcv 0.x stack, needs port |
| BasicVSR++ c64n7 (REDS/Vimeo) | VSR x4 | Apache-2.0 [V] (also in mmagic, Apache-2.0 [V]) | Apache-2.0 [I] | 29 MB [V] | high, grows with sequence; needs chunking [S mmagic #1214] | **very good (bidirectional)** | F | optional slow tier only (not default) |
| BasicVSR++ c128n25 NTIRE21 decompress track1/2/3 | **1x compressed-video enhancement** | Apache-2.0 [V] | Apache-2.0 [I]; trained on LDV (dataset terms [TODO]) | 168 MB each [V] | very high at 1080p; tile [E] | very good | track1 F, track2 P [I] | best-fit task, but heavy (<1 fps est.); benchmark candidate |
| SCUNet color real_psnr / real_gan | blind denoise 1x | Apache-2.0 [V] | Apache-2.0 [I] (KAIR release) | 68 MB [V] | 3.6 GB [S] | frame-wise | psnr = F, gan = P | **MVP "Strong clean"** (slow) |
| DPIR DRUNet color / **deblocking_color** | non-blind denoise 1x / JPEG-style deblock (noise-level or QF map input) | MIT [V] (DPIR and KAIR) | MIT [I] | 124 MB each [V] | 1.8 GB [S] | frame-wise, very stable when strength is fixed [I] | F | **MVP default "Clean"** |
| FBCNN | JPEG artifact removal, predicts QF | Apache-2.0 [V] | Apache-2.0 [I] | 274 MB [V] | ~2-3 GB [E] | frame-wise | F | alternative to DRUNet-deblock; test |
| NAFNet (SIDD/GoPro) | denoise / deblur 1x | MIT [V] | MIT [I] | 30-270 MB [E] | 2-4 GB [E] | frame-wise | F | trained on camera noise, not codec noise; test |
| SPAN / Compact community 2x | fast SR x2 | Apache-2.0 [V] (SPAN) | per model | 1-5 MB | 3-6 GB with batch 2, 3 streams [S VSGAN] | frame-wise | P | speed reference; licence per model |
| RVRT / VRT | VSR, denoise, deblur | **CC-BY-NC-4.0** [V LICENSE file] | NC [I] | 50-200 MB | very high | very good | F | **excluded (non-commercial)** |
| RealViformer (ECCV 2024) | real-world VSR | MIT [V] | [TODO: weights hosting/licence] | [TODO] | [TODO] | good (flow-based, artifact-propagation aware) [S paper] | P | watch list |
| DDColor (paper/artistic/modelscope, tiny) | **colourise B/W only** | Apache-2.0 [V] | Apache-2.0 [V HF card] | 870 MB; tiny 210 MB [V] | 2-4 GB [E] | frame-wise: colour flicker unless smoothed [I] | **G-like** (colours are guessed) | opt-in "Colourise" mode only, never on colour input |
| SeedVR / SeedVR2 (3B, 7B) | one-step diffusion VSR | Apache-2.0 [V] | Apache-2.0 [V HF] | 3B: 12.6 GB + VAE 0.9 GB [V] | ">=18 GB even for 3B"; batch 5 needed for temporal consistency [S numz] | ok only with batch >= 5 | **G** | reference only; ~0.2-0.3 fps on desktop 4090 [S] |
| FlashVSR v1.1 (CVPR 2026) | one-step diffusion streaming VSR (Wan2.1 base) | Apache-2.0 [V] | Apache-2.0 [V HF] | ~6.6 GB total [V] | 12-24 GB with tiling [S ComfyUI node] | good (streaming) | **G** | reference only; sparse-attention kernel untested on RTX 40/50 [V README] |
| UltraVSR (2025), DLoRAL (2025), STAR, Upscale-A-Video, VEnhancer | diffusion VSR | Apache-2.0 / MIT / **none** / **S-Lab NC** / **none** [V] | varies | GB-class | 16-80 GB | varies | **G** | excluded (generative; some NC or no licence) |
| NVIDIA RTX Video SDK 1.1 (VSR + artifact reduction) | SR x2-x4 + deblock, real time | proprietary NVIDIA licence | DLLs not redistributable standalone; commercial release needs notice to NVIDIA; must not be made subject to an open-source licence [S licence PDF, RTXVideoProcessor README] | n/a | small [E] | good, built for video [S] | F/P (closed) | **Windows only** (Win10+, no Linux) [S NVIDIA]; optional "Fast" engine only if licence review passes |
| NVIDIA Maxine VFX SDK (artifact reduction, SR, denoise) | 1x/2x | samples MIT [V], SDK proprietary | proprietary | n/a | small | good | F/P | same concerns; watch list |

Tooling licences [V]: spandrel MIT (loads Compact, ESRGAN, SCUNet, DRUNet, NAFNet, SPAN; NC arches live in separate `spandrel_extra_arches`), ONNX Runtime MIT, NVIDIA TensorRT OSS Apache-2.0 (runtime libraries under NVIDIA SLA [I]), Torch-TensorRT BSD-3, PySceneDetect BSD-3, VapourSynth LGPL-2.1, **vs-mlrt GPL-3.0**, **video2x AGPL-3.0**, OpenModelDB GPL-3.0 (site code). Suite apps are Apache-2.0 [V], so GPL/AGPL code is ideas-only, never bundled.

## 2. Realistic fps at 1080p

Sources (all measured on desktop GPUs; no laptop numbers found):
- vs-mlrt RTX 4090 wiki, 1920x1080 RGBS, TRT fp16, CUDA graphs: https://github-wiki-see.page/m/AmusementClub/vs-mlrt/wiki/NVIDIA-GeForce-RTX-4090 (mirror of github.com/AmusementClub/vs-mlrt/wiki)
- vs-mlrt v14 release (RTX 3090): https://github.com/AmusementClub/vs-mlrt/releases/tag/v14
- VSGAN-tensorrt-docker README benchmarks (4090 Linux, 5090 Win11): https://github.com/styler00dollar/VSGAN-tensorrt-docker
- RealBasicVSR paper Table 2 (V100, 180x320 -> 720x1280): https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.pdf
- SeedVR2 ComfyUI benchmarks (4090 desktop): https://github.com/comfyorg/comfyui_seedvr2
- FlashVSR README (A100): https://github.com/OpenImagingLab/FlashVSR

Laptop scaling [E]: RTX 4090 Laptop (AD103, 9728 cores, <=175 W) and RTX 5090 Laptop (GB203, 10496 cores, <=175 W) have about 45-60 % of desktop 4090 throughput (FP32/tensor TFLOPS and memory bandwidth ratio), lower when heat-soaked. Rule of thumb: laptop fps = desktop 4090 fps x 0.5; 5090 Laptop ~10-20 % faster than 4090 Laptop.

| Model (1080p input) | Desktop 4090, TRT fp16 [S] | Desktop, PyTorch / ORT-CUDA | 4090 Laptop TRT fp16 [E] | 5090 Laptop TRT fp16 [E] | 2 h film (172,800 frames) on 4090 Laptop [E] |
|---|---|---|---|---|---|
| Compact (realesr v2/v3 class) | 17.1 (1 stream) / 33.4 (2) / 38-41 (3) [S vs-mlrt]; Compact 2x 61.5 batch 2, 3 streams [S VSGAN] | ORT-CUDA 14.2 vs TRT 28.5 (other test, same wiki) [S] -> TRT ~2x | 15-30 | 18-35 | 1.6-3.2 h |
| SPAN 2x | 51.3 (4090 Linux); **5090 desktop Win11 64.3** [S VSGAN] | n/a | 20-28 | 25-35 | ~2 h |
| DPIR DRUNet color | 18.3 (1) / 25.1 (2) [S] | ORT-CUDA 8.7 vs TRT 21.9 [S] | 9-13 | 11-15 | 3.7-5.3 h |
| SCUNet color | 5.1-5.5 [S] | n/a | 2.5-3 | 3-3.5 | 16-19 h |
| Real-ESRGAN x2plus (RRDB) | not found | V100: 149 ms/frame at 180x320 in [S paper] | 2-4 [E] | 2.5-5 [E] | 12-24 h |
| RealBasicVSR / BasicVSR++ c64n7, 540p in -> 2160p | not found (TRT export of deformable conv is hard [I]) | V100: 63 / 77 ms/frame at 180x320 in [S paper] | PyTorch fp16: 4-8 [E] | 5-10 [E] | 6-12 h |
| BasicVSR++ NTIRE c128n25, 1080p 1x | not found | not found | PyTorch fp16: 0.3-1 [E] | 0.4-1.2 [E] | 2-7 days: impractical |
| SeedVR2 3B fp8, 480p -> 1080p, batch 5 | 0.19 fps (PyTorch) [S numz] | - | <0.1 (16 GB needs offload) [E] | 0.1-0.15 [E] | weeks |
| FlashVSR | 17 fps at 768x1408 on A100 [S paper] | - | 1-4 [E] | 2-5 [E] | 12-48 h |
| RTX Video SDK VSR/AR | "real time" [S NVIDIA forum] | - | >=30 [E] | >=30 [E] | <2 h |

Notes:
- TRT vs PyTorch: TRT fp16 + CUDA graphs is ~2-2.5x ONNX Runtime CUDA for CNNs in the same benchmark [S]. 2 streams give most of the gain; 3 streams helps only small models [S].
- TRT engines are built per GPU + TRT version (minutes, then cached). TensorRT-RTX builds faster with comparable speed [S vs-mlrt README].
- Pipeline overhead (decode, colour conversion, encode) is not in these numbers; NVDEC/NVENC run on separate engines, so expect 5-15 % loss [E].

## 3. Pipeline

| Stage | Design | Notes |
|---|---|---|
| Probe | `ffprobe -show_streams -show_format -show_chapters` -> JSON: codec, pix_fmt, bit depth, `color_primaries/trc/space/range`, fps (CFR/VFR), HDR side data | refuse/warn on Dolby Vision (RPU dropped), interlaced (deinterlace first: `bwdif`) |
| Decode | ffmpeg `-hwaccel cuda` (NVDEC) -> `-f rawvideo -pix_fmt rgb48le` (10-bit) or `rgb24` (8-bit) over a pipe; or `yuv420p10le` and convert on GPU in torch | colour matrix explicit (BT.709 SDR, BT.2020 for UHD); never rely on defaults |
| Scene chunking | PySceneDetect (BSD-3, `AdaptiveDetector`) or ffmpeg `scdet`; chunk = scene, split long scenes at <= 1000 frames | recurrent/temporal models reset state at cuts (no ghosting across cuts) |
| Temporal window | sliding window N frames with overlap; recurrent models: chunk + 8-16 frame pre/post padding, padding outputs discarded | frame-wise models need no overlap; temporal smoothing of per-frame strength (sigma/QF) across a scene to avoid pumping |
| Tiling | tiles 512-960 px, 16-32 px overlap, feathered blend; tile size chosen from the VRAM budget (section 4) | TRT static shapes per tile size; one engine per (model, tile, batch) |
| Inference | fp16, batch from VRAM budget; F model first, optional P model blended (`alpha` = "Enhanced" strength) | sharpening: classical CAS (ffmpeg `cas`) or unsharp after the model, user slider |
| Colour cleanup | deterministic, not AI: auto levels/white balance per scene (grey-world / percentile), optional 3D LUT; DDColor only in explicit "Colourise B/W" mode | keep strength per scene constant to prevent flicker |
| Encode | Default NVENC `av1_nvenc` (optional `hevc_nvenc` only if present; no H.264 output, see DECISIONS.md #6) (both GPUs have AV1 NVENC [I]), `-preset p6 -tune hq -rc vbr -cq 18-20`, 10-bit `p010` output when source >= 8-bit to avoid banding; closed GOP, IDR at chunk start | LGPL ffmpeg build (BtbN LGPL, as Audiobook Builder `ffmpeg-n8.1.3-win64-lgpl-8.1` [V]): no libx264/x265 (GPL); NVENC/NVDEC are fine in LGPL builds [I]; CPU fallback SVT-AV1 [TODO: confirm in LGPL build] |
| HDR passthrough | MVP: SDR only (decision 4); HDR10 deferred to a later version. HDR10: process in PQ domain only experimentally (models are SDR-trained, out of distribution); copy `color_*` tags and mastering display / CLL side data | [TODO] verify FFmpeg 8 `hevc_nvenc` writes HDR10 SEI from side data; else use `-bsf:v hevc_metadata` |
| Mux | per-chunk files -> concat demuxer `-c copy`, then `-map 0:v` (new) + `-map 0:a? -map 0:s? -map 0:t? -map_chapters 0 -map_metadata 0 -c copy` from the source | audio/subtitles/attachments untouched; VFR: keep source timestamps (`-copyts` / mkv timecodes) [TODO test] |
| Resume | job folder `<project>/clarity/<job>/`: `plan.json` (chunks, settings hash, model hashes), `chunks/NNNNN.mkv` written to `.partial` then `os.replace`, `journal.jsonl` (chunk done, frames, sha256) | on restart: verify last chunk frame count with ffprobe, redo only incomplete chunks; settings change -> new job |
| Preview | 10 s A/B split-screen preview before a full run | user sees result in ~1 min |

## 4. Reuse of suite infra

Sources: Movie Dubber reply (2026-10-10), voxprint/FOR-MOVIE-DUBBER.md, movie-dubber/FROM-AUDIOBOOK-BUILDER.md, movie-dubber/HARDWARE-PROFILE.md, Movie Dubber repo main `6edf6b5`, Audiobook Builder main `a3b96e5`.

| Item | Existing implementation | Clarity plan |
|---|---|---|
| Shared models folder | `%LOCALAPPDATA%\Voxprint\models`; overrides `VOXPRINT_HOME`, `VOXPRINT_MODELS_DIR`, then `state\models_dir.txt` (Linux `$XDG_DATA_HOME/voxprint`). Per-model folder, SHA-256 + size manifest, `.verified` marker, per-model `.<model>.lock` (`msvcrt.locking`/`fcntl.flock`), `.partial` download dir, 180 s stall watchdog. Code: dubber `infra/shared_paths.py`, `infra/model_store.py` [V] | same spec; models under `models\clarity-<name>\` |
| `.users.json` refcount | `{"audiobook-builder":true,"movie-dubber":true}`; `--register-models-user` / `--unregister-models-user`; delete only at 0 other users and user consent (default keep). Code: AB `infra/models_users.py`, `Voxprint.iss` (`c1369c6`) [V] | add key `"clarity"`; same uninstaller logic |
| GPU lock | `%TEMP%\voxprint-gpu.lock`, JSON `{owner, pid, job, started, eta}`, atomic create, re-entrant, held for the whole job; stale = pid dead or `eta` passed > 2 h. Code: dubber `infra/gpu_lock.py` (`gpu_job`) [V Movie Dubber + code]. AB: not in app yet [V] | one outer lock for the whole job (preview too); update `eta` per chunk |
| VRAM budget | `budget = max(0, free_now - max(2 GB, 0.08 * total))`, `free_now` from `torch.cuda.mem_get_info`, re-measured per batch group; optional cap `VOXPRINT_VRAM_FRACTION` (default off). Code: dubber `infra/resources.py` (`vram_allowance_gb`) [V]. AB `core/vram_policy.py` still older formula at `a3b96e5` [V] | same formula; re-measure before each batch group; tile + batch from budget |
| Sysmem spill guard | Monitor `\GPU Process Memory(pid_<pid>*)\Shared Usage`; growth > ~256 MB over stage baseline -> `empty_cache` + halve batch. **Not in main `6edf6b5`** (grep: no "Shared Usage" in `dubber/`); no `rc`/`999` branch on the remote as of 2026-10-10 07:4x MSK, so it is local/unpushed RC work [V] | reuse once merged; Clarity halves batch, then tile size |
| Thermal | Automatic, no modes: full speed first ~2.5 h, then cooling pauses between blocks while the 5-min GPU median >= 83 C or throttling > 60 s; resume at <= 75 C; never on a timer. Suite rule per project-notes 1f9fcd4 (stricter of earlier Dubber 87 C draft and section 3). Telemetry/throttle decoding exists: dubber `diag/procs.py` [V] | same rule, pause between chunks; no separate modes |
| Installer / runtime | Inno Setup online installer built by GitHub Actions; `install-runtime.ps1` uses uv to create a CPython 3.11 venv in shared `%LOCALAPPDATA%\Voxprint\runtime-<key>`; key moving to SHA-256(lock) + flavor. torch 2.11.0+cu128 [V Movie Dubber] | join the shared runtime if pins match; overlay onnxruntime-gpu / tensorrt wheels, spandrel, scenedetect |
| TensorRT | **none in the suite yet** [V Movie Dubber] | new Clarity decision: engine cache per GPU + TRT version, outside models (proposal `%LOCALAPPDATA%\Voxprint\trt-cache\<gpu>-<driver>-<trt>\`, rebuildable, not refcounted) |
| ffmpeg | BtbN LGPL n8.1.3 (AB) [V]; suite uses stream copy only, **no NVDEC/NVENC yet** [V Movie Dubber] | Clarity is the first NVDEC/NVENC user: verify LGPL build has `hevc_nvenc`/`av1_nvenc`/cuvid on both laptops |
| Start menu | Inno `DefaultGroupName=Voxprint` in both `.iss` [V] | folder **VOXPrint AI Suite**, shortcut **VOXPrint AI Clarity** |
| Build numbers | `BUILD.json` offset + run number, a Hebrew word from the Torah or Tanakh as the codename [V] | same scheme |

## 5. Recommended MVP stack

| Layer | Choice | Why |
|---|---|---|
| App shell | PySide6 (same as suite), worker process per job | suite consistency; VRAM fully freed on worker exit |
| Inference | PyTorch (suite runtime) + spandrel loader; **TensorRT fp16 via ONNX** for frame-wise models (ORT TensorRT EP or TRT/TensorRT-RTX directly), PyTorch fallback | TRT ~2x faster [S]; fallback when engine build fails |
| Default "Clean" (F) | DRUNet deblocking_color (codec artifacts, QF map) + DRUNet color denoise (sigma estimated per scene) | MIT, faithful, ~9-13 fps laptop TRT [E] |
| Optional slow tier "Strong clean" (F), not default | SCUNet color real_psnr | Apache-2.0, blind, but only ~2.5-3 fps [E]: misses the near-real-time goal; keep only if users want it for short clips |
| Sharpen | CAS / unsharp (classical, slider) | no hallucination, ~free |
| Colour | classical auto levels/WB per scene; DDColor only in "Colourise B/W" | deterministic |
| Upscale (optional, **off by default**) | realesr-general-x4v3 + wdn blend (strength slider), downscale to target (e.g. 720p->1080p, 2x); anime preset animevideov3 | 4 MB, BSD-3, 15-30 fps laptop [E]; lightweight, per decision 3 |
| "Enhanced" (P) | **undecided, leaning no** (decision 5); if ever added: blend of F output with Compact, alpha slider | GAN texture synthesis conflicts with the lightweight/faithful goal |
| Optional slow tier "Temporal" (later, not default) | BasicVSR++ c64n7 / RealBasicVSR, chunked; BasicVSR++ NTIRE track1 only as an offline quality reference | best temporal stability but 4-8 fps at best [E]; needs mmcv-free port; may be dropped under decision 1 |
| Excluded | RVRT/VRT (NC), Upscale-A-Video (NC), STAR/VEnhancer (no licence), SeedVR2/FlashVSR/UltraVSR/DLoRAL (generative, decision 1), frame interpolation e.g. RIFE (decision 2), vs-mlrt/video2x code (GPL/AGPL) | licence, "no repaint" or product rule |

### Benchmark plan (on the ROG 4090 Laptop, later the 5090 Laptop; after the 999 RC tests)

1. Clips (1-2 min each, 1080p, owned/free sources, e.g. Blender open movies CC-BY or Xiph test media): (a) low-bitrate H.264 web re-encode (blocking), (b) noisy low-light camera, (c) clean master re-encoded at CRF 35 (has ground truth), (d) anime, (e) old B/W (colourise check).
2. Runs per clip: DRUNet clean (default), Compact upscale and Compact+wdn (optional), SCUNet psnr and BasicVSR++ (optional slow tiers, to decide keep/drop); each in PyTorch fp16 and TRT fp16; tile/batch from the VRAM rule.
3. Measure: end-to-end fps and model-only fps; peak `memory_reserved`, free-memory trace, Shared Usage counter (must stay flat); GPU temp/power/clocks/throttle every 2 s; engine build time.
4. Quality: clip (c) full-reference PSNR/SSIM/**VMAF** vs the clean master; all clips no-reference NIQE/MUSIQ (pyiqa, check licences) as a sanity signal only.
5. Temporal: **tOF** (optical-flow difference between output and input motion) and **warping error / tLPIPS** (RAFT warp of frame t-1 to t, masked by occlusion); plus a simple flicker index = std of per-frame mean luminance difference in static regions. Lower = steadier; compare against input.
6. Visual: 10 s split-screen A/B MP4s for Aleksandr; side-by-side crops of faces/text (hallucination check: no new letters/features).
7. Pass bar (proposal): Clean mode >= 8 fps on the 4090 Laptop end-to-end, no sysmem spill, flicker index <= input, VMAF gain >= +3 on clip (c).

## Open questions

- "Enhanced" GAN mode: undecided, leaning no (DECISIONS.md #5).
- DDColor colourise mode: keep as opt-in or drop under decision 1?
- RTX Video SDK: worth a licence review (Windows only, proprietary DLLs, open-source clause), or skip entirely?
- Training-data terms behind weights (DIV2K, REDS, LDV, Vimeo-90K) for a public free app: accept, like other open apps do? [not legal advice]
- TRT engine cache location (proposal in section 4) and whether Clarity joins the shared runtime lock.
