# VOXPrint AI Clarity: product decisions

Decided by Aleksandr on 2026-10-10. These override the research recommendations in [RESEARCH.md](RESEARCH.md) where they differ.

1. **Lightweight video "quality leveller".** Target near real-time processing. Short clips must never take hours or days. No heavy or generative models in the app (no diffusion, no SeedVR/SeedVR2, FlashVSR, UltraVSR, DLoRAL and similar).
2. **No frame interpolation.** It generates new frames, so it is out (no RIFE etc.).
3. **Upscale is optional and off by default.** For example 720p -> 1080p or 2x. Only lightweight non-generative models such as realesr-general-x4v3 (Compact).
4. **MVP is SDR-only.** HDR10 is deferred to a later version.
5. **The GAN "Enhanced" mode is undecided, leaning no**, given the lightweight goal.
6. **Output formats: modern open royalty-free by default** (suite rule, COLLABORATION.md section 3). Video: AV1 by default (`av1_nvenc` on GPU, SVT-AV1 CPU fallback) in MKV (default) or MP4. Audio: stream copy of the original track by default when the container allows it (no quality loss); when re-encoding is required, Opus. H.265 only as an option and only via the system/GPU codec (`hevc_nvenc`) when present; never bundle x265. No H.264 output at all. Input of any codec (incl. H.264) is still accepted.

Consequences for the stack:
- Defaults are fast frame-wise faithful models (DRUNet deblock/denoise) plus classical sharpening and colour cleanup.
- SCUNet and BasicVSR++/RealBasicVSR are optional slow tiers at most, never defaults, and may be dropped if they miss the speed target.
- DDColor colourisation guesses colours; whether it fits decision 1 is still open (opt-in at most).
