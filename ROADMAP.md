# Roadmap

Clarity restores the frames you already have. It does not invent new picture.
This page is what is coming next, what can wait, and what will stay out.

## Next version

- Fixes from the first restore run on a real RTX 4090.
- TensorRT in fp16 by default, on the steps where it is faster than the current path.
- A model tier chosen from the memory on the card. A 16 GB GPU and a 24 GB or larger GPU do not offer the same set. Clarity keeps about 2–3 GB free. A control the card cannot run is hidden. Settings includes an advanced override when you want to turn a hidden option back on.
- Smart memory placement, described below.
- The command line brought level with the window, so a script can do what the buttons do.

## Later

- HDR10 in and HDR10 out. The current version restores SDR only.
- An optional high-quality temporal tier, in the BasicVSR++ class, for a flicker-free result when you can wait.
- More faithful restoration models as they appear under a permissive licence (MIT, Apache-2.0, or BSD). They stay non-generative.
- A queue for restoring many files one after another.

## Not planned

- Generative repainting, including any “enhanced” look that invents texture.
- Frame interpolation.
- Still-image upscaling. That is Voxprint AI Enlarge.

## Smart memory placement

Clarity’s models are small, so they are loaded one at a time and only for the step that needs them. Decoded frames stay in pinned system memory and stream to the GPU while the previous batch is still running. Decode, restore, and encode overlap. The next chunk is fetched before the current one finishes. Finished chunks are written to disk instead of being held in RAM. The batch size follows the VRAM that is free.

Mixture-of-experts offload and key-value caches are for very large models. They do not apply here.
