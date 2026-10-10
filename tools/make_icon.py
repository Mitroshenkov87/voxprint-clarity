"""Draw the Clarity icon and write PNG sizes plus a 16-256 Windows .ico.

The attached 1024 source was not on disk in this workspace, so this draws the
same mark: a glowing film frame and two sparks on a dark navy field.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "branding"
NAVY = (8, 16, 48, 255)
GOLD = (255, 176, 32, 255)
HOLE = (6, 12, 36, 255)
SIZES = (16, 32, 48, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 32, 48, 64, 128, 256)


def _star(draw: ImageDraw.ImageDraw, cx: float, cy: float, outer: float, inner: float, fill: tuple[int, int, int, int]) -> None:
    points: list[tuple[float, float]] = []
    for index in range(8):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 4
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    draw.polygon(points, fill=fill)


def draw_icon(size: int = 1024) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = int(size * 0.04)
    radius = int(size * 0.18)
    draw.rounded_rectangle((margin, margin, size - margin, size - margin), radius=radius, fill=NAVY)

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    frame = int(size * 0.14)
    outer = (frame, frame, size - frame, size - frame)
    glow_draw.rounded_rectangle(outer, radius=int(size * 0.08), outline=GOLD, width=max(2, size // 48))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1, size // 80)))
    image.alpha_composite(glow)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(outer, radius=int(size * 0.08), outline=GOLD, width=max(2, size // 28))

    inner_pad = int(size * 0.11)
    inner = (
        frame + inner_pad,
        frame + inner_pad,
        size - frame - inner_pad,
        size - frame - inner_pad,
    )
    draw.rounded_rectangle(inner, radius=int(size * 0.04), outline=GOLD, width=max(2, size // 64))

    holes = 8
    hole_r = max(1, size // 48)
    top_y = frame + int(size * 0.055)
    bot_y = size - frame - int(size * 0.055)
    left = inner[0] + hole_r
    right = inner[2] - hole_r
    for index in range(holes):
        cx = left + (right - left) * index / (holes - 1)
        draw.ellipse((cx - hole_r, top_y - hole_r, cx + hole_r, top_y + hole_r), fill=HOLE)
        draw.ellipse((cx - hole_r, bot_y - hole_r, cx + hole_r, bot_y + hole_r), fill=HOLE)

    _star(draw, size * 0.46, size * 0.50, size * 0.16, size * 0.035, GOLD)
    _star(draw, size * 0.68, size * 0.64, size * 0.055, size * 0.014, GOLD)
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master = draw_icon(1024)
    master.save(OUT / "clarity-1024.png")
    resized: dict[int, Image.Image] = {}
    for size in SIZES:
        image = master if size == 1024 else master.resize((size, size), Image.Resampling.LANCZOS)
        image.save(OUT / f"clarity-{size}.png")
        resized[size] = image
    ico = resized[256]
    ico.save(
        OUT / "clarity.ico",
        format="ICO",
        sizes=[(size, size) for size in ICO_SIZES],
        append_images=[resized[size] for size in ICO_SIZES if size != 256],
    )


if __name__ == "__main__":
    main()
