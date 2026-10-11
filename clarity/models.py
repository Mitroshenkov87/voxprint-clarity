"""Clarity's model catalogue. Weights stay in the shared Voxprint models folder.

Only MIT, Apache-2.0, and BSD. A non-commercial or missing licence is refused
before a download. Hashes were computed from the official release assets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarity.engine import ensure_weight

ALLOWED = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"})

_KAIR = "https://github.com/cszn/KAIR/releases/download/v1.0"
_ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0"


@dataclass(frozen=True)
class ClarityModel:
    """One catalogue weight: identity, licence, hash, and the engine manifest fields."""

    id: str
    name: str
    arch: str
    scale: int
    role: str
    licence: str
    author: str
    attribution: str
    source_url: str
    sha256: str
    vram_mb: int
    tile: int
    filename: str
    summary: str
    download_url: str
    condition: str
    channels: int
    selectable: bool = True
    precision: str = "fp16"

    def manifest(self) -> dict[str, Any]:
        """Fields ``voxprint_engine.registry.validate_manifest`` accepts."""
        return {
            "id": self.id,
            "name": self.name,
            "arch": self.arch,
            "scale": self.scale,
            "role": self.role,
            "licence": self.licence,
            "author": self.author,
            "attribution": self.attribution,
            "source_url": self.source_url,
            "sha256": self.sha256,
            "vram_mb": self.vram_mb,
            "tile": self.tile,
            "filename": self.filename,
            "summary": self.summary,
            "selectable": self.selectable,
            "download_url": self.download_url,
            "in_channels": self.channels,
        }


def licence_allowed(licence: str) -> bool:
    """True only for the permissive SPDX ids Clarity will fetch."""
    text = licence.strip().lower().replace("_", "-")
    if "non-commercial" in text or "noncommercial" in text or "gpl" in text:
        return False
    folded = {"bsd-3": "bsd-3-clause", "bsd-2": "bsd-2-clause", "bsd": "bsd-3-clause"}.get(text, text)
    return folded in ALLOWED


def _model(**kwargs: Any) -> ClarityModel:
    if not licence_allowed(str(kwargs["licence"])):
        raise ValueError(f"refusing {kwargs['id']}: {kwargs['licence']}")
    return ClarityModel(**kwargs)


CATALOG: tuple[ClarityModel, ...] = (
    _model(
        id="drunet-deblocking-color",
        name="DRUNet deblocking (color)",
        arch="drunet",
        scale=1,
        role="restore",
        licence="MIT",
        author="Kai Zhang",
        attribution="DRUNet deblocking_color from the KAIR v1.0 release (MIT). DPIR, Kai Zhang.",
        source_url="https://github.com/cszn/KAIR",
        sha256="b26323e888230b1b5b58957add8dc9b9ea7fa2f69bdf6a0c3cca9db82167209f",
        vram_mb=1800,
        tile=512,
        filename="drunet_deblocking_color.pth",
        summary="Default Clean step. The fourth channel is (100 - JPEG quality) / 100, as in DPIR.",
        download_url=f"{_KAIR}/drunet_deblocking_color.pth",
        condition="deblock",
        channels=4,
        precision="fp32",
    ),
    _model(
        id="drunet-color",
        name="DRUNet denoise (color)",
        arch="drunet",
        scale=1,
        role="restore",
        licence="MIT",
        author="Kai Zhang",
        attribution="DRUNet color denoiser from the KAIR v1.0 release (MIT). DPIR, Kai Zhang.",
        source_url="https://github.com/cszn/KAIR",
        sha256="479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
        vram_mb=1800,
        tile=512,
        filename="drunet_color.pth",
        summary="Default Clean denoise. The fourth channel is sigma/255.",
        download_url=f"{_KAIR}/drunet_color.pth",
        condition="sigma",
        channels=4,
        precision="fp32",
    ),
    _model(
        id="scunet-color-real-psnr",
        name="SCUNet real_psnr (color)",
        arch="scunet",
        scale=1,
        role="restore",
        licence="Apache-2.0",
        author="Kai Zhang",
        attribution=(
            "SCUNet color real_psnr. Model code is Apache-2.0 (cszn/SCUNet). "
            "The weight file is the KAIR v1.0 asset; that repository is MIT."
        ),
        source_url="https://github.com/cszn/SCUNet",
        sha256="fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
        vram_mb=3600,
        tile=256,
        filename="scunet_color_real_psnr.pth",
        summary="Optional Strong clean. Blind, faithful, slower than DRUNet. Not the default.",
        download_url=f"{_KAIR}/scunet_color_real_psnr.pth",
        condition="none",
        channels=3,
    ),
    _model(
        id="realesr-general-x4v3",
        name="realesr-general-x4v3",
        arch="srvggnet",
        scale=4,
        role="upscale",
        licence="BSD-3-Clause",
        author="Xintao Wang",
        attribution="Compact Real-ESRGAN general v3, BSD-3-Clause, Real-ESRGAN v0.2.5.0.",
        source_url="https://github.com/xinntao/Real-ESRGAN",
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        vram_mb=1100,
        tile=512,
        filename="realesr-general-x4v3.pth",
        summary="Optional 2x path: run this 4x compact net, then downscale to 2x. Off by default.",
        download_url=f"{_ESRGAN}/realesr-general-x4v3.pth",
        condition="none",
        channels=3,
    ),
    _model(
        id="realesr-general-wdn-x4v3",
        name="realesr-general-wdn-x4v3",
        arch="srvggnet",
        scale=4,
        role="upscale",
        licence="BSD-3-Clause",
        author="Xintao Wang",
        attribution="Compact Real-ESRGAN general v3 denoise twin, BSD-3-Clause. Blended with the general net by strength.",
        source_url="https://github.com/xinntao/Real-ESRGAN",
        sha256="1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        vram_mb=1100,
        tile=512,
        filename="realesr-general-wdn-x4v3.pth",
        summary="Denoise twin blended with realesr-general-x4v3 when an upscale strength is set.",
        download_url=f"{_ESRGAN}/realesr-general-wdn-x4v3.pth",
        condition="none",
        channels=3,
    ),
    _model(
        id="realesr-animevideov3",
        name="realesr-animevideov3",
        arch="srvggnet",
        scale=4,
        role="upscale",
        licence="BSD-3-Clause",
        author="Xintao Wang",
        attribution="Compact anime video v3, BSD-3-Clause, Real-ESRGAN v0.2.5.0. Anime preset only.",
        source_url="https://github.com/xinntao/Real-ESRGAN",
        sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        vram_mb=1100,
        tile=512,
        filename="realesr-animevideov3.pth",
        summary="Anime preset for the optional 2x path. Off unless the anime switch is on.",
        download_url=f"{_ESRGAN}/realesr-animevideov3.pth",
        condition="none",
        channels=3,
    ),
)

_BY_ID = {model.id: model for model in CATALOG}


def get_model(model_id: str) -> ClarityModel:
    """Look up one catalogue entry.

    Args:
        model_id: Catalogue id, such as ``drunet-color``.

    Returns:
        The matching model.

    Raises:
        KeyError: When ``model_id`` is not in the catalogue.
    """
    try:
        return _BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(model_id) from exc


def clean_models() -> tuple[ClarityModel, ClarityModel]:
    """Return the deblocking net and the denoise net used by Clean mode.

    Returns:
        ``drunet-deblocking-color``, then ``drunet-color``.
    """
    return get_model("drunet-deblocking-color"), get_model("drunet-color")


def strong_model() -> ClarityModel:
    """Return the single net used by Strong mode.

    Returns:
        ``scunet-color-real-psnr``.
    """
    return get_model("scunet-color-real-psnr")


def upscale_model(*, anime: bool) -> ClarityModel:
    """Return the optional 2x net.

    Args:
        anime: Use the anime preset when true, otherwise the general net.

    Returns:
        ``realesr-animevideov3`` or ``realesr-general-x4v3``.
    """
    if anime:
        return get_model("realesr-animevideov3")
    return get_model("realesr-general-x4v3")


def catalogue_rows() -> list[dict[str, Any]]:
    """Machine-readable catalogue. No download and no GPU."""
    rows: list[dict[str, Any]] = []
    for model in CATALOG:
        rows.append(
            {
                "id": model.id,
                "name": model.name,
                "role": model.role,
                "licence": model.licence,
                "filename": model.filename,
                "sha256": model.sha256,
                "download_url": model.download_url,
            }
        )
    return rows


def fetch_weights(ids: list[str] | None = None, *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Fetch catalogue weights through the shared engine store.

    ``dry_run`` lists the files that would be fetched and does not download them.
    """
    chosen = [get_model(model_id) for model_id in ids] if ids else list(CATALOG)
    rows: list[dict[str, Any]] = []
    for model in chosen:
        row: dict[str, Any] = {
            "id": model.id,
            "filename": model.filename,
            "sha256": model.sha256,
            "download_url": model.download_url,
        }
        if not dry_run:
            path: Path = ensure_weight(model.manifest())
            row["path"] = str(path)
        rows.append(row)
    return rows
