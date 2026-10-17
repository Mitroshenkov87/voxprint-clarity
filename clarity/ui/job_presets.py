"""Map the shared settings state onto one Clarity restore job.

Qt is not imported. Model ids live in the preset document and ``mdl.json``.
This module reads those files and the catalogue helpers. It does not name a
weight file.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from voxprint_common.mdl.catalog import AppManifest, discover, load_manifest
from voxprint_common.mdl.readiness import disk_ready
from voxprint_common.ui.presets import PresetFile, SettingState, load_presets
from voxprint_common.ui.settings import Choice, SettingSpec, SettingType, Settings
from voxprint_common.ui.variants import (
    AppSurface,
    settings_for,
    state_names,
    surface_for,
    variant_presets,
)

from clarity.appinfo import repo_root
from clarity.errors import ClarityError
from clarity.models import upscale_model
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings as JobSettings
from clarity.pipeline.job import models_for

_PRESETS = Path(__file__).with_name("presets") / "clarity.json"


@dataclass(frozen=True)
class AdoptedVideo:
    """The clarity surface and the shared video sample it starts from.

    ``shared_presets`` is the package file named by ``preset_name``. Clarity
    does not run a job from it: that file selects H.264 and H.265. The app
    document is :func:`load_clarity_presets`.
    """

    surface: AppSurface
    shared_settings: Settings
    shared_presets: PresetFile
    settings_name: str
    history_name: str


def adopted_video() -> AdoptedVideo:
    """Load ``surface_for("clarity")`` and the shared video preset file.

    Returns:
        The video surface, its settings, its preset file, and the state names.
    """
    surface = surface_for("clarity")
    shared = settings_for(surface)
    presets = variant_presets(surface, shared)
    settings_name, history_name = state_names(surface)
    return AdoptedVideo(surface, shared, presets, settings_name, history_name)


def clarity_settings() -> Settings:
    """Return descriptors whose defaults match :class:`~clarity.pipeline.job.Settings`.

    Expert rows are Restoration, Upscale, Encoding, and Advanced, in that
    order. Simple chips write the same keys the old form wrote.

    Returns:
        A new settings list. The caller owns it.
    """
    settings = Settings()
    settings.add(
        SettingSpec(
            "job.mode",
            "setting.job.mode",
            "setting.job.mode.help",
            SettingType.ENUM,
            "clean",
            group="cleanup",
            choices=(
                Choice("clean", "simple.preset.cleanup.light"),
                Choice("strong", "simple.preset.cleanup.strong"),
            ),
        )
    )
    settings.add(
        SettingSpec(
            "job.restore_models",
            "setting.job.restore_models",
            "setting.job.restore_models.help",
            SettingType.TEXT,
            "",
            expert=True,
            group="restoration",
        )
    )
    settings.add(
        SettingSpec(
            "job.sharpen",
            "setting.job.sharpen",
            "setting.job.sharpen.help",
            SettingType.FLOAT,
            0.35,
            expert=True,
            group="restoration",
            minimum=0.0,
            maximum=1.0,
        )
    )
    settings.add(
        SettingSpec(
            "job.upscale",
            "setting.job.upscale",
            "setting.job.upscale.help",
            SettingType.BOOL,
            False,
            group="size",
        )
    )
    settings.add(
        SettingSpec(
            "job.anime",
            "setting.job.anime",
            "setting.job.anime.help",
            SettingType.BOOL,
            False,
            group="look",
        )
    )
    settings.add(
        SettingSpec(
            "job.upscale_model",
            "setting.job.upscale_model",
            "setting.job.upscale_model.help",
            SettingType.TEXT,
            "",
            expert=True,
            group="upscale",
        )
    )
    settings.add(
        SettingSpec(
            "job.upscale_strength",
            "setting.job.upscale_strength",
            "setting.job.upscale_strength.help",
            SettingType.FLOAT,
            0.0,
            expert=True,
            group="upscale",
            minimum=0.0,
            maximum=1.0,
        )
    )
    settings.add(
        SettingSpec(
            "job.white_balance",
            "setting.job.white_balance",
            "setting.job.white_balance.help",
            SettingType.BOOL,
            True,
            group="colours",
        )
    )
    settings.add(
        SettingSpec(
            "job.levels",
            "setting.job.levels",
            "setting.job.levels.help",
            SettingType.BOOL,
            True,
            group="colours",
        )
    )
    settings.add(
        SettingSpec(
            "job.container",
            "setting.job.container",
            "setting.job.container.help",
            SettingType.ENUM,
            "mkv",
            group="file",
            choices=(
                Choice("mkv", "simple.preset.file.mkv"),
                Choice("mp4", "simple.preset.file.mp4"),
            ),
        )
    )
    settings.add(
        SettingSpec(
            "job.codec",
            "setting.job.codec",
            "setting.job.codec.help",
            SettingType.ENUM,
            "av1",
            expert=True,
            group="encoding",
            choices=(
                Choice("av1", "codec.av1"),
                Choice("hevc", "codec.hevc"),
            ),
        )
    )
    settings.add(
        SettingSpec(
            "job.quality_factor",
            "setting.job.quality_factor",
            "setting.job.quality_factor.help",
            SettingType.FLOAT,
            40.0,
            expert=True,
            group="encoding",
            minimum=0.0,
            maximum=100.0,
        )
    )
    settings.add(
        SettingSpec(
            "job.sigma",
            "setting.job.sigma",
            "setting.job.sigma.help",
            SettingType.FLOAT,
            15.0,
            expert=True,
            group="advanced",
            minimum=0.0,
            maximum=100.0,
        )
    )
    settings.add(
        SettingSpec(
            "job.tile",
            "setting.job.tile",
            "setting.job.tile.help",
            SettingType.INT,
            0,
            expert=True,
            group="advanced",
            minimum=0,
            maximum=2048,
        )
    )
    settings.add(
        SettingSpec(
            "job.batch",
            "setting.job.batch",
            "setting.job.batch.help",
            SettingType.INT,
            2,
            expert=True,
            group="advanced",
            minimum=1,
            maximum=16,
        )
    )
    settings.add(
        SettingSpec(
            "job.overlap",
            "setting.job.overlap",
            "setting.job.overlap.help",
            SettingType.INT,
            16,
            expert=True,
            group="advanced",
            minimum=0,
            maximum=256,
        )
    )
    return settings


def load_clarity_presets(settings: Settings | None = None) -> PresetFile:
    """Read Clarity's ``voxprint-presets/1`` document.

    Args:
        settings: Descriptors the values are checked against. Defaults to
            :func:`clarity_settings`.

    Returns:
        The cleanup, size, look, colours, and file groups.
    """
    return load_presets(_PRESETS, clarity_settings() if settings is None else settings)


def fresh_state(state_dir: Path | None = None, state_name: str = "video-settings.json") -> SettingState:
    """Build a settings state from Clarity's document.

    Args:
        state_dir: Directory for the state file. None keeps it in memory.
        state_name: File name inside ``state_dir``.

    Returns:
        Defaults, then the preset groups.
    """
    settings = clarity_settings()
    return SettingState(
        load_clarity_presets(settings),
        settings,
        state_dir=state_dir,
        state_name=state_name,
    )


def job_from_state(values: Mapping[str, object]) -> JobSettings:
    """Build a restore job from the shared state.

    Anime is stored on the Look chip even when Size is unchanged. The job
    matches the old form: anime is on only when upscale is on as well.
    Audio copy stays on. ``max_frames`` stays the job default.

    Args:
        values: Live setting values, including preset keys.

    Returns:
        Settings the runner can take.

    Raises:
        ClarityError: A mode, container, or codec value is not one Clarity runs.
    """
    mode = _text(values, "job.mode")
    if mode not in {"clean", "strong"}:
        raise ClarityError(f"unknown mode {mode}")
    try:
        container = Container(_text(values, "job.container"))
        codec = VideoCodec(_text(values, "job.codec"))
    except ValueError as exc:
        raise ClarityError(str(exc)) from exc
    upscale = _bool(values, "job.upscale")
    return JobSettings(
        mode=mode,
        upscale=upscale,
        anime=_bool(values, "job.anime") and upscale,
        upscale_strength=_float(values, "job.upscale_strength"),
        sharpen=_float(values, "job.sharpen"),
        white_balance=_bool(values, "job.white_balance"),
        levels=_bool(values, "job.levels"),
        quality_factor=_float(values, "job.quality_factor"),
        sigma=_float(values, "job.sigma"),
        tile=_int(values, "job.tile"),
        batch=_int(values, "job.batch"),
        overlap=_int(values, "job.overlap"),
        container=container,
        codec=codec,
        copy_audio=True,
    )


def apply_job(state: SettingState, job: JobSettings) -> None:
    """Write ``job`` into the shared state.

    The model text comes from :func:`models_for` and :func:`upscale_model`.
    A later group keeps its own keys.

    Args:
        state: The window's settings state.
        job: Settings loaded from a project or a CLI-shaped choice.
    """
    state.set_value("job.mode", job.mode)
    state.set_value("job.restore_models", ",".join(model.id for model in models_for(job)))
    state.set_value("job.sharpen", job.sharpen)
    state.set_value("job.upscale", job.upscale)
    state.set_value("job.anime", job.anime)
    state.set_value("job.upscale_model", upscale_model(anime=job.anime).id)
    state.set_value("job.white_balance", job.white_balance)
    state.set_value("job.levels", job.levels)
    state.set_value("job.container", job.container.value)
    state.set_value("job.codec", job.codec.value)
    state.set_value("job.sigma", job.sigma)
    state.set_value("job.tile", job.tile)
    state.set_value("job.batch", job.batch)
    state.set_value("job.overlap", job.overlap)
    state.set_value("job.quality_factor", job.quality_factor)
    state.set_value("job.upscale_strength", job.upscale_strength)


def output_suffix(container: str) -> str:
    """Return the default output suffix for ``container``.

    Args:
        container: ``mkv`` or ``mp4``.

    Returns:
        ``.clarity.mkv`` or ``.clarity.mp4``.
    """
    if container == Container.MP4.value:
        return ".clarity.mp4"
    return ".clarity.mkv"


def manifest_paths() -> tuple[Path, ...]:
    """Return the clarity manifest ``discover`` found, or the bundled file.

    Returns:
        One path, or none when neither place has a manifest.
    """
    found = next((item for item in discover() if item.id == "clarity"), None)
    if found is not None:
        return (found.path,)
    bundled = repo_root() / "mdl.json"
    if bundled.is_file():
        return (bundled,)
    return ()


def load_clarity_manifest() -> AppManifest | None:
    """Load the clarity model manifest.

    Returns:
        The parsed manifest, or None when no file is present.
    """
    paths = manifest_paths()
    if not paths:
        return None
    return load_manifest(paths[0])


def model_choices(manifest: AppManifest | None) -> tuple[Choice, ...]:
    """Build choices from models that are not marked unavailable.

    Args:
        manifest: A parsed ``mdl.json``. None yields an empty list.

    Returns:
        Choices whose value is ``Model.id`` and whose label is ``model.<id>``.
    """
    if manifest is None:
        return ()
    return tuple(Choice(model.id, f"model.{model.id}") for model in manifest.models if not model.unavailable)


def models_ready() -> bool:
    """Return whether the clarity manifest has the files it requires.

    An empty device list does not probe the GPU. Placeholder models list no
    files, so a placeholder manifest is ready.

    Returns:
        False when no manifest can be found.
    """
    paths = manifest_paths()
    if not paths:
        return False
    return disk_ready("clarity", manifests=list(paths), devices=[])


def _text(values: Mapping[str, object], key: str) -> str:
    value = values[key]
    if not isinstance(value, str):
        raise ClarityError(f"{key} must be text")
    return value


def _bool(values: Mapping[str, object], key: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise ClarityError(f"{key} must be true or false")
    return value


def _float(values: Mapping[str, object], key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ClarityError(f"{key} must be a number")
    return float(value)


def _int(values: Mapping[str, object], key: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClarityError(f"{key} must be an integer")
    return value
