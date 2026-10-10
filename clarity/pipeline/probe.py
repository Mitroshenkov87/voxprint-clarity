"""ffprobe JSON into a small record. HDR and Dolby Vision are refused."""
from __future__ import annotations

from dataclasses import dataclass, field

from clarity.errors import ClarityError

_HDR_TRANSFER = frozenset({"smpte2084", "arib-std-b67", "smpte428"})


@dataclass(frozen=True)
class Probe:
    """The video facts a job needs from ffprobe. HDR is recorded and then refused."""

    codec: str
    width: int
    height: int
    pix_fmt: str
    fps: float
    frames: int
    color_transfer: str
    color_space: str
    color_primaries: str
    color_range: str
    field_order: str
    audio_codecs: tuple[str, ...] = ()
    has_subtitles: bool = False
    hdr: bool = False
    dolby_vision: bool = False
    side_data: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interlaced(self) -> bool:
        """Return whether the field order asks for a deinterlace.

        Returns:
            True unless the field order is empty, progressive, or unknown.
        """
        order = self.field_order.lower()
        return order not in {"", "progressive", "unknown"}


def _float_fps(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "")
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _side_names(stream: dict[str, object]) -> list[str]:
    names: list[str] = []
    raw = stream.get("side_data_list")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                names.append(str(item.get("side_data_type") or ""))
    return names


def parse_probe(data: dict[str, object]) -> Probe:
    """Read ``ffprobe -print_format json -show_streams -show_format``."""
    streams = data.get("streams")
    if not isinstance(streams, list):
        raise ClarityError("ffprobe did not return streams")
    video: dict[str, object] | None = None
    audio: list[str] = []
    subtitles = False
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        kind = str(stream.get("codec_type") or "")
        if kind == "video" and video is None:
            video = stream
        elif kind == "audio":
            audio.append(str(stream.get("codec_name") or ""))
        elif kind == "subtitle":
            subtitles = True
    if video is None:
        raise ClarityError("the file has no video stream")
    sides = _side_names(video)
    transfer = str(video.get("color_transfer") or "")
    hdr = transfer.lower() in _HDR_TRANSFER or any("Mastering display" in name or "Content light" in name for name in sides)
    dolby = any("DOVI" in name.upper() or "Dolby Vision" in name for name in sides)
    frames = _frame_count(video, data)
    return Probe(
        codec=str(video.get("codec_name") or ""),
        width=int(str(video.get("width") or 0)),
        height=int(str(video.get("height") or 0)),
        pix_fmt=str(video.get("pix_fmt") or ""),
        fps=_float_fps(video.get("avg_frame_rate")),
        frames=frames,
        color_transfer=transfer,
        color_space=str(video.get("color_space") or ""),
        color_primaries=str(video.get("color_primaries") or ""),
        color_range=str(video.get("color_range") or ""),
        field_order=str(video.get("field_order") or "progressive"),
        audio_codecs=tuple(audio),
        has_subtitles=subtitles,
        hdr=hdr or dolby,
        dolby_vision=dolby,
        side_data=tuple(sides),
    )


def _frame_count(video: dict[str, object], data: dict[str, object]) -> int:
    """``nb_frames`` when the container stored it, otherwise duration times frame rate."""
    for key in ("nb_frames", "nb_read_frames"):
        text = str(video.get(key) or "")
        if text.isdigit():
            count = int(text)
            if count > 0:
                return count
    fps = _float_fps(video.get("avg_frame_rate"))
    duration = _float_fps(video.get("duration"))
    if duration <= 0:
        formatted = data.get("format")
        if isinstance(formatted, dict):
            duration = _float_fps(formatted.get("duration"))
    if fps > 0 and duration > 0:
        return max(1, int(round(duration * fps)))
    return 0


def require_sdr(probe: Probe) -> None:
    """MVP is SDR only. HDR10 and Dolby Vision stop the job with a clear message."""
    if probe.dolby_vision:
        raise ClarityError(
            "This file is Dolby Vision. Clarity's first version restores SDR only, and a Dolby Vision RPU would be dropped."
        )
    if probe.hdr:
        raise ClarityError(
            "This file is HDR. Clarity's first version restores SDR only. HDR10 is a later version."
        )
    if probe.width < 1 or probe.height < 1:
        raise ClarityError("the video size could not be read")


def probe_command(ffprobe: str, source: str) -> list[str]:
    """Build the ffprobe command that prints streams, format, and chapters as JSON.

    Args:
        ffprobe: ffprobe binary.
        source: Input path.

    Returns:
        The argument list, including ``source``.
    """
    return [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-show_chapters",
        source,
    ]
