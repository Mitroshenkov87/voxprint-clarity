"""NVENC commands. AV1 by default. No H.264, and no bundled x264 or x265."""
from __future__ import annotations

import re
from enum import Enum
from typing import Never

from clarity.errors import ClarityError

_ENCODER_LINE = re.compile(r"^\s*[VAS][A-Za-z.]{5}\s+(\S+)")


class Container(Enum):
    """Output container. Audio is copied into the same file."""

    MKV = "mkv"
    MP4 = "mp4"


class VideoCodec(Enum):
    """Output codec. AV1 is the default. HEVC is NVENC only."""

    AV1 = "av1"
    HEVC = "hevc"


def assert_encoder_allowed(name: str) -> None:
    """Reject H.264 and the GPL software encoders Clarity must not invoke."""
    token = name.strip().lower()
    if "264" in token or "x265" in token or token in {"h264", "mpeg4", "libx264", "libx265"}:
        raise ClarityError(f"Clarity does not write {name}. Output is AV1, or HEVC only through hevc_nvenc.")


def select_encoder(codec: VideoCodec, available: set[str]) -> str:
    """Pick a present encoder. SVT-AV1 is the CPU fallback only when that LGPL build has it."""
    have = {item.lower() for item in available}
    if codec is VideoCodec.AV1:
        if "av1_nvenc" in have:
            return "av1_nvenc"
        if "libsvtav1" in have:
            return "libsvtav1"
        raise ClarityError(
            "This ffmpeg has neither av1_nvenc nor libsvtav1. "
            "Clarity does not fall back to H.264. Use an LGPL ffmpeg build with one of those encoders."
        )
    if codec is VideoCodec.HEVC:
        if "hevc_nvenc" in have:
            return "hevc_nvenc"
        raise ClarityError(
            "HEVC was requested, but hevc_nvenc is not in this ffmpeg. Clarity does not bundle x265."
        )
    raise ClarityError(f"unknown codec {codec!r}")


def planned_encoder(codec: VideoCodec, available: set[str]) -> tuple[str, bool]:
    """The encoder a job would use, and whether this ffmpeg actually has it.

    A dry-run on a machine without NVENC still names ``av1_nvenc`` or
    ``hevc_nvenc`` so the printed plan is the GPU plan. It does not run it.
    """
    if codec is VideoCodec.AV1:
        preferred = "av1_nvenc"
    elif codec is VideoCodec.HEVC:
        preferred = "hevc_nvenc"
    else:
        unknown: Never = codec
        raise ClarityError(f"unknown codec {unknown!r}")
    try:
        return select_encoder(codec, available), True
    except ClarityError:
        return preferred, False


def encoder_pix_fmt(source_pix_fmt: str) -> str:
    """The pixel format NVENC will accept for this source.

    rgb24 on the pipe makes ffmpeg pick yuv444p, and av1_nvenc rejects that.
    8-bit sources are written as yuv420p. A 10-bit or deeper source is written
    as p010le.

    Args:
        source_pix_fmt: The ``pix_fmt`` ffprobe reported. Empty means 8-bit.

    Returns:
        ``yuv420p`` or ``p010le``.
    """
    folded = source_pix_fmt.lower()
    deep = ("p010", "p016", "10le", "10be", "12le", "12be", "16le", "16be")
    if any(token in folded for token in deep):
        return "p010le"
    return "yuv420p"


def video_args(encoder: str, cq: int = 20) -> list[str]:
    """ffmpeg video-encoder flags for one allowed encoder.

    Args:
        encoder: ``av1_nvenc``, ``hevc_nvenc``, or ``libsvtav1``.
        cq: Constant-quality value passed as ``-cq`` or ``-crf``.

    Returns:
        Flags starting at ``-c:v``.

    Raises:
        ClarityError: When ``encoder`` is H.264, x265, or otherwise unsupported.
    """
    assert_encoder_allowed(encoder)
    quality = str(int(cq))
    if encoder == "av1_nvenc":
        return ["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", quality]
    if encoder == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", quality]
    if encoder == "libsvtav1":
        return ["-c:v", "libsvtav1", "-preset", "8", "-crf", quality]
    raise ClarityError(f"unsupported encoder {encoder}")


def audio_args(*, copy: bool) -> list[str]:
    """Stream-copy the original track. Opus only when a re-encode cannot be avoided."""
    if copy:
        return ["-c:a", "copy"]
    return ["-c:a", "libopus", "-b:a", "128k"]


def decode_command(ffmpeg: str, source: str, *, deinterlace: bool) -> list[str]:
    """NVDEC to raw rgb24 on stdout. ``bwdif`` runs first when the source is interlaced."""
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", "-i", source]
    if deinterlace:
        command.extend(["-vf", "bwdif"])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return command


def encode_command(
    ffmpeg: str,
    source_for_audio: str,
    *,
    container: Container,
    encoder: str,
    destination: str,
    copy_audio: bool,
    width: int,
    height: int,
    fps: float,
    pix_fmt: str = "yuv420p",
) -> list[str]:
    """Raw rgb24 on stdin, plus stream-copied audio, subtitles, and chapters from the source.

    ``pix_fmt`` is the encoder output format (``yuv420p`` or ``p010le``), placed
    after ``-c:v`` so ffmpeg does not leave rgb24 as yuv444p.
    """
    suffix = "matroska" if container is Container.MKV else "mp4"
    rate = f"{fps:.6f}" if fps > 0 else "24000/1001"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        rate,
        "-i",
        "pipe:0",
        "-i",
        source_for_audio,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map_chapters",
        "1",
        "-map_metadata",
        "1",
        *video_args(encoder),
        "-pix_fmt",
        pix_fmt,
        *audio_args(copy=copy_audio),
        "-c:s",
        "copy",
        "-f",
        suffix,
        destination,
    ]
    return command


def decode_range_command(
    ffmpeg: str,
    source: str,
    start: int,
    stop: int,
    *,
    deinterlace: bool,
) -> list[str]:
    """NVDEC to raw rgb24 for the half-open frame range ``[start, stop)``."""
    if stop <= start:
        raise ClarityError("a chunk must contain at least one frame")
    filters: list[str] = []
    if deinterlace:
        filters.append("bwdif")
    filters.append(f"select='between(n,{int(start)},{int(stop) - 1})'")
    filters.append("setpts=N/FRAME_RATE/TB")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-i",
        source,
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(int(stop) - int(start)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]


def encode_chunk_command(
    ffmpeg: str,
    *,
    encoder: str,
    destination: str,
    width: int,
    height: int,
    fps: float,
    pix_fmt: str = "yuv420p",
) -> list[str]:
    """One restored chunk, video only. Audio is copied once when the chunks are joined.

    ``pix_fmt`` is the encoder output format, after ``-c:v``.
    """
    rate = f"{fps:.6f}" if fps > 0 else "24000/1001"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        rate,
        "-i",
        "pipe:0",
        *video_args(encoder),
        "-pix_fmt",
        pix_fmt,
        "-an",
        "-f",
        "matroska",
        destination,
    ]


def concat_list(paths: list[str]) -> str:
    """The concat demuxer list. Paths are escaped for a single-quoted ffmpeg line."""
    lines: list[str] = []
    for path in paths:
        escaped = path.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


def concat_command(
    ffmpeg: str,
    list_file: str,
    source_for_audio: str,
    *,
    container: Container,
    destination: str,
    copy_audio: bool,
) -> list[str]:
    """Join finished chunks with ``-c copy``. Video is not encoded a second time."""
    suffix = "matroska" if container is Container.MKV else "mp4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-i",
        source_for_audio,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map_chapters",
        "1",
        "-map_metadata",
        "1",
        "-c",
        "copy",
        *audio_args(copy=copy_audio),
        "-c:s",
        "copy",
        "-f",
        suffix,
        destination,
    ]
    return command


def parse_encoder_list(text: str) -> set[str]:
    """Names from ``ffmpeg -encoders``."""
    names: set[str] = set()
    for line in text.splitlines():
        match = _ENCODER_LINE.match(line)
        if match:
            names.add(match.group(1))
    return names
