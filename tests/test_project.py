"""Project files, recent list, and the file-association declarations."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from clarity.__main__ import build_parser
from clarity.i18n import tr
from clarity.pipeline.encode import VideoCodec
from clarity.pipeline.job import Settings
from clarity.project import (
    MIME,
    ProjectFormatError,
    fingerprint,
    load_project,
    load_recent,
    remember_recent,
    resolve_source,
    save_project,
)
from clarity.ui.main_window import MainWindow

_ROOT = Path(__file__).resolve().parents[1]


def _video(path: Path, payload: bytes | None = None) -> Path:
    path.write_bytes(payload if payload is not None else b"frame-bytes")
    return path


def test_project_round_trip_preserves_unknown_fields(tmp_path: Path) -> None:
    source = _video(tmp_path / "in.mkv")
    destination = tmp_path / "out.mkv"
    path = tmp_path / "job.vxclarity"
    settings = Settings(mode="strong", sharpen=0.1, codec=VideoCodec.AV1, copy_audio=True)
    first = save_project(path, source=source, settings=settings, destination=destination, thumb=b"\x89PNG\r\n")
    archive = zipfile.ZipFile(path)
    with archive:
        assert archive.namelist()[0] == "mimetype"
        info = archive.getinfo("mimetype")
        assert info.compress_type == zipfile.ZIP_STORED
        raw = path.read_bytes()
        magic = b"mimetype" + MIME.encode("ascii")
        assert raw[30 : 30 + len(magic)] == magic
        manifest = json.loads(archive.read("manifest.json"))
    manifest["editor_note"] = "keep me"
    settings_raw = json.loads(zipfile.ZipFile(path).read("settings.json"))
    settings_raw["custom"] = {"keep": True}
    resume = json.loads(zipfile.ZipFile(path).read("resume.json"))
    resume["note"] = "journal"
    first.manifest["editor_note"] = "keep me"
    first.settings_raw["custom"] = {"keep": True}
    first.resume["note"] = "journal"
    save_project(
        path,
        source=source,
        settings=Settings(mode="clean", copy_audio=True),
        destination=destination,
        resume={"work": str(tmp_path / "work"), "note": "journal"},
        previous=first,
    )
    loaded = load_project(path)
    assert loaded.manifest["editor_note"] == "keep me"
    assert loaded.manifest["format_version"] == "1.0"
    assert loaded.manifest["created"] == first.manifest["created"]
    assert loaded.manifest["created_by"]["app"] == "Voxprint AI Clarity"
    assert loaded.manifest["generator"]["version"] == "1.0.0-rc"
    assert loaded.manifest["source"]["format"] == "mkv"
    assert loaded.manifest["source"]["filename"] == "in.mkv"
    video = loaded.manifest["x_clarity"]["video"]
    assert video["head_sha256"]
    assert loaded.settings_raw["custom"] == {"keep": True}
    assert loaded.settings.mode == "clean"
    assert loaded.settings.copy_audio is True
    assert loaded.resume["note"] == "journal"
    assert loaded.resume["work"].endswith("work")
    assert loaded.thumb == b"\x89PNG\r\n"
    assert loaded.source == source.resolve()
    assert loaded.destination == destination
    found = fingerprint(source, project=path)
    assert found["head_sha256"] == video["head_sha256"]
    assert found["sha256"] == ""
    with zipfile.ZipFile(path) as archive:
        recorded = loaded.manifest["files"]["settings.json"]
        payload = archive.read("settings.json")
        assert recorded["size"] == len(payload)
        assert recorded["sha256"] == hashlib.sha256(payload).hexdigest()


def test_fast_match_ignores_the_middle_of_a_long_file(tmp_path: Path) -> None:
    payload = b"A" * (1024 * 1024) + b"MIDDLE" + b"Z" * (1024 * 1024)
    source = _video(tmp_path / "long.mkv", payload)
    project = tmp_path / "job.vxclarity"
    save_project(project, source=source, settings=Settings(), destination=tmp_path / "out.mkv")
    recorded = load_project(project).manifest["x_clarity"]["video"]
    moved = tmp_path / "elsewhere.mkv"
    body = bytearray(payload)
    body[1024 * 1024 + 1] = ord("Q")
    moved.write_bytes(body)
    assert resolve_source(project, {**recorded, "absolute": str(moved), "relative": moved.name}) == moved


def test_higher_major_version_is_rejected(tmp_path: Path) -> None:
    source = _video(tmp_path / "in.mkv")
    path = tmp_path / "job.vxclarity"
    save_project(path, source=source, settings=Settings(), destination=tmp_path / "out.mkv")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        others = {name: archive.read(name) for name in archive.namelist() if name != "manifest.json"}
    manifest["format_version"] = "2.0"
    rewritten = tmp_path / "newer.vxclarity"
    with zipfile.ZipFile(rewritten, "w") as archive:
        archive.writestr("mimetype", MIME)
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, payload in others.items():
            if name != "mimetype":
                archive.writestr(name, payload)
    with pytest.raises(ProjectFormatError, match="format version"):
        load_project(rewritten)
    manifest["format_version"] = "1.4"
    minor = tmp_path / "minor.vxclarity"
    with zipfile.ZipFile(minor, "w") as archive:
        archive.writestr("mimetype", MIME)
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, payload in others.items():
            if name != "mimetype":
                archive.writestr(name, payload)
    assert load_project(minor).manifest["format_version"] == "1.4"


def test_unlisted_entry_is_ignored(tmp_path: Path) -> None:
    source = _video(tmp_path / "in.mkv")
    path = tmp_path / "job.vxclarity"
    save_project(path, source=source, settings=Settings(), destination=tmp_path / "out.mkv")
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    extra = tmp_path / "extra.vxclarity"
    with zipfile.ZipFile(extra, "w") as archive:
        for info, payload in entries:
            archive.writestr(info, payload)
        archive.writestr("notes/aside.md", b"ignored\n")
    loaded = load_project(extra)
    assert "notes/aside.md" not in loaded.extras
    assert loaded.manifest["format_version"] == "1.0"


def test_oversized_expansion_is_rejected(tmp_path: Path) -> None:
    source = _video(tmp_path / "in.mkv")
    path = tmp_path / "job.vxclarity"
    save_project(path, source=source, settings=Settings(), destination=tmp_path / "out.mkv")
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    bomb = tmp_path / "bomb.vxclarity"
    with zipfile.ZipFile(bomb, "w") as archive:
        for info, payload in entries:
            archive.writestr(info, payload)
        padded = zipfile.ZipInfo("notes/pad.txt")
        padded.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(padded, b"\0" * 200_000)
    with pytest.raises(ProjectFormatError, match="expands"):
        load_project(bomb)


def test_symlink_entry_is_rejected(tmp_path: Path) -> None:
    source = _video(tmp_path / "in.mkv")
    path = tmp_path / "job.vxclarity"
    save_project(path, source=source, settings=Settings(), destination=tmp_path / "out.mkv")
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    linked = tmp_path / "linked.vxclarity"
    with zipfile.ZipFile(linked, "w") as archive:
        for info, payload in entries:
            archive.writestr(info, payload)
        link = zipfile.ZipInfo("escape")
        link.compress_type = zipfile.ZIP_STORED
        link.external_attr = 0o120777 << 16
        archive.writestr(link, b"target")
    with pytest.raises(ProjectFormatError, match="not a regular file"):
        load_project(linked)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    source = _video(tmp_path / "in.mkv")
    path = tmp_path / "job.vxclarity"
    save_project(path, source=source, settings=Settings(), destination=tmp_path / "out.mkv")
    with zipfile.ZipFile(path) as archive:
        others = {name: archive.read(name) for name in archive.namelist()}
    others["settings.json"] = b"{}\n"
    broken = tmp_path / "broken.vxclarity"
    with zipfile.ZipFile(broken, "w") as archive:
        for name, payload in others.items():
            archive.writestr(name, payload)
    with pytest.raises(ProjectFormatError, match="checksum"):
        load_project(broken)


def test_recent_files_keep_the_newest_first(tmp_path: Path) -> None:
    first = tmp_path / "a.vxclarity"
    second = tmp_path / "b.vxclarity"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    remember_recent(first, tmp_path)
    found = remember_recent(second, tmp_path)
    assert found[0] == str(second.resolve())
    assert load_recent(tmp_path)[1] == str(first.resolve())


def test_window_saves_and_opens_a_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_CLARITY_STATE", str(tmp_path / "state"))
    source = _video(tmp_path / "clip.mp4")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(runner=lambda *_args, **_kwargs: None)
    window.source.setText(str(source))
    window.codec.setCurrentIndex(1)
    window.show()
    app.processEvents()
    project = tmp_path / "clip.vxclarity"
    window.save_project(project)
    assert window.status.text() == tr("status.saved")
    assert load_recent()[0] == str(project.resolve())
    window.source.setText("")
    window.codec.setCurrentIndex(0)
    window.open_project(project)
    assert window.source.text() == str(source.resolve())
    assert window.codec.currentData() == VideoCodec.HEVC.value
    assert Path(window.output.text()).name == "clip.clarity.mkv"
    window.source.setText("")
    window.recent.activated.emit(1)
    app.processEvents()
    assert window.source.text() == str(source.resolve())
    window.close()


def test_launch_parser_accepts_a_project() -> None:
    args = build_parser().parse_args(["movie.vxclarity"])
    assert args.project == Path("movie.vxclarity")


def test_installers_declare_the_association() -> None:
    iss = (_ROOT / "installer" / "VoxprintClarity.iss").read_text(encoding="utf-8")
    assert 'ValueData: "Voxprint.Clarity.Project"' in iss
    assert "Root: HKCU;" in iss
    assert "uninsdeletekey" in iss
    desktop = (_ROOT / "installer" / "linux" / "voxprint-clarity.desktop").read_text(encoding="utf-8")
    assert "MimeType=application/vnd.voxprint.clarity+zip;" in desktop
    xml = (_ROOT / "installer" / "linux" / "voxprint-clarity.xml").read_text(encoding="utf-8")
    assert "application/vnd.voxprint.clarity+zip" in xml
    assert "*.vxclarity" in xml
    script = (_ROOT / "installer" / "linux" / "uninstall.sh").read_text(encoding="utf-8")
    assert "voxprint-clarity.xml" in script
