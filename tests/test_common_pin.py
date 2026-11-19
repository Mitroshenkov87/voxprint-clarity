"""Wheel digest checks and the staged Enlarge runtime. No network."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from voxprint_common.buildinfo import label, load

_ROOT = Path(__file__).resolve().parents[1]
_SHA = "c9361d5d00913faf39497fc55124e682ba15b2098c775068c49582c0520cd4d1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fetch():
    return _load("clarity_fetch_common", _ROOT / "installer" / "fetch_common.py")


def _stage():
    return _load("clarity_stage_suite_runtime", _ROOT / "installer" / "stage_suite_runtime.py")


def test_sidecar_accepts_the_published_line() -> None:
    module = _fetch()
    text = f"{_SHA}  voxprint_common-0.1.1-py3-none-any.whl\n"
    assert module.parse_sha256_sidecar(text) == _SHA


def test_sidecar_rejects_a_different_name() -> None:
    module = _fetch()
    with pytest.raises(module.FetchError, match="names"):
        module.parse_sha256_sidecar(f"{_SHA}  other.whl\n")


def test_verify_wheel_rejects_a_mismatch(tmp_path: Path) -> None:
    module = _fetch()
    wheel = tmp_path / module.WHEEL_NAME
    wheel.write_bytes(b"not the release wheel")
    sidecar = f"{_SHA}  {module.WHEEL_NAME}\n"
    with pytest.raises(module.FetchError, match="does not match"):
        module.verify_wheel(wheel, sidecar, _SHA)


def test_verify_wheel_accepts_a_matching_file(tmp_path: Path) -> None:
    module = _fetch()
    wheel = tmp_path / module.WHEEL_NAME
    wheel.write_bytes(b"clarity-common")
    digest = module.file_sha256(wheel)
    sidecar = f"{digest}  {module.WHEEL_NAME}\n"
    assert module.verify_wheel(wheel, sidecar, digest) == digest


def test_extract_enlarge_copies_the_package(tmp_path: Path) -> None:
    module = _stage()
    archive = tmp_path / "engine.tar.gz"
    payload = b"def ensure_runtime(app_id):\n    return app_id\n"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("voxprint-enlarge-engine/enlarge/suite_runtime.py")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        extra = tarfile.TarInfo("voxprint-enlarge-engine/enlarge/__init__.py")
        body = b""
        extra.size = len(body)
        tar.addfile(extra, io.BytesIO(body))
    destination = tmp_path / "site-packages" / "enlarge"
    module.extract_enlarge(archive, destination)
    assert (destination / "suite_runtime.py").read_bytes() == payload
    assert (destination / "__init__.py").is_file()


def test_extract_enlarge_refuses_a_path_escape(tmp_path: Path) -> None:
    module = _stage()
    archive = tmp_path / "engine.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("pkg/enlarge/suite_runtime.py")
        body = b"ok\n"
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
        escaped = tarfile.TarInfo("pkg/enlarge/../../outside.py")
        escaped.size = len(body)
        tar.addfile(escaped, io.BytesIO(body))
    with pytest.raises(module.StageError, match="refusing"):
        module.extract_enlarge(archive, tmp_path / "enlarge")


def test_missing_build_stays_null(tmp_path: Path) -> None:
    path = tmp_path / "BUILD.json"
    path.write_text(
        json.dumps({"version": "1.0.0-rc", "build": None, "codename": "North", "app": "clarity"}),
        encoding="utf-8",
    )
    info = load(path, app="clarity", env={})
    assert info.build is None
    text = label(info)
    assert "dev" in text
    assert "build 0" not in text
