"""Install paths, the suite registry, and a silent upgrade of an old install."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from clarity.appinfo import APP_NAME
from clarity.cli import splash_identity
from clarity.install_layout import (
    APP_PATHS,
    CLI_EXE,
    CLI_NAME,
    DESKTOP_FILE,
    GUI_EXE,
    GUI_NAME,
    LINUX_DIR,
    OLD_DESKTOP_FILE,
    OLD_MENU,
    OLD_WINDOWS_DIR_NAME,
    PUBLISHER,
    REGISTRY_SUBKEY,
    SETUP_LOG,
    SUITE_MENU,
    WINDOWS_APP_FOLDER,
)
from clarity.project import state_dir

_ROOT = Path(__file__).resolve().parents[1]


def test_display_name_and_user_data_path(monkeypatch) -> None:
    assert APP_NAME == "VOXPrint AI Clarity"
    identity = splash_identity()
    assert identity.app == APP_NAME
    monkeypatch.delenv("VOXPRINT_CLARITY_STATE", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    folder = state_dir()
    if os.name == "nt":
        assert folder.parts[-2:] == ("Voxprint", "Clarity")
    else:
        assert folder.parts[-2:] == ("voxprint", "clarity")
    source = (_ROOT / "clarity" / "project.py").read_text(encoding="utf-8")
    assert 'windows_base / "Voxprint" / "Clarity"' in source


def test_windows_script_uses_the_suite_folder() -> None:
    iss = (_ROOT / "installer" / "VoxprintClarity.iss").read_text(encoding="utf-8")
    assert PUBLISHER == "VOXPrint AI Suite"
    assert f'#define Publisher "{PUBLISHER}"' in iss
    assert f'#define SuiteFolder "{SUITE_MENU}"' in iss
    assert f'#define AppFolder "{WINDOWS_APP_FOLDER}"' in iss
    assert r"DefaultDirName={autopf}\{#SuiteFolder}\{#AppFolder}" in iss
    assert "DefaultGroupName={#SuiteFolder}" in iss
    assert "UsePreviousAppDir=no" in iss
    assert GUI_EXE in iss
    assert CLI_EXE in iss
    assert OLD_WINDOWS_DIR_NAME in iss
    assert "{commonprograms}\\{#OldMenu}" in iss
    assert f'#define OldMenu "{OLD_MENU}"' in iss
    assert REGISTRY_SUBKEY in iss
    assert "InstallPath" in iss
    assert 'ValueName: "Exe"' in iss
    assert 'ValueName: "ProtocolVersion"' in iss
    assert APP_PATHS in iss
    assert "UnderLocalAppData" in iss
    assert SETUP_LOG in iss
    assert "Voxprint AI Clarity.lnk" in iss
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'{GUI_NAME} = "clarity.__main__:main"' in pyproject
    assert f'{CLI_NAME} = "clarity.__main__:main"' in pyproject


def test_linux_package_uses_the_suite_directory() -> None:
    desktop = (_ROOT / "installer" / "linux" / DESKTOP_FILE).read_text(encoding="utf-8")
    assert "Name=VOXPrint AI Clarity" in desktop
    assert f'Exec="{LINUX_DIR}/{GUI_NAME}" %f' in desktop
    assert "X-Voxprint-Ipc=" in desktop
    assert "X-Voxprint-Protocol=0" in desktop
    package = (_ROOT / "installer" / "linux" / "build-package.sh").read_text(encoding="utf-8")
    assert LINUX_DIR.rsplit("/", 1)[0] in package
    assert "voxprint-clarity-linux.tar.gz" in package
    assert CLI_NAME in package
    uninstall = (_ROOT / "installer" / "linux" / "uninstall.sh").read_text(encoding="utf-8")
    assert "voxprint-clarity.xml" in uninstall
    assert ".local/share/voxprint" in (_ROOT / "installer" / "linux" / "migrate-old.sh").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="The Linux migration script needs a POSIX shell")
def test_linux_migration_keeps_user_data(tmp_path: Path) -> None:
    root = tmp_path / "sys"
    old_bin = root / "usr" / "bin"
    old_bin.mkdir(parents=True)
    (old_bin / GUI_NAME).write_text("#!/bin/sh\n", encoding="utf-8")
    desktop_dir = root / "usr" / "share" / "applications"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / OLD_DESKTOP_FILE).write_text("Name=Voxprint AI Clarity\n", encoding="utf-8")
    data = root / "home" / "user" / ".local" / "share" / "voxprint" / "models"
    data.mkdir(parents=True)
    weight = data / "weight.bin"
    weight.write_text("keep", encoding="utf-8")
    new_app = root / "opt" / "voxprint-ai-suite" / "clarity"
    new_app.mkdir(parents=True)
    (new_app / "main.py").write_text("print('clarity')\n", encoding="utf-8")
    script = _ROOT / "installer" / "linux" / "migrate-old.sh"
    subprocess.run(["bash", str(script), str(root), str(new_app)], check=True)
    assert not (old_bin / GUI_NAME).exists()
    assert not (desktop_dir / OLD_DESKTOP_FILE).exists()
    assert weight.read_text(encoding="utf-8") == "keep"
    assert (new_app / "main.py").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="The Linux desktop script needs a POSIX shell")
def test_user_desktop_registration_keeps_user_data(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data = home / ".local" / "share" / "voxprint" / "models"
    data.mkdir(parents=True)
    weight = data / "weight.bin"
    weight.write_text("keep", encoding="utf-8")
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True)
    (apps / OLD_DESKTOP_FILE).write_text("old\n", encoding="utf-8")
    app = tmp_path / "clarity"
    app.mkdir()
    (app / "BUILD.json").write_text('{"version": "1.0.0-rc"}\n', encoding="utf-8")
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("XDG_DATA_HOME", None)
    subprocess.run(
        ["bash", str(_ROOT / "installer" / "linux" / "register-user.sh"), str(app)],
        check=True,
        env=env,
    )
    desktop = (apps / DESKTOP_FILE).read_text(encoding="utf-8")
    assert "Name=VOXPrint AI Clarity" in desktop
    assert f'Exec="{app}/voxprint-clarity" %f' in desktop
    assert "X-Voxprint-Version=1.0.0-rc" in desktop
    assert not (apps / OLD_DESKTOP_FILE).exists()
    assert weight.read_text(encoding="utf-8") == "keep"
