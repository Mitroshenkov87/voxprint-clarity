"""``python -m clarity`` opens the window after the OS and GPU gates."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from clarity.appinfo import APP_NAME, version_label
from clarity.engine_shim import (
    EngineUnavailable,
    assess_launch,
    models_directory,
    register_user,
    unregister_user,
)
from clarity.errors import ClarityError
from clarity.platform_gate import assess_platform
from clarity.ui.main_window import MainWindow


def _register(out: Path | None, *, remove: bool) -> int:
    try:
        if remove:
            others = unregister_user("clarity")
            folder = models_directory()
        else:
            register_user("clarity")
            others = []
            folder = models_directory()
    except EngineUnavailable as exc:
        print(exc)
        return 2
    if out is not None:
        out.write_text(f"{len(others)}\n{folder}\n", encoding="utf-8")
    print("\n".join(others) if others else "clarity")
    return 0


def _selftest() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    if window.start.text().strip() == "":
        print("the Start button has no label")
        return 1
    print("selftest ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clarity", description=APP_NAME)
    parser.add_argument("project", nargs="?", type=Path, help="Open a .vxclarity project")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--register-models-user", action="store_true")
    parser.add_argument("--unregister-models-user", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(version_label())
        return 0
    if args.register_models_user:
        return _register(args.out, remove=False)
    if args.unregister_models_user:
        return _register(args.out, remove=True)
    if args.selftest:
        try:
            return _selftest()
        except (ImportError, ClarityError, OSError) as exc:
            print(exc)
            return 1
    platform = assess_platform()
    if not platform.ok:
        print(platform.message)
        return 2
    try:
        gate = assess_launch()
    except EngineUnavailable as exc:
        print(exc)
        return 2
    if not gate.ok:
        print(gate.message)
        return 2
    if gate.warning:
        print(gate.warning)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen" if not sys.stdout.isatty() else "")
    application = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    if args.project is not None:
        window.open_project(args.project)
    window.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
