"""Headless commands. The GUI and these commands share ``run_job`` and the restore plan."""
from __future__ import annotations

import argparse
import json
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal, Never

from PySide6.QtWidgets import QApplication

from clarity.appinfo import APP_BUILD, APP_NAME, APP_VERSION, CODENAME, version_label
from clarity.engine import (
    EngineUnavailable,
    assess_launch,
    models_directory,
    register_user,
    unregister_user,
)
from clarity.errors import Cancelled, ClarityError
from clarity.models import catalogue_rows, fetch_weights
from clarity.pipeline.encode import Container, VideoCodec
from clarity.pipeline.job import Settings
from clarity.pipeline.runner import JobSpec, dry_run_restore, run_job
from clarity.platform_gate import assess_platform
from clarity.project import ProjectFormatError, load_project, work_dir_for
from clarity.suite_repair import SuiteMissing, ensure_suite
from clarity.suite_share import release_app
from clarity.ui.main_window import MainWindow
from clarity.ui.suite_dialog import RepairProgress

CommandName = Literal["restore", "project", "info", "version", "models", "selftest"]
ProjectAction = Literal["open", "resume"]
ModelsAction = Literal["list", "fetch"]
_COMMANDS = frozenset({"restore", "project", "info", "version", "models", "selftest"})


class ExitCode(IntEnum):
    """Process status for every headless command. See ``docs/CLI.md``."""

    OK = 0
    USAGE = 1
    ENVIRONMENT = 2
    JOB = 3


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: {message}", file=sys.stderr)
        raise SystemExit(ExitCode.USAGE)


class Console:
    """Stdout for one command. JSON lines when ``json_mode`` is set, otherwise plain text."""

    def __init__(self, json_mode: bool) -> None:
        """Remember whether results and progress are JSON.

        Args:
            json_mode: Write result and progress lines to stdout.
        """
        self.json_mode = json_mode

    def fail(self, message: str, code: ExitCode) -> int:
        """Print a failure and return its process status.

        Args:
            message: Shown on stderr, and as the JSON ``error`` field.
            code: Process status to return.

        Returns:
            The integer value of ``code``.
        """
        if self.json_mode:
            print(json.dumps({"type": "result", "ok": False, "error": message}), flush=True)
        print(message, file=sys.stderr)
        return int(code)

    def ok(self, payload: dict[str, Any]) -> int:
        """Print a successful result and return 0.

        Args:
            payload: Fields merged into the result object.

        Returns:
            ``ExitCode.OK``.
        """
        body = {"type": "result", "ok": True, **payload}
        if self.json_mode:
            print(json.dumps(body), flush=True)
        else:
            _human(body)
        return int(ExitCode.OK)

    def progress(self, fraction: float, message: str) -> None:
        """Report job progress. JSON goes to stdout; plain text goes to stderr.

        Args:
            fraction: Completion from 0 to 1.
            message: Short status, such as a chunk count.
        """
        if self.json_mode:
            print(json.dumps({"type": "progress", "fraction": fraction, "message": message}), flush=True)
        else:
            print(message, file=sys.stderr)


def _human(body: dict[str, Any]) -> None:
    plan = body.get("plan")
    if isinstance(plan, dict):
        print(f"{plan.get('source')} -> {plan.get('destination')}")
        print(
            f"mode {plan.get('mode')} upscale {plan.get('upscale')} "
            f"encoder {plan.get('encoder')} available {plan.get('encoder_available')}"
        )
        models = plan.get("models")
        if isinstance(models, list):
            print("models " + ", ".join(str(item) for item in models))
        chunks = plan.get("chunks")
        if isinstance(chunks, list):
            print(f"chunks {len(chunks)}")
            for chunk in chunks:
                if isinstance(chunk, dict):
                    print("decode " + " ".join(str(part) for part in chunk.get("decode") or []))
                    print("encode " + " ".join(str(part) for part in chunk.get("encode") or []))
        concat = plan.get("concat")
        if isinstance(concat, list):
            print("concat " + " ".join(str(part) for part in concat))
        scene = plan.get("scene_command")
        if isinstance(scene, list):
            print("scene " + " ".join(str(part) for part in scene))
        return
    models = body.get("models")
    if isinstance(models, list) and body.get("command") == "models":
        for row in models:
            if isinstance(row, dict):
                print(str(row.get("id")))
            else:
                print(str(row))
        return
    if "label" in body and body.get("command") in {"info", "version"}:
        print(str(body["label"]))
        return
    if "message" in body and "source" not in body:
        print(str(body["message"]))
        return
    if body.get("command") == "project":
        print(str(body.get("path") or ""))
        print(f"source {body.get('source')}")
        print(f"destination {body.get('destination')}")
        print(f"mode {body.get('mode')} chunks {body.get('finished_chunks')}")
        return
    print(json.dumps(body, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """The window launcher. A positional ``.vxclarity`` opens that project."""
    parser = _Parser(prog="clarity", description=APP_NAME)
    parser.add_argument("project", nargs="?", type=Path, help="Open a .vxclarity project")
    parser.add_argument("--version", action="store_true", help="Print the version from BUILD.json and exit")
    parser.add_argument("--selftest", action="store_true", help="Open the window offscreen and exit")
    parser.add_argument("--register-models-user", action="store_true", help="Register Clarity in the shared models folder")
    parser.add_argument(
        "--unregister-models-user",
        action="store_true",
        help="Remove Clarity from the shared models folder",
    )
    parser.add_argument("--install-suite", action="store_true", help="Install or repair the shared suite dependencies")
    parser.add_argument("--out", type=Path, default=None, help="Write the models-folder report here")
    parser.epilog = "Headless commands: restore, project, info, version, models, selftest. See docs/CLI.md."
    return parser


def _common() -> _Parser:
    parser = _Parser(add_help=False)
    parser.add_argument("--json", action="store_true", help="Write result and progress JSON lines to stdout")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan. Do not use the GPU")
    return parser


def _command_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = _Parser(prog="clarity", description=APP_NAME)
    sub = parser.add_subparsers(dest="command", required=True)
    restore = sub.add_parser("restore", parents=[common], help="Restore one video")
    restore.add_argument("--input", required=True, type=Path, help="Source video")
    restore.add_argument("--output", required=True, type=Path, help="Destination video")
    restore.add_argument("--mode", choices=("clean", "strong"), default="clean", help="clean or strong")
    restore.add_argument("--upscale", action="store_true", help="Optional 2x. Off unless this flag is set")
    restore.add_argument("--format", default="mkv", help="mkv, mp4, av1, or hevc")
    restore.add_argument("--codec", choices=("av1", "hevc"), default=None, help="av1 or hevc, when set separately from --format")
    restore.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    restore.add_argument("--ffprobe", default="ffprobe", help="ffprobe binary")
    project = sub.add_parser("project", help="Open or resume a .vxclarity project")
    project_sub = project.add_subparsers(dest="action", required=True)
    opened = project_sub.add_parser("open", parents=[common], help="Read a project and print it")
    opened.add_argument("path", type=Path, help="Path of the .vxclarity file")
    resume = project_sub.add_parser("resume", parents=[common], help="Continue a project from its journal")
    resume.add_argument("path", type=Path, help="Path of the .vxclarity file")
    resume.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    resume.add_argument("--ffprobe", default="ffprobe", help="ffprobe binary")
    sub.add_parser("info", parents=[common], help="Print the version from BUILD.json")
    sub.add_parser("version", parents=[common], help="Print the version from BUILD.json")
    models = sub.add_parser("models", help="List or fetch catalogue weights")
    models_sub = models.add_subparsers(dest="action", required=True)
    models_sub.add_parser("list", parents=[common], help="List the catalogue")
    fetch = models_sub.add_parser("fetch", parents=[common], help="Fetch weights into the shared models folder")
    fetch.add_argument("--id", action="append", default=None, dest="ids", help="Catalogue id to fetch. Repeat for several")
    sub.add_parser("selftest", parents=[common], help="Open the window offscreen, or describe that check")
    return parser


def _settings(mode: str, upscale: bool, format_name: str, codec_name: str | None) -> Settings:
    container = Container.MKV
    codec = VideoCodec.AV1
    token = format_name.strip().lower()
    if token in {"mkv", "matroska"}:
        container = Container.MKV
    elif token == "mp4":
        container = Container.MP4
    elif token == "av1":
        codec = VideoCodec.AV1
    elif token == "hevc":
        codec = VideoCodec.HEVC
    else:
        raise ClarityError("format must be mkv, mp4, av1, or hevc")
    if codec_name == "av1":
        codec = VideoCodec.AV1
    elif codec_name == "hevc":
        codec = VideoCodec.HEVC
    elif codec_name is not None:
        raise ClarityError("codec must be av1 or hevc")
    if mode not in {"clean", "strong"}:
        raise ClarityError("mode must be clean or strong")
    return Settings(mode=mode, upscale=upscale, container=container, codec=codec)


def _spec(source: Path, destination: Path, settings: Settings, ffmpeg: str, ffprobe: str, work: Path | None) -> JobSpec:
    return JobSpec(
        source=source,
        destination=destination,
        settings=settings,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        work=work,
    )


def _project_payload(path: Path) -> dict[str, Any]:
    project = load_project(path)
    finished = project.resume.get("journal")
    count = len(finished) if isinstance(finished, list) else 0
    return {
        "command": "project",
        "path": str(project.path),
        "source": str(project.source) if project.source is not None else "",
        "destination": str(project.destination) if project.destination is not None else "",
        "mode": project.settings.mode,
        "upscale": project.settings.upscale,
        "container": project.settings.container.value,
        "codec": project.settings.codec.value,
        "finished_chunks": count,
    }


def _info_payload() -> dict[str, Any]:
    return {
        "command": "info",
        "name": APP_NAME,
        "version": APP_VERSION,
        "build": APP_BUILD,
        "codename": CODENAME,
        "label": version_label(),
    }


def _restore(args: argparse.Namespace, console: Console) -> int:
    try:
        settings = _settings(str(args.mode), bool(args.upscale), str(args.format), args.codec)
        spec = _spec(Path(args.input), Path(args.output), settings, str(args.ffmpeg), str(args.ffprobe), None)
    except ClarityError as exc:
        return console.fail(str(exc), ExitCode.USAGE)
    if args.dry_run:
        try:
            plan = dry_run_restore(spec)
        except ClarityError as exc:
            code = ExitCode.ENVIRONMENT if "failed to start" in str(exc) else ExitCode.USAGE
            return console.fail(str(exc), code)
        return console.ok({"command": "restore", "dry_run": True, "plan": plan})
    gate = _gates(console)
    if gate is not None:
        return gate
    try:
        run_job(spec, progress=console.progress)
    except Cancelled as exc:
        return console.fail(str(exc), ExitCode.USAGE)
    except ClarityError as exc:
        return console.fail(str(exc), ExitCode.JOB)
    return console.ok({"command": "restore", "dry_run": False, "destination": str(spec.destination)})


def _gates(console: Console) -> int | None:
    platform = assess_platform()
    if not platform.ok:
        return console.fail(platform.message, ExitCode.ENVIRONMENT)
    try:
        gate = assess_launch()
    except EngineUnavailable as exc:
        return console.fail(str(exc), ExitCode.ENVIRONMENT)
    if not gate.ok:
        return console.fail(gate.message, ExitCode.ENVIRONMENT)
    if gate.warning:
        print(gate.warning, file=sys.stderr)
    return None


def _project_action(value: object) -> ProjectAction:
    if value == "open" or value == "resume":
        return value
    raise ClarityError(f"unknown project action {value}")


def _models_action(value: object) -> ModelsAction:
    if value == "list" or value == "fetch":
        return value
    raise ClarityError(f"unknown models action {value}")


def _command_name(value: object) -> CommandName:
    if value == "restore":
        return "restore"
    if value == "project":
        return "project"
    if value == "info":
        return "info"
    if value == "version":
        return "version"
    if value == "models":
        return "models"
    if value == "selftest":
        return "selftest"
    raise ClarityError(f"unknown command {value}")


def _project_command(args: argparse.Namespace, console: Console) -> int:
    try:
        action = _project_action(args.action)
    except ClarityError as exc:
        return console.fail(str(exc), ExitCode.USAGE)
    path = Path(args.path)
    if action == "open":
        try:
            payload = _project_payload(path)
        except ProjectFormatError as exc:
            return console.fail(str(exc), ExitCode.USAGE)
        payload["action"] = "open"
        payload["dry_run"] = bool(args.dry_run)
        return console.ok(payload)
    if action == "resume":
        try:
            project = load_project(path)
        except ProjectFormatError as exc:
            return console.fail(str(exc), ExitCode.USAGE)
        if project.source is None or project.destination is None:
            return console.fail("the project has no source or destination", ExitCode.USAGE)
        spec = _spec(
            project.source,
            project.destination,
            project.settings,
            str(getattr(args, "ffmpeg", "ffmpeg")),
            str(getattr(args, "ffprobe", "ffprobe")),
            work_dir_for(project.path, project.resume),
        )
        if args.dry_run:
            try:
                plan = dry_run_restore(spec)
            except ClarityError as exc:
                code = ExitCode.ENVIRONMENT if "failed to start" in str(exc) else ExitCode.USAGE
                return console.fail(str(exc), code)
            return console.ok({"command": "project", "action": "resume", "dry_run": True, "plan": plan})
        blocked = _gates(console)
        if blocked is not None:
            return blocked
        try:
            run_job(spec, progress=console.progress)
        except Cancelled as exc:
            return console.fail(str(exc), ExitCode.USAGE)
        except ClarityError as exc:
            return console.fail(str(exc), ExitCode.JOB)
        return console.ok(
            {"command": "project", "action": "resume", "dry_run": False, "destination": str(spec.destination)}
        )
    unknown: Never = action
    return console.fail(f"unknown project action {unknown}", ExitCode.USAGE)


def _models_command(args: argparse.Namespace, console: Console) -> int:
    try:
        action = _models_action(args.action)
    except ClarityError as exc:
        return console.fail(str(exc), ExitCode.USAGE)
    if action == "list":
        return console.ok({"command": "models", "action": "list", "dry_run": bool(args.dry_run), "models": catalogue_rows()})
    if action == "fetch":
        ids = list(args.ids) if args.ids else None
        try:
            rows = fetch_weights(ids, dry_run=bool(args.dry_run))
        except KeyError as exc:
            return console.fail(f"unknown model {exc}", ExitCode.USAGE)
        except EngineUnavailable as exc:
            return console.fail(str(exc), ExitCode.ENVIRONMENT)
        except ClarityError as exc:
            return console.fail(str(exc), ExitCode.JOB)
        return console.ok({"command": "models", "action": "fetch", "dry_run": bool(args.dry_run), "models": rows})
    unknown: Never = action
    return console.fail(f"unknown models action {unknown}", ExitCode.USAGE)


def _selftest_window() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    if window.start.text().strip() == "":
        print("the Start button has no label", file=sys.stderr)
        return int(ExitCode.USAGE)
    return int(ExitCode.OK)


def _selftest_command(args: argparse.Namespace, console: Console) -> int:
    if args.dry_run:
        return console.ok(
            {
                "command": "selftest",
                "dry_run": True,
                "checks": ["suite dependencies", "offscreen window", "start label"],
                "gpu_used": False,
            }
        )
    try:
        ensure_suite(console.progress)
        code = _selftest_window()
    except (ImportError, ClarityError, OSError) as exc:
        return console.fail(str(exc), ExitCode.USAGE)
    if code != 0:
        return console.fail("the Start button has no label", ExitCode.USAGE)
    return console.ok({"command": "selftest", "dry_run": False, "message": "selftest ok"})


def _dispatch(command: CommandName, args: argparse.Namespace) -> int:
    console = Console(bool(args.json))
    if command == "restore":
        return _restore(args, console)
    if command == "project":
        return _project_command(args, console)
    if command == "info" or command == "version":
        payload = _info_payload()
        payload["command"] = command
        payload["dry_run"] = bool(args.dry_run)
        return console.ok(payload)
    if command == "models":
        return _models_command(args, console)
    if command == "selftest":
        return _selftest_command(args, console)
    unknown: Never = command
    return console.fail(f"unknown command {unknown}", ExitCode.USAGE)


def _command_main(argv: list[str]) -> int:
    try:
        args = _command_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code in {None, 0}:
            return 0
        if isinstance(code, int):
            return code
        return int(ExitCode.USAGE)
    try:
        command = _command_name(args.command)
    except ClarityError as exc:
        print(exc, file=sys.stderr)
        return int(ExitCode.USAGE)
    return _dispatch(command, args)


def _gui(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(version_label())
        return 0
    if args.install_suite:
        return _install_suite()
    if args.register_models_user or args.unregister_models_user:
        return _register(args.out, remove=bool(args.unregister_models_user))
    if args.selftest:
        try:
            ensure_suite(lambda fraction, message: print(message, file=sys.stderr))
            code = _selftest_window()
        except (ImportError, ClarityError, OSError) as exc:
            print(exc)
            return 1
        if code != 0:
            return code
        print("selftest ok")
        return 0
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
    try:
        ensure_suite(RepairProgress())
    except SuiteMissing as exc:
        print(exc)
        return int(ExitCode.ENVIRONMENT)
    window = MainWindow()
    if args.project is not None:
        window.open_project(args.project)
    window.show()
    return int(application.exec())


def _install_suite() -> int:
    """Install or repair the shared dependencies and print each step.

    Returns:
        ``0`` when every component is ready, ``2`` when a repair fails.
    """
    try:
        ensure_suite(lambda _fraction, message: print(message, file=sys.stderr))
    except (SuiteMissing, OSError) as exc:
        print(exc)
        return int(ExitCode.ENVIRONMENT)
    return int(ExitCode.OK)


def _register(out: Path | None, *, remove: bool) -> int:
    if remove:
        release_app()
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
        return int(ExitCode.ENVIRONMENT)
    if out is not None:
        out.write_text(f"{len(others)}\n{folder}\n", encoding="utf-8")
    print("\n".join(others) if others else "clarity")
    return int(ExitCode.OK)


def main(argv: list[str] | None = None) -> int:
    """Dispatch a headless command, or open the window.

    A leading ``--json`` is kept for the window launcher and moved to the end
    of a headless command so argparse still sees the subcommand first.

    Args:
        argv: Arguments without the program name. Defaults to ``sys.argv``.

    Returns:
        A process status from ``ExitCode``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    json_first = False
    while args and args[0] == "--json":
        json_first = True
        args = args[1:]
    if args and args[0] in _COMMANDS:
        if json_first and "--json" not in args:
            args = [*args, "--json"]
        return _command_main(args)
    if json_first:
        args = ["--json", *args]
    return _gui(args)
