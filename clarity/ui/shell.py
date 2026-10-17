"""Clarity's product window on the shared video shell.

The video tab is ``SimpleWidget`` on ``surface_for("clarity")``, including the
before/after wipe. Project and Setup keep the rest of the old window and the
CLI actions. ``reference_shell`` opens ``ReferenceShell(app="clarity")``.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from voxprint_common.i18n import add_catalogs, set_language, tr
from voxprint_common.mdl import CLI_NAME
from voxprint_common.mdl.catalog import CatalogError
from voxprint_common.paths import state_dir as suite_state_dir
from voxprint_common.suite_settings import set_ui_language
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.presets import SettingState
from voxprint_common.ui.shell import ReferenceShell
from voxprint_common.ui.safety import ReviewedOutput
from voxprint_common.ui.simple_image import GalleryItem, Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.theme import ACCENT, build_shell_style

from clarity.appinfo import APP_NAME, APP_VERSION, CODENAME, build_info, repo_root, version_label
from clarity.engine import (
    OWNER,
    EngineUnavailable,
    ensure_shared_runtime,
    models_directory,
    register_user,
    release_shared_runtime,
    unregister_user,
)
from clarity.errors import Cancelled, ClarityError
from clarity.ffmpeg_setup import FfmpegMissing, ensure_ffmpeg
from clarity.i18n import clear_cache, current_language
from clarity.models import catalogue_ids, catalogue_rows, fetch_weights
from clarity.pipeline.job import Settings as JobSettings
from clarity.pipeline.job import models_for
from clarity.pipeline.runner import CancelToken, JobSpec, dry_run_restore, run_job
from clarity.progress import progress_stage
from clarity.project import (
    Project,
    ProjectFormatError,
    load_project,
    load_recent,
    remember_recent,
    save_project,
    sync_resume,
    video_fingerprint,
    work_dir_for as project_work_dir,
)
from clarity.ui.acrylic import enable_acrylic
from clarity.ui.job_presets import (
    apply_job,
    clarity_settings,
    job_from_state,
    load_clarity_presets,
    models_ready,
    output_suffix,
)
from clarity.ui.job_presets import adopted_video as load_adopted_video
from clarity.ui.shell_dress import install as install_dress
from clarity.ui.shell_dress import run_graphics_check
from clarity.ui.shell_pages import ProjectPage, SetupPage
from clarity.ui.suite_dialog import RepairProgress
from clarity.ui.theme import CONTROL_BG, CONTROL_BORDER, IDENTITY, TEXT_MUTED, TEXT_ON_PRIMARY

Runner = Callable[..., None]


class _PathGate:
    """Allow a restore when the prompt is a file path."""

    def review_prompt(self, prompt: str) -> str | None:
        """Refuse a path that is not a file.

        Args:
            prompt: The dropped or typed path.

        Returns:
            A plain reason, or None when the file is there.
        """
        if Path(prompt.strip()).is_file():
            return None
        return tr("error.missing_source")

    def review_result(self, prompt: str, item_id: str, output: ReviewedOutput) -> str | None:
        """Keep the finished restore.

        Args:
            prompt: The source path.
            item_id: The history id.
            output: The restored file.

        Returns:
            None.
        """
        return None


class _RestoreSignals(QObject):
    """Cross-thread progress for one restore."""

    progressed = Signal(float, str)
    finished = Signal(bool, str)


class _RestoreThread(QThread):
    """Run ``run_job`` off the UI thread."""

    def __init__(
        self,
        parent: QObject,
        spec: JobSpec,
        token: CancelToken,
        runner: Runner,
        on_chunk: Callable[[Path], None],
    ) -> None:
        """Start idle.

        Args:
            parent: The window.
            spec: The restore to run.
            token: Cancel flag shared with the Cancel button.
            runner: ``run_job`` or a test double.
            on_chunk: Called with the work folder after a chunk is stored.
        """
        super().__init__(parent)
        self._spec = spec
        self._token = token
        self._runner = runner
        self._on_chunk = on_chunk
        self.signals = _RestoreSignals()

    def run(self) -> None:
        """Run the job and emit the outcome."""

        def report(fraction: float, text: str) -> None:
            if self._token.is_set():
                return
            self.signals.progressed.emit(fraction, text)

        try:
            self._runner(self._spec, token=self._token, progress=report, on_chunk=self._on_chunk)
        except Cancelled:
            self.signals.finished.emit(False, "cancelled")
        except Exception as exc:
            self.signals.finished.emit(False, str(exc))
        else:
            self.signals.finished.emit(True, "")


class _KeepPath(QObject):
    """Keep a dropped path whole. A click opens the file dialog.

    The shell drop keeps only the file name. This filter is installed after
    that one, so it runs first.
    """

    def __init__(self, window: MainWindow, label: QWidget) -> None:
        """Watch ``label``.

        Args:
            window: Receives the path.
            label: The file drop area.
        """
        super().__init__(label)
        self._window = window

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Handle a drop or a click on the file area.

        Args:
            watched: The drop label.
            event: The Qt event.

        Returns:
            True when this filter accepted the event.
        """
        kind = event.type()
        if kind == QEvent.Type.Drop:
            urls = event.mimeData().urls()  # type: ignore[attr-defined]
            if not urls:
                return False
            raw = urls[0].toLocalFile() or urls[0].toString()
            if raw:
                self._window.use_source_text(raw)
            event.acceptProposedAction()  # type: ignore[attr-defined]
            return True
        if kind == QEvent.Type.MouseButtonRelease:
            self._window.browse_source()
            return True
        return False


def hold_float_precision(root: QWidget, state: SettingState) -> None:
    """Let expert number boxes keep values such as 0.35.

    The shell builds those boxes with one decimal place. Two places are applied
    after the panel exists, without writing a new value through the signal.

    Args:
        root: The simple screen.
        state: Live values to copy back into the boxes.
    """
    for box in root.findChildren(QDoubleSpinBox):
        key = box.objectName()
        box.blockSignals(True)
        box.setDecimals(2)
        box.setSingleStep(0.05)
        if state.has(key):
            value = state.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                box.setValue(float(value))
        box.blockSignals(False)


def shell_style() -> str:
    """Return the shared shell stylesheet with Clarity's teal on the actions.

    The shared sheet paints primary buttons and selected chips yellow. The
    rules appended here paint those same controls in Clarity's teal. A
    disabled Restore button stays muted.

    Returns:
        A Qt stylesheet.
    """
    sheet = build_shell_style(False).replace(ACCENT, IDENTITY)
    teal = IDENTITY
    ink = TEXT_ON_PRIMARY
    muted = f"background: {CONTROL_BG}; color: {TEXT_MUTED}; border-color: {CONTROL_BORDER};"
    override = (
        f"QPushButton#generate {{ background: {teal}; color: {ink}; border-color: {teal}; }}\n"
        f"QPushButton#primary {{ background: {teal}; color: {ink}; border-color: {teal}; }}\n"
        f"QPushButton#expertToggle:checked {{ background: {teal}; color: {ink}; border-color: {teal}; }}\n"
        f'QPushButton[chip="true"]:checked {{ background: {teal}; color: {ink}; border-color: {teal}; }}\n'
        f"QPushButton#generate:disabled {{ {muted} }}\n"
        f'QPushButton[chip="true"]:disabled {{ {muted} }}\n'
        f"QCheckBox::indicator:checked {{ background: {teal}; border-color: {teal}; }}\n"
        f'QPushButton[thumb="true"]:checked {{ border: 2px solid {teal}; }}\n'
        'QPushButton[chip="true"] { min-height: 24px; max-height: 26px; padding: 0 10px; }\n'
        "QLabel#fileDrop { padding: 8px 16px; }\n"
        "QLabel#settingGroup { padding: 4px 0 0 0; }\n"
        "QFrame#expertPanel QSpinBox, QFrame#expertPanel QDoubleSpinBox,\n"
        "QFrame#expertPanel QComboBox { min-height: 22px; padding: 0 6px; }\n"
    )
    return f"{sheet}\n{override}"


def reference_shell(*, state_dir: Path | None = None) -> ReferenceShell:
    """Open the shared reference window for the clarity surface.

    The product window is :class:`MainWindow`. This helper is the adoption
    entry ``ReferenceShell(app="clarity")``. It loads the shared video presets.

    Args:
        state_dir: Directory for that window's sample state.

    Returns:
        The reference window. The caller shows it.
    """
    add_catalogs(repo_root() / "locales")
    return ReferenceShell(app="clarity", state_dir=state_dir)


def selftest_plan() -> dict[str, object]:
    """Return the plan ``python -m clarity selftest`` builds.

    ``clarity.cli`` imports this window, so this import stays inside the
    function. A top-level import would cycle.

    Returns:
        The encoder, pixel format, and chunk count of the default plan.
    """
    from clarity.cli import _selftest_plan as plan

    return plan()


class MainWindow(QMainWindow):
    """The restore window: the shared video screen, plus project and setup."""

    def __init__(self, runner: Runner | None = None) -> None:
        """Build the window.

        Args:
            runner: Callable with the same signature as ``run_job``. Tests pass
                a double. The default is ``run_job``.
        """
        super().__init__()
        self._runner = run_job if runner is None else runner
        self._thread: _RestoreThread | None = None
        self._token: CancelToken | None = None
        self._closing = False
        self._project_path: Path | None = None
        self._project: Project | None = None
        self._output_text = ""
        self._ffmpeg = "ffmpeg"
        self._ffprobe = "ffprobe"
        self._report = ""
        self._selected_ids: list[str] = []
        self._runs = 0
        self._seeded = False
        self._last_phase: Phase | None = None
        self._screen: SimpleWidget | None = None
        self._project_page: ProjectPage | None = None
        self._setup_page: SetupPage | None = None
        self._tabs: QTabWidget | None = None
        self._drop_filter: _KeepPath | None = None
        add_catalogs(repo_root() / "locales")
        self._language = current_language()
        set_language(self._language)
        adopted = load_adopted_video()
        self._settings = clarity_settings()
        self._layout = SimpleLayout(
            presets=load_clarity_presets(self._settings),
            settings=self._settings,
            state_dir=suite_state_dir() / adopted.surface.app_id,
            gate=_PathGate(),
            state_name=adopted.settings_name,
            history_name=adopted.history_name,
        )
        self._expert = ExpertMode()
        self._stop_expert = self._expert.subscribe(self._on_expert_mode)
        self.start: QPushButton
        self.setObjectName("shell")
        self.setStyleSheet(shell_style())
        self.setMinimumSize(adopted.surface.min_width, adopted.surface.min_height)
        self.resize(1200, 900)
        self._assemble()

    def showEvent(self, event: object) -> None:  # noqa: N802
        """Apply the acrylic backdrop once the native window exists.

        Args:
            event: Qt show event.
        """
        super().showEvent(event)  # type: ignore[arg-type]
        enable_acrylic(int(self.winId()))

    def closeEvent(self, event: object) -> None:  # noqa: N802
        """Cancel the job and wait for the worker before closing.

        Args:
            event: Qt close event.
        """
        self._closing = True
        self._stop_expert()
        token = self._token
        if token is not None:
            token.cancel()
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.wait(3000)
        super().closeEvent(event)  # type: ignore[arg-type]

    def open_project(self, path: Path) -> None:
        """Load a ``.vxclarity`` file into the form.

        Args:
            path: Project file. A format error is shown in a dialog.
        """
        try:
            project = load_project(path)
        except ProjectFormatError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._project = project
        self._project_path = project.path
        if project.source is not None:
            text = str(project.source)
        else:
            video = video_fingerprint(project.manifest)
            text = str(video.get("absolute") or "")
        if self._screen is not None:
            self._screen.set_prompt_text(text)
        self._output_text = "" if project.destination is None else str(project.destination)
        apply_job(self.settings_state, project.settings)
        self._remember(project.path)
        self._sync_pages()
        if self._screen is not None:
            self._screen.refresh(prompt=True)
            if project.resume.get("journal"):
                self._screen.set_note(tr("status.resumed"))
            else:
                self._publish_output()

    def open_project_path(self, text: str) -> None:
        """Open the project at ``text``.

        Args:
            text: A ``.vxclarity`` path from the recent list.
        """
        self.open_project(Path(text))

    def save_project(self, path: Path, *, retarget: bool = True) -> None:
        """Write the current form to a ``.vxclarity`` file.

        Args:
            path: Destination project path.
            retarget: When true, later saves use ``path``. Save a copy passes
                false.
        """
        source = self._source_path()
        if source is None or not source.is_file():
            QMessageBox.warning(self, APP_NAME, tr("error.missing_source"))
            return
        destination = self._destination()
        self._output_text = str(destination)
        try:
            project = save_project(
                path,
                source=source,
                settings=self.current_settings(),
                destination=destination,
                resume=self._project.resume if self._project is not None else {},
                previous=self._project,
            )
        except ProjectFormatError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if retarget:
            self._project = project
            self._project_path = project.path
        self._remember(project.path)
        self._sync_pages()
        if self._screen is not None:
            self._screen.set_note(tr("status.saved"))

    @property
    def settings_state(self) -> SettingState:
        """Return the shared settings state."""
        state = self._layout.state
        if state is None:
            raise RuntimeError("the video screen has no settings state")
        return state

    def current_settings(self) -> JobSettings:
        """Return the restore settings shown in the window."""
        return job_from_state(self.settings_state.values)

    def source_text(self) -> str:
        """Return the source path currently shown."""
        return self._layout.prompt

    def output_text(self) -> str:
        """Return the output path, or an empty string when it is still the default."""
        return self._output_text

    def set_output_text(self, text: str) -> None:
        """Store the output path typed on the Project tab.

        Args:
            text: The path, or an empty string to use the default suffix again.
        """
        self._output_text = text.strip()
        self._publish_output()

    def use_source_text(self, text: str) -> None:
        """Use ``text`` as the source video, or open it when it is a project.

        Args:
            text: A path from the drop area, the dialog, or the Project field.
        """
        cleaned = text.strip()
        path = Path(cleaned) if cleaned else None
        if path is not None and path.suffix.lower() == ".vxclarity" and path.is_file():
            self.open_project(path)
            return
        if self._screen is not None:
            self._screen.set_prompt_text(cleaned)
        if path is not None and cleaned and not self._output_text:
            self._output_text = str(path.with_suffix(self._suffix()))
        self._sync_pages()
        self._publish_output()

    def browse_source(self) -> None:
        """Ask for a source video."""
        path, _selected = QFileDialog.getOpenFileName(self, tr("source.browse"), "", tr("source.filter"))
        if path:
            self.use_source_text(path)

    def browse_output(self) -> None:
        """Ask for an output video path. This is Export."""
        current = self._output_text or (str(self._destination()) if self._layout.prompt.strip() else "")
        path, _selected = QFileDialog.getSaveFileName(self, tr("project.export"), current, tr("source.filter"))
        if path:
            self.set_output_text(path)
            self._sync_pages()

    def browse_project(self) -> None:
        """Ask for a ``.vxclarity`` file and open it."""
        path, _selected = QFileDialog.getOpenFileName(self, tr("project.open"), "", tr("project.filter"))
        if path:
            self.open_project(Path(path))

    def save_current(self) -> None:
        """Save the open project, or ask for a path when there is none yet."""
        if self._project_path is not None:
            self.save_project(self._project_path)
            return
        self.browse_save()

    def browse_save(self) -> None:
        """Ask for a new project path and save into it."""
        suggested = str(self._project_path or "")
        path, _selected = QFileDialog.getSaveFileName(self, tr("project.save_as"), suggested, tr("project.filter"))
        if path:
            self.save_project(Path(path))

    def browse_copy(self) -> None:
        """Save a copy. The project that is already open stays current."""
        suggested = str(self._project_path or "")
        path, _selected = QFileDialog.getSaveFileName(self, tr("project.save_copy"), suggested, tr("project.filter"))
        if path:
            self.save_project(Path(path), retarget=False)

    def download_models(self) -> None:
        """Fetch the weights for the settings on screen."""
        try:
            ids = [model.id for model in models_for(self.current_settings())]
            fetch_weights(ids, dry_run=False)
        except (EngineUnavailable, ClarityError, KeyError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if self._screen is not None:
            self._screen.set_note(tr("status.downloaded"))

    def recent_paths(self) -> list[str]:
        """Return recent project paths, newest first."""
        return load_recent()

    def codec_choices(self) -> tuple[tuple[str, str], ...]:
        """Return codec values and labels for the Setup combo."""
        spec = self._settings.get("job.codec")
        if spec is None:
            return ()
        return tuple((choice.value, tr(choice.label_key)) for choice in spec.choices)

    def codec_value(self) -> str:
        """Return the live codec value."""
        value = self.settings_state.get("job.codec")
        return value if isinstance(value, str) else "av1"

    def set_codec(self, value: str) -> None:
        """Set the codec from the Setup combo or the Expert panel.

        Args:
            value: ``av1`` or ``hevc``.
        """
        if value == self.codec_value():
            return
        self.settings_state.set_value("job.codec", value)
        self._refresh_screen()

    def sharpen_percent(self) -> int:
        """Return the sharpen slider position, from 0 through 100."""
        value = self.settings_state.get("job.sharpen")
        number = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.35
        return max(0, min(100, int(round(number * 100))))

    def set_sharpen_percent(self, percent: int) -> None:
        """Set sharpen from the slider.

        Args:
            percent: 0 through 100. 35 is the default 0.35.
        """
        self.settings_state.set_value("job.sharpen", max(0, min(100, int(percent))) / 100)
        self._refresh_screen()

    def anime_checked(self) -> bool:
        """Return whether the Look chip is Anime."""
        return self.settings_state.get("job.anime") is True

    def set_anime(self, checked: bool) -> None:
        """Select the Look chip that matches ``checked``.

        Args:
            checked: True selects Anime.
        """
        if checked == self.anime_checked():
            return
        self._layout.choose("look", "anime" if checked else "normal")
        self._refresh_screen()

    def upscale_enabled(self) -> bool:
        """Return whether Size is twice as big."""
        return self.settings_state.get("job.upscale") is True

    def ffmpeg_text(self) -> str:
        """Return the ffmpeg binary name or path."""
        return self._ffmpeg

    def ffprobe_text(self) -> str:
        """Return the ffprobe binary name or path."""
        return self._ffprobe

    def report_text(self) -> str:
        """Return the optional models-folder report path."""
        return self._report

    def set_tool_paths(self, ffmpeg: str, ffprobe: str, report: str) -> None:
        """Store the CLI binary paths and the report path.

        Args:
            ffmpeg: ffmpeg binary. Blank becomes ``ffmpeg``.
            ffprobe: ffprobe binary. Blank becomes ``ffprobe``.
            report: Report file for register and unregister. Blank skips the file.
        """
        self._ffmpeg = ffmpeg.strip() or "ffmpeg"
        self._ffprobe = ffprobe.strip() or "ffprobe"
        self._report = report.strip()

    def selected_model_ids(self) -> list[str]:
        """Return the model ids checked on Setup."""
        return list(self._selected_ids)

    def set_selected_model_ids(self, model_ids: list[str]) -> None:
        """Remember which catalogue rows are checked.

        Args:
            model_ids: Ids from the catalogue. An empty list means fetch all.
        """
        self._selected_ids = list(model_ids)

    def set_language_code(self, code: str) -> None:
        """Switch English or Russian and rebuild the labels.

        Args:
            code: ``en`` or ``ru``.
        """
        if code == self._language or code not in {"en", "ru"}:
            return
        if self._layout.phase is Phase.BUSY:
            return
        self._language = code
        set_language(code)
        clear_cache()
        try:
            set_ui_language(code)
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        QTimer.singleShot(0, self._assemble)

    def repair_runtime(self) -> None:
        """Repair the shared runtime and ffmpeg. A healthy check stays quiet."""
        self._run_repair(fetch=False)

    def install_suite(self) -> None:
        """Repair the runtime, ffmpeg, and every catalogue weight."""
        self._run_repair(fetch=True)

    def manage_models(self) -> None:
        """Start Manage models for Clarity."""
        try:
            subprocess.Popen([CLI_NAME, "--gui", "--app", "clarity"])
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))

    def list_models(self) -> None:
        """Show the catalogue. Nothing is downloaded."""
        self._show_text(tr("clarity.action.list_models"), json.dumps(catalogue_rows(), indent=2))

    def dry_run_restore(self) -> None:
        """Build the restore plan without the GPU."""
        source = self._source_path()
        if source is None or not source.is_file():
            QMessageBox.warning(self, APP_NAME, tr("error.missing_source"))
            return
        spec = self._job_spec(source)
        try:
            plan = dry_run_restore(spec)
        except (ClarityError, OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._show_text(tr("clarity.action.dry_run"), json.dumps(plan, indent=2, default=str))

    def fetch_selected(self) -> None:
        """Fetch the checked models, or the whole catalogue when none are checked."""
        self._fetch(dry_run=False)

    def fetch_selected_dry_run(self) -> None:
        """List the files a fetch would download."""
        self._fetch(dry_run=True)

    def register_models(self) -> None:
        """Register Clarity in the shared models folder."""
        self._register(remove=False)

    def unregister_models(self) -> None:
        """Release the shared runtime and remove Clarity from the models folder."""
        self._register(remove=True)

    def check_graphics_card(self) -> None:
        """Run the GPU selftest in a subprocess and show pass or fail."""
        run_graphics_check(self)

    def plan_check(self) -> None:
        """Show the self-test plan. It does not open a CUDA session."""
        try:
            plan = selftest_plan()
        except (ClarityError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._show_text(tr("clarity.action.plan_check"), json.dumps(plan, indent=2))

    def show_version(self) -> None:
        """Show the build label and its JSON form."""
        identity = build_info()
        payload = {
            "command": "info",
            "name": APP_NAME,
            "version": APP_VERSION,
            "build": identity.build,
            "codename": CODENAME,
            "label": version_label(),
        }
        self._show_text(version_label(), json.dumps(payload, indent=2))

    def _assemble(self) -> None:
        screen = self._make_screen()
        project = ProjectPage(self)
        setup = SetupPage(self)
        tabs = QTabWidget()
        tabs.setObjectName("clarityTabs")
        video = QWidget()
        video_layout = QVBoxLayout(video)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(screen.root)
        tabs.addTab(video, tr("clarity.tab.video"))
        tabs.addTab(project, tr("clarity.tab.project"))
        tabs.addTab(setup, tr("clarity.tab.setup"))
        self._screen = screen
        self._project_page = project
        self._setup_page = setup
        self._tabs = tabs
        setup.select_language(self._language)
        chrome = QWidget()
        column = QVBoxLayout(chrome)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        subtitle = QLabel(tr("app.subtitle"))
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("padding: 12px 16px 8px 16px; background: transparent;")
        column.addWidget(subtitle)
        column.addWidget(tabs, 1)
        footer = QLabel(version_label())
        footer.setObjectName("versionLabel")
        footer.setStyleSheet("padding: 8px 16px 12px 16px; background: transparent;")
        column.addWidget(footer)
        self.setCentralWidget(chrome)
        self.setWindowTitle(tr("splash.app.clarity"))
        self._rebuild_menus()
        self._seed_recent()
        screen.refresh()
        self._sync_pages()
        self._publish_output()

    def _make_screen(self) -> SimpleWidget:
        adopted = load_adopted_video()
        screen = SimpleWidget(
            self._layout,
            self._settings,
            on_generate=self._on_generate,
            on_command=self._on_command,
            on_history=self._on_history,
            on_refresh=self._on_refresh,
            surface=adopted.surface,
        )
        original = screen.set_expert

        def set_expert(enabled: bool) -> None:
            original(enabled)
            if enabled:
                hold_float_precision(screen.root, self.settings_state)

        screen.set_expert = set_expert  # type: ignore[method-assign]
        install_dress(screen, self._layout)
        drop = screen.root.findChild(QLabel, "fileDrop")
        if drop is not None:
            watcher = _KeepPath(self, drop)
            drop.installEventFilter(watcher)
            self._drop_filter = watcher
        cancel = screen.root.findChild(QPushButton, "cancel")
        if cancel is not None:
            cancel.clicked.connect(self._cancel_job)
        button = screen.root.findChild(QPushButton, "generate")
        if button is None:
            raise RuntimeError("the shell did not build a Restore button")
        self.start = button
        if self._expert.enabled():
            screen.set_expert(True)
        return screen

    def _rebuild_menus(self) -> None:
        bar = self.menuBar()
        bar.clear()
        file_menu = bar.addMenu(tr("shell.menu.file"))
        file_menu.addAction(tr("project.open"), self.browse_project)
        file_menu.addAction(tr("project.save"), self.save_current)
        file_menu.addAction(tr("project.save_as"), self.browse_save)
        file_menu.addAction(tr("project.save_copy"), self.browse_copy)
        file_menu.addAction(tr("project.export"), self.browse_output)
        models = bar.addMenu(tr("clarity.menu.models"))
        models.addAction(tr("clarity.action.manage_models"), self.manage_models)
        models.addAction(tr("clarity.action.list_models"), self.list_models)
        models.addAction(tr("project.download"), self.download_models)
        models.addAction(tr("clarity.action.fetch"), self.fetch_selected)
        models.addAction(tr("clarity.action.fetch_dry_run"), self.fetch_selected_dry_run)
        models.addAction(tr("clarity.action.register"), self.register_models)
        models.addAction(tr("clarity.action.unregister"), self.unregister_models)
        tools = bar.addMenu(tr("clarity.menu.tools"))
        tools.addAction(tr("clarity.action.repair"), self.repair_runtime)
        tools.addAction(tr("clarity.action.install_suite"), self.install_suite)
        tools.addAction(tr("clarity.action.dry_run"), self.dry_run_restore)
        tools.addAction(tr("clarity.action.plan_check"), self.plan_check)
        tools.addAction(tr("clarity.action.check_gpu"), self.check_graphics_card)
        tools.addAction(version_label(), self.show_version)

    def _on_expert_mode(self, enabled: bool) -> None:
        if self._screen is not None:
            self._screen.set_expert(enabled)

    def _on_command(self, command_id: str) -> None:
        if command_id == "view.toggle_expert_mode":
            self._expert.toggle()
            return
        if command_id == "models.manage":
            self.manage_models()
            return
        if command_id == "file.save":
            self.save_current()
            return
        if command_id == "file.open_folder":
            self._open_output_folder()
            return
        if command_id == "view.settings":
            tabs = self._tabs
            setup = self._setup_page
            if tabs is not None and setup is not None:
                tabs.setCurrentWidget(setup)

    def _on_history(self, action: str, item_id: str) -> None:
        if action != "open":
            return
        item = self._layout.item(item_id)
        if item is None:
            return
        path = Path(item.prompt)
        if path.suffix.lower() == ".vxclarity" and path.is_file():
            self.open_project(path)

    def _on_refresh(self) -> None:
        phase = self._layout.phase
        if phase is Phase.BUSY and self._last_phase is Phase.BUSY:
            return
        self._last_phase = phase
        self._sync_pages()
        if phase is not Phase.BUSY:
            self._publish_output()

    def _on_generate(self, settings: Mapping[str, object] | None = None) -> None:
        # voxprint-common 0.2.5 calls this before the phase becomes busy.
        # Start on the next turn so the checks below see the busy phase, as before.
        del settings
        QTimer.singleShot(0, self._start_restore)

    def _start_restore(self) -> None:
        if self._closing or self._layout.phase is not Phase.BUSY:
            return
        try:
            ready = models_ready()
        except (CatalogError, OSError, ValueError) as exc:
            self._layout.unavailable(str(exc))
            self._refresh_screen()
            return
        if not ready:
            self._layout.unavailable(tr("clarity.models.unavailable"))
            self._refresh_screen()
            return
        source = self._source_path()
        if source is None or not source.is_file():
            self._layout.cancel()
            self._refresh_screen()
            QMessageBox.warning(self, APP_NAME, tr("error.missing_source"))
            return
        destination = self._destination()
        self._output_text = str(destination)
        self._sync_pages()
        spec = self._job_spec(source)
        token = CancelToken()
        job = _RestoreThread(self, spec, token, self._runner, self._sync_project)
        job.signals.progressed.connect(self._on_progress)
        job.signals.finished.connect(self._on_finished)
        self._thread = job
        self._token = token
        job.start()

    def _job_spec(self, source: Path) -> JobSpec:
        return JobSpec(
            source=source,
            destination=self._destination(),
            settings=self.current_settings(),
            ffmpeg=self._ffmpeg or "ffmpeg",
            ffprobe=self._ffprobe or "ffprobe",
            work=self._work_for(self._destination()),
        )

    def _on_progress(self, fraction: float, text: str) -> None:
        if self._layout.phase is not Phase.BUSY or self._screen is None:
            return
        try:
            self._layout.progress(self._step(fraction), 100, self._stage_label(text))
        except ValueError:
            return
        self._screen.refresh()

    def _on_finished(self, ok: bool, message: str) -> None:
        self._thread = None
        self._token = None
        if self._closing or self._screen is None:
            return
        if ok and self._layout.phase is Phase.BUSY:
            self._runs += 1
            self._layout.finish(f"restore-{self._runs}", ReviewedOutput(path=self._destination()))
            self._screen.refresh()
            self._publish_output()
            self._sync_project(self._work_for(self._destination()))
            return
        if message == "cancelled":
            self._sync_project(self._work_for(Path(self._output_text or "out.mkv")))
            self._screen.set_note(tr("status.cancelled"))
            self._sync_pages()
            return
        self._screen.refresh()
        self._screen.set_note(message)
        if message:
            QMessageBox.warning(self, APP_NAME, message)

    def _cancel_job(self) -> None:
        token = self._token
        if token is not None:
            token.cancel()
        if self._screen is not None and self._layout.phase is Phase.BUSY:
            self._screen.set_note(tr("status.cancelling"))

    def _run_repair(self, *, fetch: bool) -> None:
        progress = RepairProgress(self)
        try:
            ensure_shared_runtime(progress)
            ensure_ffmpeg(progress)
            if fetch:
                fetch_weights(catalogue_ids(), dry_run=False)
        except (FfmpegMissing, EngineUnavailable, ClarityError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if self._screen is not None:
            self._screen.set_note(tr("status.ready"))

    def _fetch(self, *, dry_run: bool) -> None:
        ids = self._selected_ids or None
        try:
            rows = fetch_weights(ids, dry_run=dry_run)
        except (EngineUnavailable, ClarityError, KeyError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._show_text(tr("clarity.action.fetch"), json.dumps(rows, indent=2))
        if not dry_run and self._screen is not None:
            self._screen.set_note(tr("status.downloaded"))

    def _register(self, *, remove: bool) -> None:
        try:
            if remove:
                release_shared_runtime(RepairProgress(self))
                others = unregister_user(OWNER)
                folder = models_directory()
            else:
                register_user(OWNER)
                others = []
                folder = models_directory()
        except (EngineUnavailable, ClarityError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if self._report:
            Path(self._report).write_text(f"{len(others)}\n{folder}\n", encoding="utf-8")
        text = "\n".join(others) if others else OWNER
        self._show_text(tr("clarity.action.register"), f"{text}\n{folder}")

    def _show_text(self, title: str, body: str) -> None:
        box = QDialog(self)
        box.setWindowTitle(title)
        box.resize(720, 480)
        layout = QVBoxLayout(box)
        edit = QPlainTextEdit()
        edit.setPlainText(body)
        edit.setReadOnly(True)
        layout.addWidget(edit)
        copy = QPushButton(tr("clarity.action.copy_json"))
        copy.setObjectName("copyJson")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(body))
        close = QPushButton(tr("clarity.action.close"))
        close.clicked.connect(box.accept)
        layout.addWidget(copy)
        layout.addWidget(close)
        box.exec()

    def _open_output_folder(self) -> None:
        folder = self._destination().parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _sync_pages(self) -> None:
        busy = self._layout.phase is Phase.BUSY
        if self._project_page is not None:
            self._project_page.sync(busy=busy)
        if self._setup_page is not None:
            self._setup_page.sync(busy=busy)

    def _refresh_screen(self) -> None:
        if self._screen is not None:
            self._screen.refresh()

    def _publish_output(self) -> None:
        if self._screen is None or not self._layout.prompt.strip():
            return
        destination = self._destination()
        self._screen.set_note(tr("clarity.output.note", path=destination.name))
        note = self._screen.root.findChild(QLabel, "simpleNote")
        if note is not None:
            note.setToolTip(str(destination))

    def _source_path(self) -> Path | None:
        text = self._layout.prompt.strip()
        if not text:
            return None
        return Path(text)

    def _suffix(self) -> str:
        value = self.settings_state.get("job.container")
        return output_suffix(value if isinstance(value, str) else "mkv")

    def _destination(self) -> Path:
        if self._output_text.strip():
            return Path(self._output_text.strip())
        source = self._source_path()
        if source is None:
            return Path("out").with_suffix(self._suffix())
        return source.with_suffix(self._suffix())

    def _work_for(self, destination: Path) -> Path:
        if self._project_path is not None:
            resume = self._project.resume if self._project is not None else {}
            return project_work_dir(self._project_path, resume)
        return destination.with_name(destination.name + ".work")

    def _sync_project(self, work: Path) -> None:
        path = self._project_path
        if path is None or not path.is_file() or not work.is_dir():
            return
        try:
            self._project = sync_resume(path, work)
        except (OSError, ProjectFormatError, ClarityError):
            return

    def _remember(self, path: Path) -> None:
        remember_recent(path)

    def _seed_recent(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        seen = {item.prompt for item in self._layout.history}
        for path in load_recent():
            if path in seen:
                continue
            item_id = f"recent-{len(self._layout.history) + 1}"
            self._layout.history.append(GalleryItem(item_id, path, tuple(self._layout.presets.items()), 0))

    def _stage_label(self, text: str) -> str:
        stage = progress_stage(text)
        if stage == "building_engine":
            return tr("status.building_engine")
        if stage == "loading_cached_engine":
            return tr("status.loading_cached_engine")
        done, separator, total = text.partition("/")
        if separator:
            return tr("status.running", done=done, total=total)
        label = text.strip()
        return label or tr("status.running", done=0, total=1)

    @staticmethod
    def _step(fraction: float) -> int:
        if isinstance(fraction, bool) or not isinstance(fraction, int | float):
            return 0
        return max(0, min(100, int(round(float(fraction) * 100))))

