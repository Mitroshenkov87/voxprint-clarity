"""Program-file locations for the suite install layout.

Per-user data stays where ``voxprint_common.paths`` puts it:
``%LOCALAPPDATA%\\Voxprint`` on Windows and ``~/.local/share/voxprint`` on
Linux. Nothing here moves that tree.

``voxprint-common`` 0.1.1 has no installer helper, so the Windows script and
the Linux package write the registry and the default paths themselves.
Peers read ``HKCU\\Software\\Voxprint\\Clarity`` (see the suite discovery
note) or the default directories below.
"""

from clarity.appinfo import APP_NAME

SUITE_MENU = "VOXPrint AI Suite"
WINDOWS_APP_FOLDER = "Clarity"
LINUX_DIR = "/opt/voxprint-ai-suite/clarity"
GUI_EXE = "voxprint-clarity.exe"
CLI_EXE = "voxprint-clarity-cli.exe"
GUI_NAME = "voxprint-clarity"
CLI_NAME = "voxprint-clarity-cli"
OLD_WINDOWS_DIR_NAME = "VoxprintClarity"
OLD_MENU = "Voxprint"
REGISTRY_SUBKEY = r"Software\Voxprint\Clarity"
APP_PATHS = r"Software\Microsoft\Windows\CurrentVersion\App Paths"
DESKTOP_FILE = "ai.voxprint.Clarity.desktop"
OLD_DESKTOP_FILE = "voxprint-clarity.desktop"
PUBLISHER = "VOXPrint"
SETUP_LOG = "voxprint-clarity-setup.log"
DISPLAY_NAME = APP_NAME
