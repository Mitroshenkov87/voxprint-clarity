#!/bin/sh
# Remove Clarity's launchers, shortcuts, and the /opt install.
# The default prefix is /usr. Per-user data in ~/.local/share/voxprint stays.
set -eu
prefix=${1:-/usr}
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
if [ -f "$here/main.py" ]; then
  PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" python3 "$here/main.py" --unregister-models-user || true
fi
rm -f "$prefix/share/mime/packages/voxprint-clarity.xml"
rm -f "$prefix/share/applications/voxprint-clarity.desktop"
rm -f "$prefix/share/applications/ai.voxprint.Clarity.desktop"
rm -f "$prefix/share/icons/hicolor/256x256/apps/voxprint-clarity.png"
rm -f "$prefix/bin/voxprint-clarity"
rm -f "$prefix/bin/voxprint-clarity-cli"
if [ -f "$here/migrate-old.sh" ]; then
  sh "$here/migrate-old.sh" "" "/opt/voxprint-ai-suite/clarity"
fi
if [ -d /opt/voxprint-ai-suite/clarity ]; then
  rm -rf /opt/voxprint-ai-suite/clarity
fi
rmdir /opt/voxprint-ai-suite 2>/dev/null || true
rm -f /usr/local/bin/voxprint-clarity /usr/local/bin/voxprint-clarity-cli
rm -f /usr/local/share/applications/ai.voxprint.Clarity.desktop
rm -f /usr/local/share/applications/voxprint-clarity.desktop
if [ -n "${HOME:-}" ]; then
  apps=${XDG_DATA_HOME:-$HOME/.local/share}/applications
  case $apps in
    /*)
      rm -f "$apps/voxprint-clarity.desktop"
      rm -f "$apps/ai.voxprint.Clarity.desktop"
      ;;
  esac
fi
if command -v update-mime-database >/dev/null 2>&1; then
  update-mime-database "$prefix/share/mime" || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$prefix/share/applications" || true
fi
