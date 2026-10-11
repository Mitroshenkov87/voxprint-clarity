#!/bin/sh
# Remove the Clarity file association and the package launcher from a prefix.
# The default prefix is /usr.
set -eu
prefix=${1:-/usr}
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
if [ -f "$here/main.py" ]; then
  PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" python3 "$here/main.py" --unregister-models-user || true
fi
rm -f "$prefix/share/mime/packages/voxprint-clarity.xml"
rm -f "$prefix/share/applications/voxprint-clarity.desktop"
rm -f "$prefix/share/icons/hicolor/256x256/apps/voxprint-clarity.png"
rm -f "$prefix/bin/voxprint-clarity"
if command -v update-mime-database >/dev/null 2>&1; then
  update-mime-database "$prefix/share/mime" || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$prefix/share/applications" || true
fi
