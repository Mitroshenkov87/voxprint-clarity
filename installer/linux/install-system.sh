#!/bin/sh
# Copy a Clarity payload to /opt/voxprint-ai-suite/clarity and register it.
# Usage: install-system.sh <payload-dir>
# Needs permission to write /opt and /usr/local. Does not touch user data.
set -eu

payload=$(CDPATH= cd -- "$1" && pwd)
target_root=${VOXPRINT_SUITE_ROOT:-/opt/voxprint-ai-suite}
target=$target_root/clarity
script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)

mkdir -p "$target_root"
if [ "$payload" != "$target" ]; then
  mkdir -p "$target"
  cp -a "$payload"/. "$target"/
fi

sh "$script_dir/migrate-old.sh" "" "$target"

ln -sfn "$target/voxprint-clarity" /usr/local/bin/voxprint-clarity
ln -sfn "$target/voxprint-clarity-cli" /usr/local/bin/voxprint-clarity-cli

mkdir -p /usr/local/share/applications /usr/share/mime/packages /usr/local/share/icons/hicolor/256x256/apps
if [ -f "$target/BUILD.json" ]; then
  version=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$target/BUILD.json")
else
  version=""
fi
if [ -z "$version" ]; then
  version=1.0.0-rc
fi
case $version in
  *[!0-9A-Za-z._-]*) version=1.0.0-rc ;;
esac
sed "s#^Exec=.*#Exec=\"$target/voxprint-clarity\" %f#; s#^X-Voxprint-Version=.*#X-Voxprint-Version=$version#" \
  "$script_dir/ai.voxprint.Clarity.desktop" > /usr/local/share/applications/ai.voxprint.Clarity.desktop
if [ -f "$target/share/mime/packages/voxprint-clarity.xml" ]; then
  cp "$target/share/mime/packages/voxprint-clarity.xml" /usr/share/mime/packages/voxprint-clarity.xml
elif [ -f "$script_dir/voxprint-clarity.xml" ]; then
  cp "$script_dir/voxprint-clarity.xml" /usr/share/mime/packages/voxprint-clarity.xml
fi
icon=$target/share/icons/hicolor/256x256/apps/voxprint-clarity.png
if [ ! -f "$icon" ] && [ -f "$target/voxprint-clarity.png" ]; then
  icon=$target/voxprint-clarity.png
fi
if [ -f "$icon" ]; then
  cp "$icon" /usr/local/share/icons/hicolor/256x256/apps/voxprint-clarity.png
fi
if command -v update-mime-database >/dev/null 2>&1; then
  update-mime-database /usr/share/mime || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/local/share/applications || true
fi
