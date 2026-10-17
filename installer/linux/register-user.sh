#!/bin/sh
# Register Clarity for the current user so other suite apps can find it.
# Usage: register-user.sh [app-dir]
# Removes the old desktop name. Does not remove ~/.local/share/voxprint.
set -eu

app=${1:-/opt/voxprint-ai-suite/clarity}
app=${app%/}
if [ -n "${XDG_DATA_HOME:-}" ]; then
  case $XDG_DATA_HOME in
    /*) data=$XDG_DATA_HOME ;;
    *) data=$HOME/.local/share ;;
  esac
else
  data=$HOME/.local/share
fi

apps=$data/applications
mkdir -p "$apps"
rm -f "$apps/voxprint-clarity.desktop"

version=1.0.0-rc
if [ -f "$app/BUILD.json" ]; then
  parsed=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$app/BUILD.json")
  if [ -n "$parsed" ]; then
    version=$parsed
  fi
fi
case $version in
  *[!0-9A-Za-z._-]*) version=1.0.0-rc ;;
esac

cat > "$apps/ai.voxprint.Clarity.desktop" << EOF
[Desktop Entry]
Type=Application
Name=VOXPrint AI Clarity
Comment=Lightweight non-generative video restoration
Exec="$app/voxprint-clarity" %f
Icon=voxprint-clarity
Terminal=false
MimeType=application/vnd.voxprint.clarity+zip;
Categories=AudioVideo;Video;
StartupWMClass=voxprint-clarity
X-Voxprint-Ipc=
X-Voxprint-Protocol=0
X-Voxprint-Version=$version
EOF
