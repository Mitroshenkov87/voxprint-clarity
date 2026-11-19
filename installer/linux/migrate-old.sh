#!/bin/sh
# Remove a previous Clarity install's launchers and shortcuts.
# Usage: migrate-old.sh <sysroot> <new-app-dir>
# An empty sysroot means the real filesystem root.
# Per-user data under ~/.local/share/voxprint is never removed.
set -eu

sysroot=${1:-}
new_app=${2:-/opt/voxprint-ai-suite/clarity}
new_app=${new_app%/}

remove_file() {
  path=$1
  if [ -e "$path" ] || [ -L "$path" ]; then
    rm -f "$path"
  fi
}

# A symlink that already points at the new install stays.
remove_launcher() {
  path=$1
  if [ ! -e "$path" ] && [ ! -L "$path" ]; then
    return 0
  fi
  if [ -L "$path" ]; then
    dest=$(readlink -f "$path" 2>/dev/null || true)
    case $dest in
      "$new_app" | "$new_app"/*) return 0 ;;
    esac
  fi
  case $path in
    */.local/share/voxprint | */.local/share/voxprint/*) return 0 ;;
  esac
  rm -f "$path"
}

remove_tree() {
  path=$1
  case $path in
    */.local/share/voxprint | */.local/share/voxprint/*) return 0 ;;
    "$new_app" | "$new_app"/*) return 0 ;;
  esac
  if [ -d "$path" ]; then
    rm -rf "$path"
  fi
}

prefix=$sysroot
remove_launcher "$prefix/usr/bin/voxprint-clarity"
remove_launcher "$prefix/usr/bin/voxprint-clarity-cli"
remove_launcher "$prefix/usr/local/bin/voxprint-clarity"
remove_launcher "$prefix/usr/local/bin/voxprint-clarity-cli"
remove_file "$prefix/usr/bin/voxprint-clarity-uninstall"
remove_file "$prefix/usr/share/applications/voxprint-clarity.desktop"
remove_file "$prefix/usr/local/share/applications/voxprint-clarity.desktop"
remove_file "$prefix/usr/share/icons/hicolor/256x256/apps/voxprint-clarity.png"

# Old fixed directories from before the suite folder. Only a Clarity tree
# is removed, and the new install directory is kept.
remove_old_app_dir() {
  path=$1
  if [ -f "$path/main.py" ] && [ -f "$path/clarity/appinfo.py" ]; then
    remove_tree "$path"
  fi
}
remove_old_app_dir "$prefix/opt/VoxprintClarity"
remove_old_app_dir "$prefix/opt/voxprint-clarity"

# Only the real install (an empty sysroot) edits the caller's home.
# A test sysroot must not touch the account that runs the test.
if [ -z "$sysroot" ] && [ -n "${HOME:-}" ]; then
  apps=${XDG_DATA_HOME:-$HOME/.local/share}/applications
  case $apps in
    /*) remove_file "$apps/voxprint-clarity.desktop" ;;
  esac
fi
