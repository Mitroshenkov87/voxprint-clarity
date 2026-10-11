#!/bin/sh
# Experimental Linux package: a tarball and, when appimagetool runs, an AppImage.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
stage="$root/dist/VoxprintClarity.AppDir"
rm -rf "$stage"
mkdir -p "$stage/usr/bin" "$stage/usr/share/applications" "$stage/usr/share/icons/hicolor/256x256/apps" "$stage/usr/share/mime/packages"
cp -a "$root/clarity" "$root/locales" "$root/docs" "$root/assets" "$stage/usr/bin/"
cp "$root/main.py" "$root/LICENSE" "$root/NOTICE" "$root/requirements.txt" "$root/requirements-gpu.txt" "$root/pyproject.toml" "$root/BUILD.json" "$stage/usr/bin/"
cp "$root/branding/clarity-256.png" "$stage/usr/share/icons/hicolor/256x256/apps/voxprint-clarity.png"
cp "$root/branding/clarity-256.png" "$stage/clarity.png"
cp "$root/branding/clarity-256.png" "$stage/voxprint-clarity.png"
cp "$root/branding/clarity-256.png" "$stage/.DirIcon"
cp "$root/installer/linux/voxprint-clarity.desktop" "$stage/usr/share/applications/voxprint-clarity.desktop"
cp "$root/installer/linux/voxprint-clarity.desktop" "$stage/voxprint-clarity.desktop"
cp "$root/installer/linux/voxprint-clarity.xml" "$stage/usr/share/mime/packages/voxprint-clarity.xml"
cp "$root/installer/linux/uninstall.sh" "$stage/usr/bin/voxprint-clarity-uninstall"
chmod 755 "$stage/usr/bin/voxprint-clarity-uninstall"
cp "$root/installer/linux/install-shared.sh" "$stage/usr/bin/voxprint-clarity-install"
chmod 755 "$stage/usr/bin/voxprint-clarity-install"
cp "$root/installer/runtime_lock.json" "$stage/usr/bin/runtime_lock.json"
cat > "$stage/usr/bin/voxprint-clarity" << 'EOF'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${VOXPRINT_HOME:-}" ]; then
  suite_home=$VOXPRINT_HOME
else
  case ${XDG_DATA_HOME:-} in
    /*) suite_home=$XDG_DATA_HOME/voxprint ;;
    *) suite_home=$HOME/.local/share/voxprint ;;
  esac
fi
if [ -d "$suite_home/shared/ffmpeg" ]; then
  PATH="$suite_home/shared/ffmpeg:$PATH"
  export PATH
fi
exec python3 "$here/main.py" "$@"
EOF
chmod 755 "$stage/usr/bin/voxprint-clarity"
cat > "$stage/AppRun" << 'EOF'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
export PYTHONPATH="$here/usr/bin${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${VOXPRINT_HOME:-}" ]; then
  suite_home=$VOXPRINT_HOME
else
  case ${XDG_DATA_HOME:-} in
    /*) suite_home=$XDG_DATA_HOME/voxprint ;;
    *) suite_home=$HOME/.local/share/voxprint ;;
  esac
fi
if [ -d "$suite_home/shared/ffmpeg" ]; then
  PATH="$suite_home/shared/ffmpeg:$PATH"
  export PATH
fi
exec python3 "$here/usr/bin/main.py" "$@"
EOF
chmod 755 "$stage/AppRun"
mkdir -p "$root/dist"
tar -C "$root/dist" -czf "$root/dist/VoxprintClarity-linux.tar.gz" VoxprintClarity.AppDir
echo "wrote $root/dist/VoxprintClarity-linux.tar.gz"
tool="$root/dist/appimagetool"
if [ ! -x "$tool" ]; then
  curl -fsSL -o "$tool" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" || true
  chmod 755 "$tool" 2>/dev/null || true
fi
if [ -x "$tool" ]; then
  ARCH=x86_64 "$tool" --appimage-extract-and-run "$stage" "$root/dist/VoxprintClarity-x86_64.AppImage" || echo "AppImage step failed; the tarball is the package"
fi
ls -l "$root/dist"
