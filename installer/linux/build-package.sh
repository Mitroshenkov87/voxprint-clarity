#!/bin/sh
# Experimental Linux package: a tarball and, when appimagetool runs, an AppImage.
# The tree installs to /opt/voxprint-ai-suite/clarity.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
stage="$root/dist/voxprint-clarity"
rm -rf "$stage" "$root/dist/voxprint-clarity.AppDir"
mkdir -p "$stage/share/applications" "$stage/share/icons/hicolor/256x256/apps" "$stage/share/mime/packages"
cp -a "$root/clarity" "$root/locales" "$root/docs" "$root/assets" "$stage/"
find "$stage" -depth -type d -name __pycache__ -exec rm -rf {} +
find "$stage" -type f -name '*.pyc' -delete
cp "$root/main.py" "$root/LICENSE" "$root/NOTICE" "$root/requirements.txt" "$root/requirements-gpu.txt" "$root/pyproject.toml" "$root/BUILD.json" "$root/mdl.json" "$stage/"
cp "$root/branding/clarity-256.png" "$stage/share/icons/hicolor/256x256/apps/voxprint-clarity.png"
cp "$root/branding/clarity-256.png" "$stage/clarity.png"
cp "$root/branding/clarity-256.png" "$stage/voxprint-clarity.png"
cp "$root/branding/clarity-256.png" "$stage/.DirIcon"
version=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$root/BUILD.json")
if [ -z "$version" ]; then
  version=1.0.0-rc
fi
sed "s/^X-Voxprint-Version=.*/X-Voxprint-Version=$version/" "$root/installer/linux/ai.voxprint.Clarity.desktop" > "$stage/share/applications/ai.voxprint.Clarity.desktop"
sed "s/^X-Voxprint-Version=.*/X-Voxprint-Version=$version/" "$root/installer/linux/voxprint-clarity.desktop" > "$stage/voxprint-clarity.desktop"
cp "$root/installer/linux/voxprint-clarity.xml" "$stage/share/mime/packages/voxprint-clarity.xml"
cp "$root/installer/linux/uninstall.sh" "$stage/voxprint-clarity-uninstall"
cp "$root/installer/linux/migrate-old.sh" "$stage/migrate-old.sh"
cp "$root/installer/linux/install-system.sh" "$stage/install-system.sh"
cp "$root/installer/linux/register-user.sh" "$stage/register-user.sh"
cp "$root/installer/linux/ai.voxprint.Clarity.desktop" "$stage/ai.voxprint.Clarity.desktop"
chmod 755 "$stage/voxprint-clarity-uninstall" "$stage/migrate-old.sh" "$stage/install-system.sh" "$stage/register-user.sh"
cp "$root/installer/linux/install-shared.sh" "$stage/voxprint-clarity-install"
chmod 755 "$stage/voxprint-clarity-install"
cp "$root/installer/runtime_lock.json" "$stage/runtime_lock.json"
cp "$root/installer/pin_runtime.py" "$stage/pin_runtime.py"
cp "$root/installer/fetch_common.py" "$stage/fetch_common.py"
cp "$root/installer/stage_suite_runtime.py" "$stage/stage_suite_runtime.py"
cat > "$stage/voxprint-clarity" << 'EOF'
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
if [ -x "$suite_home/shared/venv/bin/python" ]; then
  py=$suite_home/shared/venv/bin/python
else
  py=python3
fi
exec "$py" "$here/main.py" "$@"
EOF
chmod 755 "$stage/voxprint-clarity"
cat > "$stage/voxprint-clarity-cli" << 'EOF'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
exec "$here/voxprint-clarity" "$@"
EOF
chmod 755 "$stage/voxprint-clarity-cli"
cat > "$stage/AppRun" << 'EOF'
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
if [ -x "$suite_home/shared/venv/bin/python" ]; then
  py=$suite_home/shared/venv/bin/python
else
  py=python3
fi
exec "$py" "$here/main.py" "$@"
EOF
chmod 755 "$stage/AppRun"
mkdir -p "$root/dist"
tar -C "$root/dist" -czf "$root/dist/voxprint-clarity-linux.tar.gz" voxprint-clarity
echo "wrote $root/dist/voxprint-clarity-linux.tar.gz"
cp -a "$stage" "$root/dist/voxprint-clarity.AppDir"
tool="$root/dist/appimagetool"
if [ ! -x "$tool" ]; then
  curl -fsSL -o "$tool" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" || true
  chmod 755 "$tool" 2>/dev/null || true
fi
if [ -x "$tool" ]; then
  ARCH=x86_64 "$tool" --appimage-extract-and-run "$root/dist/voxprint-clarity.AppDir" "$root/dist/voxprint-clarity-x86_64.AppImage" || echo "AppImage step failed; the tarball is the package"
fi
ls -l "$root/dist"
