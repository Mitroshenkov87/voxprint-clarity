#!/bin/sh
# Install Clarity into the shared suite folder.
# Python 3.14 comes from uv. uv is downloaded when it is not already on PATH.
# The engine pin is tag engine-v1.2.1 as a GitHub archive, with the gpu extra.
# pyproject.toml is at the repository root, so the URL has no subdirectory.
# tensorrt-cu13-libs is installed from the NVIDIA package index.
# Clarity does not download the engine's default upscale weight.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
if [ -f "$script_dir/requirements.txt" ]; then
  app=$script_dir
elif [ -f "$script_dir/../../requirements.txt" ]; then
  app=$(CDPATH= cd -- "$script_dir/../.." && pwd)
else
  echo "requirements.txt was not found beside the installer" >&2
  exit 1
fi

if [ -n "${VOXPRINT_HOME:-}" ]; then
  suite_home=$VOXPRINT_HOME
else
  case ${XDG_DATA_HOME:-} in
    /*) suite_home=$XDG_DATA_HOME/voxprint ;;
    *) suite_home=$HOME/.local/share/voxprint ;;
  esac
fi

engine="voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.2.1.tar.gz"

uv_bin=$(command -v uv || true)
uv_dir=""
if [ -z "$uv_bin" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download uv." >&2
    exit 1
  fi
  if ! command -v tar >/dev/null 2>&1; then
    echo "tar is required to unpack uv." >&2
    exit 1
  fi
  case $(uname -m) in
    x86_64) uv_arch=x86_64-unknown-linux-gnu ;;
    aarch64|arm64) uv_arch=aarch64-unknown-linux-gnu ;;
    *)
      echo "uv has no Linux build for $(uname -m)." >&2
      exit 1
      ;;
  esac
  uv_dir=$(mktemp -d)
  curl -fsSL -o "$uv_dir/uv.tar.gz" "https://github.com/astral-sh/uv/releases/latest/download/uv-${uv_arch}.tar.gz"
  tar -xzf "$uv_dir/uv.tar.gz" -C "$uv_dir"
  for candidate in "$uv_dir"/*/uv "$uv_dir"/uv; do
    if [ -f "$candidate" ]; then
      uv_bin=$candidate
      break
    fi
  done
  if [ -z "$uv_bin" ]; then
    echo "uv was not in the downloaded archive." >&2
    rm -rf "$uv_dir"
    exit 1
  fi
  chmod +x "$uv_bin"
fi

set +e
"$uv_bin" python install 3.14
install_status=$?
python=$("$uv_bin" python find 3.14)
find_status=$?
set -e
if [ -n "$uv_dir" ]; then
  rm -rf "$uv_dir"
fi
if [ "$install_status" -ne 0 ] || [ "$find_status" -ne 0 ] || [ -z "$python" ] || [ ! -x "$python" ]; then
  echo "Python 3.14 was not found after uv python install." >&2
  exit 1
fi

shared=$suite_home/shared
venv=$shared/venv
mkdir -p "$shared"
if [ ! -x "$venv/bin/python" ]; then
  "$python" -m venv "$venv"
fi
if ! "$venv/bin/python" -c "import pip"; then
  "$venv/bin/python" -m ensurepip --upgrade
fi
"$venv/bin/python" -m pip install -r "$app/requirements.txt"
"$venv/bin/python" -m pip install --extra-index-url "https://pypi.nvidia.com" "$engine"
"$venv/bin/python" -m pip install "Pillow==12.3.0"
VOXPRINT_HOME=$suite_home PYTHONPATH="$app${PYTHONPATH:+:$PYTHONPATH}" "$venv/bin/python" -m clarity --install-suite
if [ -f "$app/installer/pin_runtime.py" ]; then
  pin=$app/installer/pin_runtime.py
  lock=$app/installer/runtime_lock.json
else
  pin=$app/pin_runtime.py
  lock=$app/runtime_lock.json
fi
reqs=$app/requirements-gpu.txt
if [ ! -f "$pin" ] || [ ! -f "$lock" ] || [ ! -f "$reqs" ]; then
  echo "pin_runtime.py, the runtime lock, or requirements-gpu.txt is missing" >&2
  exit 1
fi
"$venv/bin/python" "$pin" --python "$venv/bin/python" --lock "$lock" --requirements "$reqs"
"$venv/bin/python" -m pip install "Pillow==12.3.0"
