#!/bin/sh
# Install Clarity by calling the engine's shared runtime.
# Python 3.14, the CUDA 13 wheels, and the engine's default model come from
# voxprint_engine.suite_runtime.ensure_runtime (holder "clarity").
# The engine pin is commit 9183489 (package 1.2.0, API 1).
# TODO: switch the engine pin to tag engine-v1.2.0 when it is published.
# It installs ONNX Runtime GPU and TensorRT.
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

engine="voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@9183489"
if ! command -v python3.14 >/dev/null 2>&1; then
  echo "Python 3.14 was not found. Install the Enlarge bundle, which carries that runtime." >&2
  exit 1
fi

shared=$suite_home/shared
venv=$shared/venv
mkdir -p "$shared"
if [ ! -x "$venv/bin/python" ]; then
  python3.14 -m venv "$venv"
fi
"$venv/bin/python" -m pip install -r "$app/requirements.txt"
"$venv/bin/python" -m pip install "$engine"
"$venv/bin/python" -m pip install "Pillow==12.3.0"
VOXPRINT_HOME=$suite_home PYTHONPATH="$app${PYTHONPATH:+:$PYTHONPATH}" "$venv/bin/python" -m clarity --install-suite
"$venv/bin/python" -m pip install "Pillow==12.3.0"
