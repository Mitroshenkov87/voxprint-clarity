#!/bin/sh
# Install Clarity by calling the engine's shared runtime.
# Python 3.14, the CUDA 13 wheels, and the engine's default model come from
# voxprint_engine.suite_runtime.ensure_runtime (holder "clarity").
# TODO: switch the engine pin to tag engine-v1.1.0 when it is published.
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

engine="voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@d11d60921448b697d832b2699e6e3e04c62c6dfd"
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
if [ -f "$script_dir/runtime_lock.json" ]; then
  lock=$script_dir/runtime_lock.json
elif [ -f "$script_dir/../runtime_lock.json" ]; then
  lock=$script_dir/../runtime_lock.json
else
  echo "runtime lock missing" >&2
  exit 1
fi
ort=$("$venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$lock" onnxruntime_gpu)
libs=$("$venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$lock" tensorrt_cu13_libs)
meta=$("$venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$lock" tensorrt_cu13)
"$venv/bin/python" -m pip uninstall -y onnxruntime onnxruntime-gpu tensorrt tensorrt-cu13 tensorrt-cu13-bindings tensorrt-cu13-libs || true
"$venv/bin/python" -m pip install --extra-index-url "https://pypi.nvidia.com" "onnxruntime-gpu==$ort" "tensorrt-cu13-libs==$libs"
"$venv/bin/python" -m pip install --no-deps "tensorrt-cu13==$meta"
"$venv/bin/python" -c 'import importlib.metadata as m; names={d.metadata["Name"].lower() for d in m.distributions()}; raise SystemExit(1 if "onnxruntime" in names else 0)'
