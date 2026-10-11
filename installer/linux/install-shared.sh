#!/bin/sh
# Install the shared suite during setup. A later launch does not download
# these pieces again. Re-running the script reuses a runtime and an ffmpeg
# binary that are already in the shared folder.
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

shared=$suite_home/shared
runtime=$shared/runtime/py3.14-torch2.14.1-cu130
venv=$runtime/venv
py=$venv/bin/python
mkdir -p "$runtime"

if [ ! -x "$py" ]; then
  uvdir=$(mktemp -d)
  curl -fsSL -o "$uvdir/uv.tar.gz" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
  tar -C "$uvdir" -xzf "$uvdir/uv.tar.gz"
  uv=$(find "$uvdir" -type f -name uv | head -n 1)
  if [ -z "$uv" ]; then
    echo "uv was not in the archive" >&2
    exit 1
  fi
  chmod 755 "$uv"
  "$uv" python install 3.14
  base=$("$uv" python find 3.14)
  "$uv" venv --python "$base" "$venv"
  "$uv" pip install --python "$py" "torch==2.14.1" --index-url "https://download.pytorch.org/whl/cu130"
  "$uv" pip install --python "$py" -r "$app/requirements.txt"
  if [ -f "$app/requirements-bench.txt" ]; then
    "$uv" pip install --python "$py" -r "$app/requirements-bench.txt"
  fi
  "$uv" pip install --python "$py" "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0"
  "$uv" pip install --python "$py" --upgrade "onnxruntime-gpu"
fi

ffmpeg_dir=$shared/ffmpeg
ffmpeg=$ffmpeg_dir/ffmpeg
if [ ! -x "$ffmpeg" ]; then
  mkdir -p "$ffmpeg_dir"
  archive=$(mktemp)
  curl -fsSL -o "$archive" "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl.tar.xz"
  if tar -tJf "$archive" | grep -q '\.\.'; then
    echo "refusing an ffmpeg archive that contains a parent path" >&2
    exit 1
  fi
  extract=$(mktemp -d)
  tar -C "$extract" -xJf "$archive" --no-absolute-names
  found=$(find "$extract" -type f -name ffmpeg | head -n 1)
  if [ -z "$found" ]; then
    echo "the LGPL ffmpeg archive has no ffmpeg binary" >&2
    exit 1
  fi
  cp "$found" "$ffmpeg"
  chmod 755 "$ffmpeg"
  probe=$(find "$extract" -type f -name ffprobe | head -n 1)
  if [ -n "$probe" ]; then
    cp "$probe" "$ffmpeg_dir/ffprobe"
    chmod 755 "$ffmpeg_dir/ffprobe"
  fi
  rm -rf "$archive" "$extract"
fi

for name in python torch ffmpeg engine; do
  mkdir -p "$shared/$name"
  printf 'ready\n' > "$shared/$name/READY"
done

VOXPRINT_SHARED=$shared "$py" - << 'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["VOXPRINT_SHARED"])
path = root / "manifest.json"
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    data = {}
comps = data.get("components") if isinstance(data.get("components"), dict) else {}
versions = {
    "python": "3.14",
    "torch": "2.14.1+cu130",
    "ffmpeg": "lgpl",
    "engine": "engine-v1.0.0",
}
for name, version in versions.items():
    entry = comps.get(name) if isinstance(comps.get(name), dict) else {"users": {}}
    users = entry.get("users") if isinstance(entry.get("users"), dict) else {}
    users["clarity"] = int(users.get("clarity") or 0) + 1
    entry["version"] = version
    entry["users"] = users
    comps[name] = entry
path.write_text(json.dumps({"format_version": 1, "components": comps}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
PY

if [ "${VOXPRINT_SUITE_INSTALLING:-}" != "1" ]; then
  VOXPRINT_HOME=$suite_home PYTHONPATH="$app${PYTHONPATH:+:$PYTHONPATH}" "$py" -m clarity --install-suite
fi
