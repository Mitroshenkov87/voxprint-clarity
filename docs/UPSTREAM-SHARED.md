# Shared suite folder

Clarity installs Python, torch, an LGPL ffmpeg build, and the engine into one
directory that the other Voxprint apps can reuse. This note is the addition
proposed for `voxprint_engine` at tag `engine-v1.0.0`. Clarity implements the
layout itself until that function exists.

## What the engine already owns

`voxprint_engine.paths.voxprint_home()` is the suite home:

- `VOXPRINT_HOME` when it is set
- `%LOCALAPPDATA%\Voxprint` on Windows
- `$XDG_DATA_HOME/voxprint` or `~/.local/share/voxprint` on Linux

`voxprint_engine.paths.models_dir()` is the weights folder. `store.py` keeps a
boolean `.users.json` (`register` / `unregister`) beside those weights. Clarity
keeps using both. Weight files stay in the models folder.

## What the engine does not own yet

`engine-v1.0.0` has no versioned runtime directory and no integer refcount.
Clarity therefore adds `shared/` under the same home, and will call
`voxprint_engine.paths.shared_dir()` when Enlarge ships it.

Proposed function:

```text
shared_dir() -> <voxprint_home()>/shared
```

Proposed `shared/manifest.json`:

```json
{
  "format_version": 1,
  "components": {
    "python": {"version": "3.14", "users": {"clarity": 1}},
    "torch": {"version": "2.14.1+cu130", "users": {"clarity": 1}},
    "ffmpeg": {"version": "lgpl", "users": {"clarity": 1}},
    "engine": {"version": "engine-v1.0.0", "users": {"clarity": 1}},
    "models": {"version": "default", "users": {"clarity": 1}}
  }
}
```

`users` is an integer per app. An install increments the count. A launch
records the app at 1 when it is absent and leaves a higher count unchanged.
Uninstall decrements. A component directory is removed at zero users. The
runtime directory is removed only when `python`, `torch`, and `engine` are
all at zero. The models entry does not delete weight files; those stay under
`models_dir()` and `.users.json`.

Ready markers are `shared/<component>/READY`. The venv is
`shared/runtime/py3.14-torch2.14.1-cu130/venv`. The ffmpeg binary is
`shared/ffmpeg/ffmpeg` (or `ffmpeg.exe`).
