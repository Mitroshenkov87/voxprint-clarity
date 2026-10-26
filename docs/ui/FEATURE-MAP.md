# Feature map

The shared video sample is the main screen (the Video tab). Every control from
`clarity/ui/main_window.py` and `clarity/ui/suite_dialog.py`, and every CLI
flag and subcommand, stays reachable from that window. Simple chips, the
Project and Setup tabs, the menus, and the Expert panel are the places. The
CLI program itself is unchanged: the same flags, the same subcommands, and the
same exit codes.

Status is `kept` when the action is reachable. `done` means an owner decision
was settled and the action is in this shell.

## Current window

| Feature | Old place | New place | Status |
|---|---|---|---|
| Window title | Title bar, `app.title` | Title bar, `splash.app.clarity` | kept |
| Subtitle | Label under the title | Label under the tab bar | kept |
| Source video path | Source field | Video tab drop area, and Project tab Source field. Once a file is loaded the drop area shows that file name and Choose another video. The full path stays in the tooltip | kept |
| Browse for a source | Browse button | Project tab Browse, and a click on the Video tab drop area | kept |
| Output path | Output field | Project tab Output field. The Video tab note shows the output file name. The full path is the note tooltip, and Open folder opens that folder | kept |
| Export (choose the output video) | Export button | Project tab Export, and File → Export | kept |
| Default output name | `.clarity.mkv` or `.clarity.mp4` from the container | Same suffix from the File chip. An empty Output field uses it | kept |
| Cleanup strength | Mode combo, Clean or Strong | Video tab Cleanup chips: Light, Medium, or Strong. Light is clean mode and sigma 15. Medium stays on clean mode and raises sigma to 25. Strong is the strong model and leaves sigma at 15, because that model does not read sigma | kept |
| Container | Container combo, MKV or MP4 | Video tab Format chips: MKV or MP4 | kept |
| Codec | Codec combo, AV1 or HEVC | Setup tab Codec combo, and the Expert panel codec field | kept |
| Language | Language combo (shown, not wired) | Setup tab Language combo. It switches English and Russian | kept |
| Upscale | Upscale checkbox | Video tab Size chip: Same size or Twice as big | kept |
| Anime look | Anime checkbox, disabled unless upscale is on | Video tab Look chip, and the Setup tab Anime checkbox. The checkbox stays disabled unless Size is Twice as big. The job turns anime on only when upscale is on | kept |
| Colour cleanup | Colour checkbox. It set white balance and levels together | Video tab What to fix. Colours turns white balance and levels on or off together. Noise and compression blocks are not separate switches: clean mode always runs both models, and strong mode runs one blind model for both. Those two chips stay selected | kept |
| Sharpen | Sharpen slider, 0 to 100 | Video tab Sharpness chips: Off, Normal, or Sharper (0, 0.35, 0.6). The Setup slider and the Expert field write the same number. Normal is 0.35 | kept |
| Upscale hint | Hint under the options | Setup tab hint under Anime | kept |
| Progress | Progress bar, 0 to 100 | Video tab progress, with the stage label | kept |
| Status text | Ready, running, cancelling, done, cancelled, saved, resumed, downloaded, and errors | Video tab note, and the stage label while a job runs | kept |
| Start | Start button | Video tab Restore button | kept |
| Cancel | The same button while a job runs | Video tab Cancel. It cancels the job token | kept |
| Open a `.vxclarity` project | Open button | Project tab Open, File → Open, and a dropped or typed `.vxclarity` path | kept |
| Save | Save button | Project tab Save, File → Save, and the shell Save action after a result | kept |
| Save as | Save as button | Project tab Save as, and File → Save as | kept |
| Save a copy | Save a copy button | Project tab Save a copy, and File → Save a copy | kept |
| Download models | Download models button | Project tab Download models, and Models → Download models | kept |
| Recent projects | Recent combo | Project tab recent combo. Choosing one opens that project | kept |
| Suite icons | Save, Save as, Save a copy, Export, Download | The same icons on those Project tab buttons | kept |
| Resume note | Status text when the project has a journal | Video tab note after Open, when the project has a journal | kept |
| Missing source or a bad project | Warning dialog | The same warning dialog | kept |
| Acrylic backdrop | Applied when the window is shown | Applied when the window is shown | kept |
| Version label | Footer | Footer, and Tools → the version label, which opens the version text | kept |
| Close | Cancel the job and wait for the worker | Cancel the job and wait for the worker | kept |
| Runtime and ffmpeg repair dialog | `RepairProgress` while a line starts with Installing, Repairing, or Removing | Setup tab Repair runtime, Install suite, and Unregister. Tools has Repair and Install suite. The same dialog opens only for those lines | kept |

## Job settings the old form did not show

The old form left these at the `Settings` defaults. They stay reachable. Audio
copy and the frame cap had no control before, and they still have none.
Clarity does not change the frame rate, and it does not stabilise the picture.
The only geometry change is the optional 2x size. There is no control for
either of those.

| Setting | Old place | New place | Status |
|---|---|---|---|
| Sigma | Default 15, no control | Expert panel, and the Cleanup chips (Light 15, Medium 25, Strong 15) | kept |
| Tile | Default 0, no control | Expert panel. The stored value stays 0, and the field shows Auto | kept |
| Batch | Default 2, no control | Expert panel | kept |
| Overlap | Default 16, no control | Expert panel | kept |
| Quality factor | Default 40, no control | Expert panel, under Restoration, labeled Block removal. It sets how hard the deblocker removes compression blocks. A lower number removes more. It is not the file size | kept |
| File quality | The encoder used a fixed AV1 CQ of 20 | Video tab File quality chips: Smaller file (CRF 32), Balanced (CRF 20), Best quality (CRF 16). Balanced is the previous fixed value. HEVC receives the same number as its CQ. The Expert field is labeled AV1 quality (CRF) | kept |
| Upscale strength | Default 0, no control | Expert panel, under Upscale, labeled Softer 2× mix. It mixes the softer denoised 2× model into the upscale. 0 leaves that mix off | kept |
| Restore model list | Chosen from the mode, no field | Expert panel text. Presets write it. Manage models edits the catalogue | kept |
| Upscale model | Chosen from the anime switch, no field | Expert panel text. The Look chip writes it | kept |
| Copy audio | Always on, no control | Always on, no control | kept |
| Max frames | Default 1000, no control | Default 1000, no control | kept |

## Command line

The parsers in `clarity/cli.py` are unchanged. A GUI row is the place a person
reaches that option without typing it. `--help` and `-h` stay on the program.

| Feature | Old place | New place | Status |
|---|---|---|---|
| Positional project path | `clarity PATH.vxclarity` opens that project | CLI unchanged. Project tab Open does the same load | kept |
| `--version` | Prints the build label and exits | CLI unchanged. Tools → the version label shows the same text | kept |
| `--help`, `-h` | Parser help | CLI unchanged | kept |
| `--selftest` | Opens the window offscreen and checks Start | CLI unchanged. The window it opens is this shell | kept |
| `--selftest` GPU TensorRT and CUDA sessions | A GPU selftest opens real sessions and fails if either falls back to CPU | Tools → Check the graphics card runs `python -m clarity --selftest` in a subprocess and shows pass or fail, with the process output behind Details. Setup has the same button. The CLI flag is unchanged | done |
| `--no-gpu` | With `--selftest`, skip the CUDA and TensorRT sessions. The plan is still built | CLI unchanged. Setup tab Check the restore plan shows that plan. Tools → Check the restore plan is the same action | kept |
| `--register-models-user` | Register Clarity in the shared models folder | CLI unchanged. Setup tab Register, and Models → Register | kept |
| `--unregister-models-user` | Release the runtime and unregister | CLI unchanged. Setup tab Unregister, and Models → Unregister. The repair dialog is shown while runtimes are removed | kept |
| `--install-suite` | Repair the runtime and ffmpeg, then fetch every catalogue weight | CLI unchanged. Setup tab Install suite, and Tools → Install suite | kept |
| `--out` | Write the models-folder report | CLI unchanged. Setup tab Report file receives the same `{count}\n{folder}\n` text when Register or Unregister runs | kept |
| `--json` | JSON lines on stdout for a headless command | CLI unchanged. List, fetch, dry-run, plan, and version dialogs have Copy JSON | kept |
| `--dry-run` | Validate and print a plan. Do not use the GPU | CLI unchanged. Setup tab Dry-run restore, Fetch dry-run, and Check the restore plan. Tools has Dry-run restore and Check the restore plan | kept |
| `restore` | Restore one video | CLI unchanged. Video tab Restore runs the same job | kept |
| `restore --input` | Source video | CLI unchanged. Project tab Source, or the Video tab drop area | kept |
| `restore --output` | Destination video | CLI unchanged. Project tab Output and Export | kept |
| `restore --mode` | `clean` or `strong` | CLI unchanged. Video tab Cleanup chip | kept |
| `restore --upscale` | Optional 2x | CLI unchanged. Video tab Size chip | kept |
| `restore --format` | `mkv`, `mp4`, `av1`, or `hevc` | CLI unchanged. Format chips set `mkv` or `mp4`. Codec combo and the Expert panel set `av1` or `hevc` | kept |
| `restore --codec` | `av1` or `hevc` | CLI unchanged. Setup tab Codec, and the Expert panel | kept |
| `restore --ffmpeg` | ffmpeg binary | CLI unchanged. Setup tab ffmpeg field. `project resume` uses the same field | kept |
| `restore --ffprobe` | ffprobe binary | CLI unchanged. Setup tab ffprobe field. `project resume` uses the same field | kept |
| `project` | Project commands | CLI unchanged. Project tab | kept |
| `project open` | Read a project and print it. Argument `path` | CLI unchanged. Project tab Open loads that file into the form | kept |
| `project resume` | Continue from the journal. Argument `path`, plus `--ffmpeg` and `--ffprobe` | CLI unchanged. Open loads the journal, and Restore continues in the work folder. ffmpeg and ffprobe come from Setup | kept |
| `info` | Print the build label | CLI unchanged. Tools → the version label | kept |
| `version` | Print the build label | CLI unchanged. Tools → the version label | kept |
| `models` | Catalogue commands | CLI unchanged. Models menu and the Setup tab | kept |
| `models list` | Print the catalogue | CLI unchanged. Setup tab List models, and Models → List models | kept |
| `models fetch` | Fetch weights | CLI unchanged. Setup tab Fetch, Project tab Download models, and Models → Fetch and Download models | kept |
| `models fetch --id` | Repeat to fetch selected ids | CLI unchanged. Setup tab model checklist, then Fetch or Fetch dry-run. An empty checklist fetches the catalogue | kept |
| `selftest` | Headless selftest, including `--no-gpu`, `--json`, and `--dry-run` | CLI unchanged. The window half is this shell. The plan half is Check the restore plan | kept |
| `selftest` GPU sessions | Opens real TensorRT and CUDA sessions unless `--no-gpu` is set | Tools → Check the graphics card, and Setup → Check the graphics card. The same subprocess shows pass or fail | done |
