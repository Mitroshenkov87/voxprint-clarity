# `.vxclarity` format version 1

A Clarity project is a ZIP archive. The conventions match a suite document
package: the first entry is the media type, stored (not deflated), so a file
manager can identify the package without unpacking it.

The public Audiobook Builder repository at `949dc13` does not contain a
`.vxbook` writer or spec. Clarity therefore follows the layout below, which is
the EPUB-style package the suite uses when a product defines one.

## Media type

`application/vnd.voxprint.clarity+zip`

The file name ends with `.vxclarity`.

## Layout

| Entry | Compression | Required |
|---|---|---|
| `mimetype` | stored, no extra field | yes, and it is the first entry |
| `manifest.json` | deflated | yes |
| `settings.json` | deflated | yes |
| `resume.json` | deflated | yes |
| `thumb.png` | deflated | no |

`mimetype` is the ASCII bytes `application/vnd.voxprint.clarity+zip` with no
trailing newline. The local file header has an empty extra field, so at byte
offset 30 the archive contains `mimetypeapplication/vnd.voxprint.clarity+zip`.

## `manifest.json`

```json
{
  "format": "vxclarity",
  "format_version": 1,
  "app": "Voxprint AI Clarity",
  "app_version": "0.1.0",
  "created": "2026-10-10T08:52:00Z",
  "modified": "2026-10-10T09:10:00Z",
  "source": {
    "absolute": "/videos/in.mkv",
    "relative": "in.mkv",
    "size": 47065346,
    "head_sha256": "...",
    "tail_sha256": "...",
    "sha256": ""
  }
}
```

`format_version` is an integer major version. A reader accepts major version 1
only. A higher major version is rejected. A lower major version is rejected
because version 1 is the first version. A value written as `"1.2"` has major
version 1 and is accepted.

`created` stays as it was on the first save. `modified` is UTC
`YYYY-MM-DDTHH:MM:SSZ`.

`source.relative` is the video path relative to the project file's directory
when the video lives inside that directory, otherwise the file name.
`head_sha256` is the SHA-256 of the first 1 MiB. `tail_sha256` is the SHA-256
of the last 1 MiB. A file shorter than 1 MiB uses the whole file for both.
`sha256` is the full-file SHA-256 when the writer computed it, and `""` when
it did not. Opening a project prefers an absolute path whose size and
head/tail hashes match, then the relative path with the same check.

## `settings.json`

Every field of a Clarity job: mode, upscale, anime, upscale strength, sharpen,
white balance, levels, quality factor, sigma, tile, batch, overlap, container
(`mkv` or `mp4`), codec (`av1` or `hevc`), `copy_audio`, and `max_frames`.
`destination` is the output path the window last saved.

## `resume.json`

The chunk journal, so a crash or a cancel can continue:

- `work`: absolute path of the work folder that holds `plan.json`, `journal.jsonl`, and `chunks/`.
- `ranges`: half-open frame ranges, in order.
- `cuts`: scene-cut frame indexes.
- `settings_sha256`: digest of the settings that produced those ranges.
- `source`: absolute source path recorded when the plan was written.
- `journal`: finished chunks `{index, start, stop, sha256}`.

Chunk media stays in the work folder. It is not copied into the ZIP. The work
folder is `<project>.vxclarity.work` beside the project, or
`<output-name>.work` beside the output when no project is open. A chunk file
is resumed only when its bytes still hash to the journal. Changing settings,
the source, or the ranges starts the chunks over.

## `thumb.png`

Optional. A poster still. Writers may omit it.

## Forward compatibility

Readers keep every key they do not understand. Saving a project writes those
keys back out. Adding a field does not require a new major version.

A reader must refuse a higher major version instead of guessing. New required
entries, or a breaking change to the meaning of an existing field, increment
`format_version`.

## File association

Windows: ProgID `Voxprint.Clarity.Project`, per-user `HKCU`, document icon
`branding/clarity.ico`. Uninstall deletes the ProgID key and the
`.vxclarity` class entry.

Linux: shared-mime-info type `application/vnd.voxprint.clarity+zip` with a
glob of `*.vxclarity`, and a `.desktop` file whose `MimeType` is that type.
Removing the installed prefix removes both files. `installer/linux/uninstall.sh`
deletes them from a prefix and refreshes the mime and desktop databases.
