# `.vxclarity` format version 1.0

A Clarity project is a ZIP archive. The container matches the suite document
package frozen in `.vxbook` v1.0 (2026-10-10): the first entry is the media
type, stored, and `manifest.json` uses the same version, creator, source, and
checksum fields. Clarity-only data sits in `x_clarity`.

## Media type

`application/vnd.voxprint.clarity+zip`

The file name ends with `.vxclarity`.

## Layout

| Entry | Compression | Required |
|---|---|---|
| `mimetype` | stored, extra field length 0 | yes, and it is the first entry |
| `manifest.json` | stored or deflated | yes |
| `settings.json` | stored or deflated | yes |
| `resume.json` | stored or deflated | yes |
| `thumb.png` | stored or deflated | no |

Other entries use ZIP method 0 (stored) or 8 (deflate). Names are relative,
use `/`, and do not contain `..`, a leading `/`, a drive letter, `\`, or
control characters. Names are unique even when compared case-insensitively.
Readers ignore an entry they do not know. A listed entry is still checked
against `files` before it is ignored.

`mimetype` is the ASCII bytes `application/vnd.voxprint.clarity+zip` with no
trailing newline and no BOM. The local file header has an empty extra field,
so at byte offset 30 the archive contains
`mimetypeapplication/vnd.voxprint.clarity+zip`.

Writers use one timestamp, the manifest `created` time, and this entry order:
`mimetype`, `manifest.json`, `settings.json`, `resume.json`, `thumb.png`,
then any other entries in name order.

## `manifest.json`

UTF-8 JSON, no BOM, one object. Duplicate keys are rejected.

```json
{
  "format_version": "1.0",
  "created_by": {"app": "VOXPrint AI Clarity", "version": "1.0.0-rc"},
  "generator": {"app": "VOXPrint AI Clarity", "version": "1.0.0-rc"},
  "created": "2026-10-10T08:52:00Z",
  "source": {"format": "mkv", "filename": "in.mkv"},
  "files": {
    "settings.json": {"sha256": "<64 lowercase hex>", "size": 120},
    "resume.json": {"sha256": "<64 lowercase hex>", "size": 80}
  },
  "x_clarity": {
    "product": "vxclarity",
    "modified": "2026-10-10T09:10:00Z",
    "video": {
      "absolute": "/videos/in.mkv",
      "relative": "in.mkv",
      "size": 47065346,
      "head_sha256": "<64 lowercase hex>",
      "tail_sha256": "<64 lowercase hex>",
      "sha256": ""
    }
  }
}
```

Shared fields, same meaning as the suite container:

| Field | Required | Meaning |
|---|---|---|
| `format_version` | yes | `"MAJOR.MINOR"`. This document is `"1.0"`. |
| `created_by` | yes | `{"app", "version"}` of the app that first created the file. |
| `created` | should | ISO 8601 UTC (`2026-10-10T08:00:00Z`). It does not change on later saves. |
| `generator` | may | `{"app", "version"}` of the app that wrote this copy. A re-save updates it and leaves `created_by` as it was. |
| `source` | may | Where the picture came from: `format` (required inside the object), `filename` (the file name only), and `sha256` (64 lowercase hex of the whole file, omitted when it was not computed). |
| `files` | yes | Path to `{"sha256", "size"}` for every entry except `mimetype` and `manifest.json`. |

`x_` fields are for one product. Readers ignore `x_` names they do not know.
Clarity stores the video fingerprint and `modified` in `x_clarity`:

- `x_clarity.video.relative` is the video path relative to the project directory when the video lives in that directory, otherwise the file name.
- `head_sha256` is the SHA-256 of the first 1 MiB. `tail_sha256` is the SHA-256 of the last 1 MiB. A file shorter than 1 MiB uses the whole file for both.
- `sha256` is the full-file digest when the writer computed it, and `""` when it did not. The suite `source.sha256` is set only when that digest exists.
- Opening a project prefers an absolute path whose size and head/tail hashes match, then the relative path with the same check.

## Versioning

`format_version` is `"MAJOR.MINOR"`.

- A minor bump adds optional fields or entries. It does not change the meaning of anything that already exists. A reader accepts any minor of major 1 and ignores unknown fields, entries, and `x_` keys.
- A major bump is a breaking change. A reader refuses a major it does not know, with a message that says to update Clarity.
- Writers emit `"1.0"`. An integer major from a draft file is read as that major with minor 0.

## Checksums

Before use, the reader checks every `files` entry: the name is in the archive,
the byte length equals `size`, and the SHA-256 equals `sha256`. A mismatch
rejects the file. An entry that is present but not listed is ignored, apart
from the name and size limits. `settings.json` and `resume.json` are required,
so a missing listing for either one is an error.

The reader also rejects encrypted entries, compression other than stored or
deflate, symlinks and other special files, names that are not marked UTF-8
when they contain non-ASCII, more than 1,000 entries, more than 512 MiB
uncompressed, a JSON entry over 4 MiB, a compression ratio over 100:1, JSON
nesting deeper than 32, and duplicate keys. The size and ratio limits are
applied to the bytes that actually come out of the inflater, including
entries the reader then ignores.

## `settings.json`

Every field of a Clarity job: mode, upscale, anime, upscale strength, sharpen,
white balance, levels, quality factor, sigma, tile, batch, overlap, container
(`mkv` or `mp4`), codec (`av1` or `hevc`), `crf` (AV1 CRF, default 20; the
same number is the HEVC constant quality), `copy_audio`, and `max_frames`.
`destination` is the output path the window last saved. Unknown keys are kept
and written back.

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

Optional. A poster still. Writers may omit it. When it is present it is listed
in `files`.

## File association

Windows: ProgID `Voxprint.Clarity.Project`, per-user `HKCU`, document icon
`branding/clarity.ico`. The visible type name is **VOXPrint AI Clarity project**.
Open runs `voxprint-clarity.exe`. Uninstall deletes the ProgID key and the
`.vxclarity` class entry.

Linux: shared-mime-info type `application/vnd.voxprint.clarity+zip` with a
glob of `*.vxclarity`, and `ai.voxprint.Clarity.desktop` whose `MimeType` is
that type and whose `Name` is **VOXPrint AI Clarity**.
`installer/linux/uninstall.sh` deletes the association and the desktop file
and refreshes the mime and desktop databases. It does not delete
`~/.local/share/voxprint`.
