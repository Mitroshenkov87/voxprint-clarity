# Release codenames registry

This file is the vendored copy that Clarity's CI checks (`python -m clarity.codenames`). The canonical registry lives in the suite's private project notes; update both in the same change as `BUILD.json`.

Rule (2026-10-10): every new release codename is a Hebrew word from the Torah or Tanakh describing the build's state of spirit, and it
**must not repeat any codename already used anywhere in the suite**. Before tagging, check this table, then add
the new codename here in the same commit as the `BUILD.json` change. Published past releases keep their names,
including the two historical duplicates below.

`BUILD.json` (`offset`, `codename`) is the single source of truth. The release title, the window footer, the splash and
`--version` all show the `voxprint_common.buildinfo` label. A plain version is `X.Y.Z build N "Codename"`. A version string that contains `-rc` is `X.Y.Z RC build N "Codename"`.

## Used codenames

| App | Codenames used (oldest first) |
|---|---|
| audiobook-builder | Tikkun, Zahav, Menuchah, Kolot, Kaporet, Shalem, Shelishi, Toledot, Achim, Nachon, Chazak, Tamim, Ometz, Amud, Shomer |
| enlarge | Gadol, Hagdil, Petach, Bahir, Marpe, Tikvah, Chayil, Emunah |
| movie-dubber | Hineni, Chazak, Bochan, Kumi |
| clarity | Bereshit, Tohar, Arukhah, Emet, Barur |
| plotweaver | Sefer, Rikmah, Machashavah, Tzaruf, Chut, Oreg, Shti |
| lucent | Zohar, Ner, Yashar, Tov, Orah |
| canvas | Yotzer |
| model-forge | Charash |
| voxprint-common | Yesod, Chesed, Gevurah, Tiferet, Netzach, Hod, Malkhut |

Historical duplicate (published, kept as is): **Chazak** (audiobook-builder and movie-dubber). **Menuchah** is also the codename of voxprint-common 0.1.1.

Barur is clear (Iyov 33:3). Clarity build 1005.

## Assigned next codenames (reserved)

No codename is reserved. Yotzer, Charash, Tamim, Yashar, and Marpe have shipped and are in the used table.

When a reserved codename ships, move it to the "Used codenames" table.
