# Release codenames registry

This file is the vendored copy that Clarity's CI checks (`python -m clarity.codenames`). The canonical registry lives in the suite's private project notes; update both in the same change as `BUILD.json`.

Rule (2026-10-10): every new release codename is a Hebrew word from the Torah or Tanakh describing the build's state of spirit, and it
**must not repeat any codename already used anywhere in the suite**. Before tagging, check this table, then add
the new codename here in the same commit as the `BUILD.json` change. Published past releases keep their names,
including the two historical duplicates below.

`BUILD.json` (`offset`, `codename`) is the single source of truth. The release title, About box, splash and
`--version` all show `X.Y.Z RC · build N "Codename"`.

## Used codenames

| App | Codenames used (oldest first) |
|---|---|
| audiobook-builder | Tikkun, Zahav, Menuchah, Kolot, Kaporet, Shalem, Shelishi, Toledot, Achim, Nachon, Chazak |
| enlarge | Gadol, Hagdil, Petach |
| movie-dubber | Hineni, Chazak |
| clarity | Bereshit, Tohar |
| plotweaver | Sefer, Rikmah, Machashavah, Tzaruf |

Historical duplicate (published, kept as is): **Chazak** (audiobook-builder and movie-dubber).

## Assigned next codenames (reserved)

| App | Codename | Meaning |
|---|---|---|
| canvas | Yotzer | fashioner (Yeshayahu) |
| model-forge | Charash | craftsman, smith (Yeshayahu 44:12) |
| movie-dubber | Bochan | tested, proven (even bochan, Yeshayahu 28:16) |

When a reserved codename ships, move it to the "Used codenames" table.
