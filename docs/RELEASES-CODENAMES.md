# Release codenames registry

Rule (2026-10-10): every new release codename is a Hebrew word from the Torah or Tanakh describing the build's state of spirit, and it
**must not repeat any codename already used anywhere in the suite**. Before tagging, check this table, then add
the new codename here in the same commit as the `BUILD.json` change. Published past releases keep their names,
including the two historical duplicates below.

`BUILD.json` (`version`, `build`, `offset`, `codename`) is the single source of truth. The release title, About box, splash and
`--version` all show `X.Y.Z RC · build N "Codename"`.

This file is the vendored copy that Clarity's CI checks (`python -m clarity.codenames`).
Clarity does not fetch the other suite repositories. A live cross-repo query is not part of CI.
Before tagging, update the suite table and this copy in the same commit as the `BUILD.json` change.

## Used codenames

| App | Codenames used (oldest first) |
|---|---|
| audiobook-builder | Tikkun, Zahav, Menuchah, Kolot, Kaporet, Shalem, Shelishi, Toledot, Achim, Nachon, Chazak |
| enlarge | Gadol, Hagdil |
| movie-dubber | Hineni, Chazak |
| clarity | Bereshit |

Historical duplicate (published, kept as is): **Chazak** (audiobook-builder and movie-dubber).

## Assigned next codenames (reserved)

| App | Codename | Meaning |
|---|---|---|
| lucent | Zohar | radiance |
| canvas | Yotzer | fashioner (Yeshayahu) |
| model-forge | Charash | craftsman, smith (Yeshayahu 44:12) |
| movie-dubber | Bochan | tested, proven (even bochan, Yeshayahu 28:16) |
| plotweaver | Sefer | book; replaces its unpublished "Bereshit", which is taken by clarity |

When a reserved codename ships, move it to the "Used codenames" table.
