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
| enlarge | Gadol, Hagdil, Petach, Bahir |
| movie-dubber | Hineni, Chazak, Bochan |
| clarity | Bereshit, Tohar, Arukhah |
| plotweaver | Sefer, Rikmah, Machashavah, Tzaruf |
| lucent | Zohar, Ner |

Historical duplicate (published, kept as is): **Chazak** (audiobook-builder and movie-dubber).

## Assigned next codenames (reserved)

| App | Codename | Meaning |
|---|---|---|
| canvas | Yotzer | fashioner (Yeshayahu) |
| model-forge | Charash | craftsman, smith (Yeshayahu 44:12) |
| audiobook-builder | Tamim | whole, complete (Bereshit 17:1, Devarim 18:13); 1.0.0 RC3 build 1001 |
| lucent | Yashar | upright, straight (Devarim 6:18); fixes from the first VM test |
| enlarge | Marpe | healing, remedy (Mishlei 4:22); fixes from the first GPU test |

When a reserved codename ships, move it to the "Used codenames" table.
