# Changelog

## Unreleased

- Pin `voxprint-engine[gpu]` at tag `engine-v1.4.0`. API stays 1.
- Pin `voxprint-common` 0.1.1 "Menuchah" from the release wheel. The installer
  checks the wheel against the published `.sha256` file.
- Import paths, the GPU lock, the thermal rule, and the GPU probe from
  `voxprint_common`. Call `enlarge.suite_runtime.ensure_runtime("clarity")`.
  That call writes the app id and does not install Enlarge's default model.
- Pass the display name `Voxprint AI Clarity` to the GPU probe.
- Keep a missing build number as null. The version label says `dev`.
  When `GITHUB_RUN_NUMBER` is set, the build is that number plus `offset`.
