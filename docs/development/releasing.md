# Releasing qmlx-serve

Releases are **manual**. The old automated pipeline (the `auto-release`,
`publish`, `version-check`, and `release-preflight` workflows, plus the
Homebrew tap dispatch) was removed with the rest of the upstream cruft during
the Qwen-only simplification. No bot tags, builds, or publishes for you.
`ci.yml` is the only workflow, and it just runs lint and tests on every PR and
push to `main`.

## Cutting a release

1. **Bump the version.** Edit `version = "X.Y.Z"` in `pyproject.toml` and commit:

   ```bash
   git commit -am "chore: bump version to X.Y.Z"
   ```

   The commit subject is a convention now, not a trigger. Nothing parses it.

2. **Run the clean-room gate** (mandatory):

   ```bash
   make release-smoke          # == python scripts/release_smoke.py
   ```

   It builds the wheel, installs it into a fresh venv with only PyPI deps, then
   imports every module the published entrypoints load (`vllm_mlx`,
   `vllm_mlx.scheduler`, `vllm_mlx.server`, `vllm_mlx.cli`). This catches the
   class of bug where code imports fine against the dev mlx but not against a
   released wheel (#408). Do not publish if it fails.

3. **Build and upload to PyPI:**

   ```bash
   python -m build
   python -m twine upload dist/qmlx_serve-X.Y.Z*
   ```

   You need a PyPI token with upload rights for `qmlx-serve`. Pass it as
   `TWINE_USERNAME=__token__ TWINE_PASSWORD=<token>`, or put it in `~/.pypirc`.
   No token is stored on any server; supply it at upload time.

4. **Tag the release:**

   ```bash
   git tag -a vX.Y.Z -m "qmlx-serve X.Y.Z"
   git push origin main --tags
   ```

5. **Verify against PyPI:**

   ```bash
   python scripts/release_smoke.py --version X.Y.Z
   ```

   Installs `qmlx-serve==X.Y.Z` from PyPI into a clean venv and re-imports the
   release surfaces, confirming the published artifact is actually usable.

## End-user staleness warning

Separate from the release mechanics, `vllm_mlx/_version_check.py` still warns
users on stale installs. `qmlx models` (and any entrypoint that calls
`print_staleness_warning_if_any()`) prints a one-line notice when the installed
version is two or more patches behind the latest GitHub release, on the same
major.minor, when stderr is a TTY, unless `QMLX_DISABLE_VERSION_CHECK` is set.
Cache at `~/.cache/qmlx/version_check.json` (24h TTL, 2s network timeout,
fail-silent on every error path). Contract in `tests/test_version_check.py`.

## Notes

- CI (`ci.yml`) runs lint (ruff + audit) and the test matrix on every PR. It
  does not build or publish anything.
- The old M3 "doctor" gauntlet (`qmlx doctor`, `qmlx bench --tier`, the G1-G11
  gate table, `scripts/release_check_m3.sh`) went with the doctor harness in the
  Qwen-only strip. Those commands no longer exist; `release_check_m3.sh` is a
  leftover and does not run.
