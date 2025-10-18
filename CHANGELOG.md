## 0.2.9 - 2025-10-18

- Publishing docs: clarified release order and added web template asset check
- Packaging: ensure web templates verified in wheel inspection
- Minor: dependency constraints documented to match `pyproject.toml`

## 0.2.8 - 2025-10-18

- Local LLM: realistic prompt mode and per-run model override in diagnose
- CLI: add `diagnose local compare` subcommand; refine init/init_app UX
- Model manager: more robust model path resolution and cache safety
- Packaging: wheel verified (prompts/static bundled); `uvx` smoke tests pass
- Published to PyPI and tagged `v0.2.8`

## 0.2.5 - 2025-10-17

- Add `version` command to CLI (`healthyselfjournal -- version`)
- Publish to PyPI; wheel verified and prompt/static assets intact
- Validation note: pin to `==0.2.5` once PyPI indexing completes

## 0.2.4 - 2025-10-17

- Publish to PyPI; `uvx healthyselfjournal -- --help` validated
- Package now includes `healthyselfjournal/prompts/insights.prompt.md.jinja`
- Verified wheel assets: prompts and static files bundled; entrypoint present
- Minor prompt and web UI script updates

## 0.2.1 - 2025-09-20

- Publish to PyPI with bundled prompts and static assets verified
- Clarified CLI grouping; ensured `--help` works without web deps
- Prep for desktop build in CI: add macOS PyInstaller workflow (unsigned)

# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-19

### Added
- Published project to PyPI; `uvx healthyselfjournal -- --help` works out of the box.
- Web UI static assets bundled in wheel (`healthyselfjournal/static/{css,js}`).
- Quick release checklist in `docs/reference/PYPI_PUBLISHING.md`.

### Changed
- Switched to Hatch build backend; explicit inclusion of prompt assets.
- Lazy import for web server to avoid FastHTML import at CLI startup.
- Dependencies: `gjdutils>=0.6.1`, `python-fasthtml>=0.3,<0.4`, `fastcore>=1.5,<1.6`.

### Fixed
- TestPyPI validation and prompt asset load smoke test confirmed.

[0.2.0]: https://pypi.org/project/healthyselfjournal/0.2.0/
 [0.2.4]: https://pypi.org/project/healthyselfjournal/0.2.4/
 [0.2.8]: https://pypi.org/project/healthyselfjournal/0.2.8/
