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
