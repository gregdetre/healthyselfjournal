# PyPI publishing (healthyselfjournal)

## Introduction

This document describes the end‑to‑end process for packaging and publishing the `healthyselfjournal` CLI to TestPyPI and PyPI using Hatch (build backend) and Twine. It is evergreen guidance: concise, complete‑enough, and updated as the release process evolves.

## See also

- `../planning/250917c_publish_to_pypi.md` – implementation plan, acceptance criteria, commands, risks
- `./SETUP_DEV.md` – preferred venv + uv workflow and local development setup
- `./CLI_COMMANDS.md` – CLI usage; helpful for post‑install validation
- `./LLM_PROMPT_TEMPLATES.md` – prompt assets that must ship inside the wheel
- `./FILE_FORMATS_ORGANISATION.md` – runtime files created by the app; not packaged
- `./DESKTOP_APP_PYWEBVIEW.md` – Desktop app packaging & release steps
- External: https://packaging.python.org/en/latest/ (Packaging guide), https://test.pypi.org/ (TestPyPI), https://twine.readthedocs.io/en/stable/ (Twine), https://docs.pypi.org/trusted-publishers/ (Trusted Publishing)

## Current state

- Build backend: Hatch (`hatchling.build`) configured in `pyproject.toml`.
- Wheel packaging: includes Python modules plus required runtime assets:
  - Jinja prompts: `healthyselfjournal/prompts/*.jinja`
  - Static assets for the web UI: `healthyselfjournal/static/{css,js}/*`
  - Web templates for the web UI: `healthyselfjournal/web/templates/*.jinja`
- CLI entry point: `healthyselfjournal` → `healthyselfjournal.__main__:app` (Typer app defined in `healthyselfjournal/cli.py`).
- Dependencies: pinned lower bounds; notable constraints include `python-fasthtml>=0.12,<0.13` and `fastcore>=1.6.7,<2.0` (kept compatible with `python-fasthtml`).
- Python requirement: `>=3.10` (with `tomli` fallback for TOML parsing on <3.11).
- Dev convenience: `[tool.uv.sources] gjdutils = { path = "./gjdutils", editable = true }` for local work; published wheels depend on the public `gjdutils` (`>=0.6.1`).
- Verified: build + smoke tests pass; published to TestPyPI and validated via a fresh venv.

## Principles and key decisions

- Use standard build+upload tooling (Hatch + Twine); avoid repo‑specific publishers.
- Keep local dev ergonomic (editable `gjdutils`) without leaking into wheels.
- Bundle all required runtime assets (prompts, web static) into the wheel.
- Avoid importing heavy/optional web dependencies at CLI startup (lazy import in `web` command) so `healthyselfjournal --help` works in minimal environments.
- Prefer `uv` tooling for build and `uvx` for quick smoke testing.

## How to: publish a release

### 1) Preflight checks

- Ensure a clean working tree and passing tests.
- Confirm version bump in `pyproject.toml` → `[project] version` (single source of truth).
- Verify dependencies are correct (no local path deps in `[project.dependencies]`).
- Optional: update `README.md` and relevant reference docs.

### 2) Build distributions

```bash
uv build
# Outputs under dist/: healthyselfjournal-<version>-py3-none-any.whl and .tar.gz
```

### 3) Verify wheel contents and entry point

```bash
# Inspect wheel contents for assets
unzip -l dist/*.whl | rg -n "prompts/|static/|web/templates/|RECORD|healthyselfjournal/__main__|healthyselfjournal/cli.py"

# Smoke test console script directly from the wheel (no install)
uvx --from dist/*.whl healthyselfjournal -- --help

# Verify prompt assets load at runtime (ensures bundling is correct)
uvx --from dist/*.whl python -c "import healthyselfjournal.llm as m; print(m._load_prompt('question.prompt.md.jinja')[:40])"
```

Expected: prompts and static files present; `--help` renders; prompt snippet prints.

### 4) Upload to TestPyPI

Credentials: configure `~/.pypirc` with TestPyPI and PyPI tokens (never commit secrets):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-***

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-***
```

Upload:

```bash
uvx twine upload -r testpypi dist/*.whl dist/*.tar.gz
```

### 5) Validate TestPyPI install

```bash
python -m venv /tmp/hsj-testpypi && /tmp/hsj-testpypi/bin/pip \
  install --index-url https://test.pypi.org/simple/ \
          --extra-index-url https://pypi.org/simple/ \
          healthyselfjournal==<version>

/tmp/hsj-testpypi/bin/healthyselfjournal --help
```

Optional: verify prompt asset load in that venv as above.

### 6) Upload to PyPI

- Bump version if you re‑built.
- Ensure the git tree matches the release.

```bash
uvx twine upload dist/*.whl dist/*.tar.gz
```

Validate from a clean machine:

```bash
# Run without installing, using uv's shim
uvx healthyselfjournal -- --help

# Optionally pin a version
uvx healthyselfjournal==<version> -- --help

# If users are on older Python, suggest an explicit version
uvx --python 3.12 healthyselfjournal -- --help
```

## Troubleshooting and gotchas

- CLI fails on `--help` due to FastHTML import errors:
  - Symptom: `ValueError: not enough values to unpack` during import from `fasthtml` when merely running `--help`.
  - Resolution: keep the web server import lazy inside the `web` command. Implemented in `healthyselfjournal/cli.py`.

- Missing prompt/static assets in wheel:
  - Ensure `pyproject.toml` includes a `[tool.hatch.build.targets.wheel.force-include]` section mapping prompts and static files.
  - Rebuild and re‑inspect the wheel via `unzip -l dist/*.whl | rg -n "prompts/|static/"`.

- Local `gjdutils` path dependency accidentally embedded:
  - Keep `[tool.uv.sources]` for development only; published dependencies must be regular PyPI specifiers (e.g., `gjdutils>=0.6.1`).
  - Validate with `python -m build` or `uv build` and inspect the generated `METADATA`/`RECORD`.

- Python version mismatches on user systems:
  - Project requires Python `>=3.10`. Document `uvx --python 3.12 healthyselfjournal` (or your preferred version) for easy runs on older default interpreters.

- Platform/system deps:
  - `ffmpeg` is optional (for background MP3 conversion) and is not bundled. Document as an external prerequisite.

- Dependency constraints drift:
  - `python-fasthtml` currently constrained to `<0.4`; ensure `fastcore` remains compatible (e.g., `fastcore>=1.5,<1.6`). Review constraints before release.

## Versioning and release hygiene

- Version lives in `pyproject.toml` → `[project] version`. Code reads it at runtime via `importlib.metadata` (computed `__version__`), with a local fallback like `0.0.0+local` when not installed. Bump before building.
- Tagging: `git tag v<version>` after a successful release (optional but recommended).
- Changelog: keep concise release notes (consider adding `CHANGELOG.md`).
- Commits: follow `gjdutils/docs/instructions/GIT_COMMIT_CHANGES.md` guidance for clean, typed messages.

## Future improvements (target state)

- CI Trusted Publishing
  - Use GitHub Actions with PyPI Trusted Publishing to avoid local tokens.
  - Build on tag; run smoke tests; publish to PyPI on success.
  - References: `pypa/gh-action-pypi-publish`, PyPI Trusted Publishers docs.

- Broader artifact checks
  - Add automated checks that the wheel contains prompts/static and that `uvx` smoke tests pass post‑publish.

## Release checklist (copy/paste)

- [x] Bump version in `pyproject.toml`
- [x] Update `CHANGELOG.md` with highlights for this release
- [x] Commit version and changelog (clean working tree)
- [x] `uv build` completes; wheel and sdist produced
- [x] Wheel contains `prompts/`, `static/`, and `web/templates/` assets
- [x] `uvx --from dist/*.whl healthyselfjournal -- --help` works
- [x] Prompt asset load smoke test passes
- [x] Push commit to main: `git push origin main`
- [x] `uvx twine upload dist/*.whl dist/*.tar.gz` publishes to PyPI
- [ ] `uvx healthyselfjournal -- --help` works on a clean machine
- [ ] Pin-run check: `uvx healthyselfjournal==<version> -- --help`
- [ ] Tag and push (triggers desktop CI build): `git tag v<version> && git push origin v<version>`
- [ ] (Optional) Create a GitHub release from the tag
- [ ] Docs updated (`README.md`, `CLI_COMMANDS.md`) if flags/flows changed
- [ ] (If enabled) CI Trusted Publishing workflow runs and succeeds
- [ ] If distributing the desktop app, follow the Desktop app release checklist in `docs/reference/DESKTOP_APP_PYWEBVIEW.md`

## Quick checklist for a new version

Short, practical steps for routine releases (example bumps 0.2.0 → 0.2.1):

1. Edit `pyproject.toml` → `[project] version = "0.2.1"`
2. Update `CHANGELOG.md` with a `0.2.1` section
3. Commit both: `git add pyproject.toml CHANGELOG.md && git commit -m "chore: bump to 0.2.1; docs: changelog"`
4. Build: `uv build`
5. Inspect wheel: `unzip -l dist/*.whl | rg -n "prompts/|static/|web/templates/"`
6. Local smoke test:
   - `uvx --from dist/*.whl healthyselfjournal -- --help`
   - `uvx --from dist/*.whl python -c "import healthyselfjournal.llm as m; print(m._load_prompt('question.prompt.md.jinja')[:40])"`
7. Upload:
   - TestPyPI (optional): `uvx twine upload -r testpypi dist/*.whl dist/*.tar.gz`
   - PyPI: `uvx twine upload dist/*.whl dist/*.tar.gz`
8. Validate from PyPI:
   - `uvx healthyselfjournal==0.2.1 -- --help`
   - Optional: `uvx --python 3.12 healthyselfjournal==0.2.1 -- --help`
9. Push/tag (triggers desktop build): `git push origin main && git tag v0.2.1 && git push origin v0.2.1`
10. If shipping the desktop app, follow the Desktop app release checklist in `docs/reference/DESKTOP_APP_PYWEBVIEW.md`

### Helper script

For routine releases, you can use the helper script which wraps the steps above. It assumes you've already edited and committed the version and changelog.

```bash
# TestPyPI example (no git tag/push):
scripts/release.sh -v <version> -r testpypi --validate --no-git

# PyPI example (with tag/push):
scripts/release.sh -v <version> --validate
```

Flags:
- `-v, --version`: the version (must match `pyproject.toml` and exist in `CHANGELOG.md`)
- `-r, --repo`: `pypi` (default) or `testpypi`
- `--validate`: runs a simple `uvx healthyselfjournal==<version> -- --help`
- `--no-git`: skips tagging/pushing
- `--skip-clean-check`: allow a dirty working tree (not recommended)
