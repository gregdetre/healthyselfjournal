# Developer Setup

## Introduction
This guide explains how to set up a development environment for Healthy Self Journal using uv and the shared external virtual environment, with an editable local clone of `gjdutils` for smooth editing in Cursor. It also covers building TypeScript assets and common troubleshooting.

## See also
- `ARCHITECTURE.md` – System dependencies and configuration management overview
- `AGENTS.md` – Quick repo guidance for agents/tools
- `DOCUMENTATION_ORGANISATION.md` – Where to find other docs by topic/persona
- `CLI_COMMANDS.md` – How to run the CLI locally during development
- `AUDIO_VOICE_RECOGNITION_WHISPER.md` – STT backends and requirements (e.g., optional `ffmpeg`)

## Principles and decisions
- Prefer the external virtual environment at `/Users/greg/.venvs/experim__healthyselfjournal`.
- Use uv for dependency management and execution; when targeting the external venv, pass `--active` to uv project commands.
- Keep a local editable clone of `gjdutils` at `./gjdutils` and track it in `pyproject.toml` via `[tool.uv.sources]`.

## Requirements
- Python >= 3.12
- uv installed and on PATH

## Project layout
```
healthyselfjournal/
  ├─ gjdutils/               # local clone (editable)
  ├─ healthyselfjournal/     # package
  ├─ pyproject.toml
  └─ docs/reference/
```

## Setup using the external venv (recommended)
1) Activate the external virtual environment
```bash
source /Users/greg/.venvs/experim__healthyselfjournal/bin/activate
python -V
```

2) Clone the local editable dependency into the project root
```bash
git clone https://github.com/gregdetre/gjdutils.git gjdutils
```

3) Point the project to the local editable source (writes to pyproject)
```bash
uv add --editable ./gjdutils
```
This adds `gjdutils` to `[project.dependencies]` and records its local source in `[tool.uv.sources]`.

4) Sync dependencies into the active (external) venv
```bash
uv sync --active
```

5) Run commands using the active venv
```bash
uv run --active healthyselfjournal -- --help
uv run --active healthyselfjournal journal cli -- --help
```

6) Lock dependencies when ready
```bash
uv lock
```

## Alternative: project-local `.venv`
If you prefer a project-local environment:
```bash
uv venv .venv
uv sync              # no --active needed when using the project env
uv run healthyselfjournal -- --help
```
Note: If both an active external venv and a project `.venv` exist, uv prefers the project `.venv` unless you pass `--active`.

## Front-end assets
The web UI ships with a TypeScript recorder. Install/build assets with `npm` (or your preferred Node package manager):

```bash
# install dev dependencies (TypeScript compiler)
npm install

# one-off compile (emits ES modules under healthyselfjournal/static/js/)
npm run build

# or rebuild on change
npm run watch
```

`npm run build` transpiles `healthyselfjournal/static/ts/app.ts` into `healthyselfjournal/static/js/app.js`. The compiled output is committed, but rerun the build after editing the TypeScript sources.

## Configuration snippet
`pyproject.toml` excerpt:
```toml
[project]
name = "healthyselfjournal"
requires-python = ">=3.12"
dependencies = [
    "gjdutils",
]

[tool.uv.sources]
gjdutils = { path = "./gjdutils", editable = true }
```

## Verification
Confirm that `gjdutils` resolves to the local project clone:
```bash
uv run --active python -c "import gjdutils, pathlib; p=pathlib.Path(gjdutils.__file__).resolve(); print(p)"
# Expect a path like: .../healthyselfjournal/gjdutils/src/gjdutils/__init__.py
```

Smoke-test the CLI entry point:
```bash
uv run --active healthyselfjournal -- --help | cat
uv run --active healthyselfjournal journal cli -- --help | cat
```

## Gotchas and troubleshooting
- uv prefers a project `.venv` over an active external venv unless you pass `--active`.
- Always source the venv activation script (`source .../bin/activate`) so it modifies the current shell.
- If `uv sync` recreates a `.venv` you didn't intend to use, delete it and use `uv sync --active`.
- If imports don’t resolve to `./gjdutils`, resync with `uv sync --active` and re-check `[tool.uv.sources]`.

## Maintenance
- Re-run `uv sync --active` after changing dependencies.
- Run `uv lock` before sharing or deploying to ensure reproducible environments.


## CI builds and packaging (multi‑OS)

Build artifacts are produced with PyInstaller on each OS via GitHub Actions. We do not cross‑compile.

- Workflow: `.github/workflows/build-desktop.yml`
- Triggers: manual (workflow_dispatch) and tags matching `v*`

Local packaging (macOS dev box):

```bash
pip install -U pip wheel setuptools
pip install -e ./gjdutils
pip install pyinstaller
pyinstaller packaging/HealthySelfJournal.spec
# artifacts in ./dist/
```

CI matrix builds (what happens):

- macOS (arm64) on `macos-14` with `--target-arch arm64`
- macOS (x86_64) on `macos-13` with `--target-arch x86_64`
- Windows (x86_64) on `windows-latest`
- Linux (x86_64) on `ubuntu-latest`

Download artifacts:

- Navigate to the workflow run → Artifacts → download the zip for your OS/arch.

Signing/notarisation (macOS):

- For distribution, sign the app bundle with a Developer ID certificate and enable hardened runtime; notarize with Apple, then staple the ticket. Do this after PyInstaller output is produced.

References:

- PyInstaller FAQ (not a cross‑compiler)
- pywebview Packaging guide

