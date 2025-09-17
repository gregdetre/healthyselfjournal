# SETUP

## Introduction
This document explains how to set up and work on this project using uv, with a focus on using the external virtual environment and a local editable clone of `gjdutils` inside the project root for easy editing in Cursor.

## See also
- `AGENTS.md` – brief signpost for agents/tools working in this repo.
- Evergreen guidance – structure and maintenance approach for evergreen docs: [WRITE_EVERGREEN_DOC.md](https://raw.githubusercontent.com/gregdetre/gjdutils/refs/heads/main/docs/instructions/WRITE_EVERGREEN_DOC.md)

## Principles and decisions
- Prefer the external virtual environment at `/Users/greg/.venvs/experim__examinedlifejournal`.
- Use uv for dependency management and execution; when targeting the external venv, pass `--active` to uv project commands.
- Keep a local editable clone of `gjdutils` at `./gjdutils` and track it in `pyproject.toml` via `[tool.uv.sources]`.

## Requirements
- Python >= 3.12
- uv installed and on PATH

## Project layout
```
examinedlifejournal/
  ├─ gjdutils/               # local clone (editable)
  ├─ examinedlifejournal.py
  ├─ pyproject.toml
  └─ docs/reference/SETUP.md
```

## Setup using the external venv (recommended)
1) Activate the external virtual environment
```bash
source /Users/greg/.venvs/experim__examinedlifejournal/bin/activate
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
uv run --active python examinedlifejournal.py --duration 3
uv run --active ipython
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
uv run python examinedlifejournal.py --duration 3
```
Note: If both an active external venv and a project `.venv` exist, uv prefers the project `.venv` unless you pass `--active`.

## Configuration snippet
`pyproject.toml` excerpt:
```toml
[project]
name = "examinedlifejournal"
version = "0.1.0"
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
# Expect a path like: .../examinedlifejournal/gjdutils/src/gjdutils/__init__.py
```

## Gotchas and troubleshooting
- uv prefers a project `.venv` over an active external venv unless you pass `--active`.
- Always source the venv activation script (`source .../bin/activate`) so it modifies the current shell.
- If `uv sync` recreates a `.venv` you didn't intend to use, delete it and use `uv sync --active`.
- If imports don’t resolve to `./gjdutils`, resync with `uv sync --active` and re-check `[tool.uv.sources]`.

## Maintenance
- Re-run `uv sync --active` after changing dependencies.
- Run `uv lock` before sharing or deploying to ensure reproducible environments.

