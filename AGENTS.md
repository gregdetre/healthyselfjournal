# AGENTS

Short guidelines for agents/tools working in this repo.

- For full setup and usage instructions (uv, external venv, editable deps), see:
  - `docs/reference/SETUP.md`

Quick pointers:
- Preferred venv: `/Users/greg/.venvs/experim__examinedlifejournal` (source it before running commands)
- By default, `uv` prefers a project `.venv`, so make sure to pass `--active` to target the active external venv
- Local editable dep: `./gjdutils` tracked via `[tool.uv.sources]`
