### Goal, context

- Enable zero-flags execution via `uvx examinedlifejournal` for users.
- Publish the `examinedlifejournal` package to PyPI (TestPyPI first), following a similar approach to `gjdutils` where sensible.
- Keep local dev ergonomics (editable local `gjdutils` via `[tool.uv.sources]`) while producing a clean distributable wheel for end users.


### References

- `pyproject.toml` (root): current metadata, `[project.scripts]` entry, `dependencies`, `[tool.uv.sources]` for `gjdutils`.
- `examinedlifejournal/__main__.py`: CLI entrypoint calling `examinedlifejournal.cli:app`.
- `examinedlifejournal/llm.py`: loads prompt templates via `PROMPTS_DIR = Path(__file__).parent / "prompts"` → ensure prompt files ship in the wheel.
- `examinedlifejournal/prompts/*.jinja`: runtime assets required by `llm.py`.
- `gjdutils/src/gjdutils/cli/pypi/{app.py, check.py, deploy.py}` and `src/gjdutils/pypi_build.py`: working PyPI flows for `gjdutils` (opinionated, but tightly coupled to `gjdutils`).
- Repo rule doc: `docs/reference/SETUP.md` (venv + `uv` usage), `docs/reference/COMMAND_LINE_INTERFACE.md` (CLI expectations).
- External: Standard packaging flow (Hatch build backend, `python -m build`, `twine upload`), `uvx` usage for running packages.


### Principles, key decisions

- Use standard packaging (Hatch + Build + Twine). Follow `gjdutils` where it helps, but avoid reusing its CLI since it’s hard-coded to `gjdutils` metadata.
- Preserve local dev UX: keep `[tool.uv.sources] gjdutils = { path = "./gjdutils", editable = true }` for development, but ensure published wheel depends on public `gjdutils` (PyPI or VCS URL).
- Keep Python requirement `>=3.12`. Document `uvx --python 3.12` for users on older interpreters.
- Ship all required runtime assets (prompt templates) inside the wheel to avoid `FileNotFoundError` at runtime.
- Publish first to TestPyPI, verify end-to-end install + CLI run, then publish to PyPI.
- Provide clear user docs for install/run and required env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` for specific modes) and optional system deps (`ffmpeg`).


### Stages & actions

#### Stage: Prepare packaging configuration
- [ ] Add a build backend to `pyproject.toml` (Hatch):
  - `[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"`.
  - `[tool.hatch.build.targets.wheel] packages = ["examinedlifejournal"]`.
  - Ensure non-Python assets are included (see next action).
- [ ] Ensure prompt templates are included in wheels:
  - Add Hatch build config to include `examinedlifejournal/prompts/*.jinja` (e.g., `tool.hatch.build.targets.wheel.force-include` mapping, or broader include for the package directory).
  - Acceptance: fresh wheel contains both `question.prompt.md.jinja` and `summary.prompt.md.jinja` paths under the package.
- [ ] Review and finalize `project` metadata:
  - `name`, `version`, `description`, `readme`, `license`, `classifiers`, `urls`.
  - Keep `requires-python = ">=3.12"`.
  - Optional extras (if any) to group optional features.

#### Stage: Dependencies and local overrides
- [ ] Ensure `gjdutils` dependency resolves for published artifacts:
  - Prefer `gjdutils>=X.Y` from PyPI (gjdutils is already on PyPI), keeping `[tool.uv.sources]` for local dev only.
  - Alternative: PEP 508 VCS URL `gjdutils @ git+https://github.com/gregdetre/gjdutils` if we need unreleased features.
  - Acceptance: building with `python -m build` does not embed a local path dependency.

#### Stage: Build and local validation
- [ ] Clean builds and produce distributions:
  - `uv build` (or `python -m build`) → `dist/*.whl`, `dist/*.tar.gz`.
- [ ] Verify wheel contents include prompts and console script:
  - Inspect wheel (e.g., `unzip -l dist/*.whl | rg prompts/`), check that entry point `examinedlifejournal` exists in `RECORD`.
- [ ] Smoke-test from wheel with `uvx --from dist/*.whl examinedlifejournal -- --help`.
- [ ] Verify runtime prompt load without network/API:
  - `uvx --from dist/*.whl python -c "import examinedlifejournal.llm as m; print(m._load_prompt('question.prompt.md.jinja')[:40])"` (ensures assets are bundled).

#### Stage: TestPyPI publish and validation
- [ ] Configure credentials for TestPyPI (`~/.pypirc` or env vars for `twine`).
- [ ] Upload to TestPyPI: `twine upload -r testpypi dist/*`.
- [ ] Validate install from TestPyPI in a temp venv:
  - `python -m venv /tmp/eljournal-testpypi && /tmp/eljournal-testpypi/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ examinedlifejournal`.
  - Run `/tmp/.../bin/examinedlifejournal --help` and the prompt asset check above.

#### Stage: PyPI publish and validation
- [ ] Bump version if needed, ensure git clean, tag workflow (optional).
- [ ] Upload to PyPI: `twine upload dist/*`.
- [ ] Validate with `uvx examinedlifejournal -- --help` (and optionally `uvx --python 3.12 examinedlifejournal`).
- [ ] Optional: pin run check `uvx examinedlifejournal==<version>`.

#### Stage: Documentation & follow-ups
- [ ] Update `README.md` with install/run snippets:
  - `uvx examinedlifejournal`
  - Notes on Python 3.12, env vars, optional `ffmpeg`.
- [ ] Update `docs/reference/SETUP.md` and `COMMAND_LINE_INTERFACE.md` with PyPI install notes.
- [ ] (Optional) Add CI job for publishing on tag, using trusted publishing or API token.
- [ ] (Optional) Add `CHANGELOG.md` and release notes workflow.


### Notes on reusing `gjdutils` PyPI scripts

- The `gjdutils` CLI under `src/gjdutils/cli/pypi/` is tightly coupled to `gjdutils` (hard-coded package name in checks, metadata queries, install commands, and version existence checks).
- Reusing as-is would require generalization (package name parameterization, metadata discovery), which adds maintenance and risk.
- Recommendation: use the standard Build + Twine flow for `examinedlifejournal` now. If we later want a reusable publisher, we can extract a generic helper.


### Risks & mitigations

- Missing prompt assets in wheel → Explicitly include via Hatch build config and verify by inspecting the wheel and running a prompt-load smoke test.
- `gjdutils` resolution issues → Depend on PyPI release of `gjdutils` (or pinned VCS URL) and keep `[tool.uv.sources]` for dev-only convenience.
- Platform/system deps (`ffmpeg`, audio libs) → Document clearly; they’re optional or used only in specific modes.
- Python version mismatch on user systems → Document `uvx --python 3.12` usage.


### Acceptance criteria (overall)

- Running `uvx examinedlifejournal -- --help` works on a clean machine with only Python and `uv` installed.
- `examinedlifejournal` installs and runs from TestPyPI and PyPI.
- Prompt templates load at runtime from the installed wheel.
- README and reference docs updated with clear install/run instructions and prerequisites.


