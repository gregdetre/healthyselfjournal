### Goal, context

- Enable zero-flags execution via `uvx healthyselfjournal` for users.
- Publish the `healthyselfjournal` package to PyPI (TestPyPI first), following a similar approach to `gjdutils` where sensible.
- Keep local dev ergonomics (editable local `gjdutils` via `[tool.uv.sources]`) while producing a clean distributable wheel for end users.


### References

- `pyproject.toml` (root): current metadata, `[project.scripts]` entry, `dependencies`, `[tool.uv.sources]` for `gjdutils`.
- `healthyselfjournal/__main__.py`: CLI entrypoint calling `healthyselfjournal.cli:app`.
- `healthyselfjournal/llm.py`: loads prompt templates via `PROMPTS_DIR = Path(__file__).parent / "prompts"` → ensure prompt files ship in the wheel.
- `healthyselfjournal/prompts/*.jinja`: runtime assets required by `llm.py`.
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
- [x] Add a build backend to `pyproject.toml` (Hatch):
  - `[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"`.
  - `[tool.hatch.build.targets.wheel] packages = ["healthyselfjournal"]`.
  - Ensure non-Python assets are included (see next action).
- [x] Ensure prompt templates are included in wheels:
  - Added Hatch force-include for `healthyselfjournal/prompts/*.jinja` and web static under `healthyselfjournal/static/{css,js}`.
  - Acceptance: verified in built wheel.
- [x] Review and finalize `project` metadata:
  - `requires-python = ">=3.12"` kept. `gjdutils>=0.6.1` set; added `fastcore>=1.5,<1.6`; constrained `python-fasthtml>=0.3,<0.4`.

#### Stage: Dependencies and local overrides
- [x] Ensure `gjdutils` dependency resolves for published artifacts:
  - Using `gjdutils>=0.6.1` from PyPI; `[tool.uv.sources]` retained for local dev.
  - Acceptance: `uv build` wheel shows no local path deps.

#### Stage: Build and local validation
- [x] Clean builds and produce distributions:
  - `uv build` → `dist/*.whl`, `dist/*.tar.gz`.
- [x] Verify wheel contents include prompts and console script:
  - Inspected wheel; prompts and static present; console script works.
- [x] Smoke-test from wheel with `uvx --from dist/*.whl healthyselfjournal -- --help`.
- [x] Verify runtime prompt load without network/API:
  - Confirmed prompt text loads from installed wheel.

#### Stage: TestPyPI publish and validation
- [x] Configure credentials for TestPyPI (`~/.pypirc` or env vars for `twine`).
- [x] Upload to TestPyPI: `twine upload -r testpypi dist/*`.
- [x] Validate install from TestPyPI in a temp venv:
  - Installed and ran `healthyselfjournal --help`; prompt asset load verified.

#### Stage: PyPI publish and validation
- [ ] Bump version if needed, ensure git clean, tag workflow (optional).
- [ ] Upload to PyPI: `twine upload dist/*`.
- [ ] Validate with `uvx healthyselfjournal -- --help` (and optionally `uvx --python 3.12 healthyselfjournal`).
- [ ] Optional: pin run check `uvx healthyselfjournal==<version>`.

#### Stage: Documentation & follow-ups
- [x] Update `README.md` with install/run snippets:
  - `uvx healthyselfjournal` examples added; PyPI flow documented.
- [x] Update `docs/reference/SETUP.md` and `COMMAND_LINE_INTERFACE.md` with PyPI install notes.
- [ ] (Optional) Add CI job for publishing on tag, using trusted publishing or API token.
- [ ] (Optional) Add `CHANGELOG.md` and release notes workflow.


### Notes on reusing `gjdutils` PyPI scripts

- The `gjdutils` CLI under `src/gjdutils/cli/pypi/` is tightly coupled to `gjdutils` (hard-coded package name in checks, metadata queries, install commands, and version existence checks).
- Reusing as-is would require generalization (package name parameterization, metadata discovery), which adds maintenance and risk.
- Recommendation: use the standard Build + Twine flow for `healthyselfjournal` now. If we later want a reusable publisher, we can extract a generic helper.


### Risks & mitigations

- Missing prompt assets in wheel → Explicitly include via Hatch build config and verify by inspecting the wheel and running a prompt-load smoke test.
- `gjdutils` resolution issues → Depend on PyPI release of `gjdutils` (or pinned VCS URL) and keep `[tool.uv.sources]` for dev-only convenience.
- Platform/system deps (`ffmpeg`, audio libs) → Document clearly; they’re optional or used only in specific modes.
- Python version mismatch on user systems → Document `uvx --python 3.12` usage.


### Acceptance criteria (overall)

- Running `uvx healthyselfjournal -- --help` works on a clean machine with only Python and `uv` installed.
- `healthyselfjournal` installs and runs from TestPyPI and PyPI.
- Prompt templates load at runtime from the installed wheel.
- README and reference docs updated with clear install/run instructions and prerequisites.


