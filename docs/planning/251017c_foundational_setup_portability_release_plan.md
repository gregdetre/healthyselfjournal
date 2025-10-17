### Goal, context

Raise reliability and portability for a shareable “friends can run it” release by unifying versioning, cleaning init/config flows (CWD‑first), setting sane defaults (thinking enabled), clarifying platform support (Windows primary, Linux nice‑to‑have), relaxing the Python floor to >=3.10, improving CLI messaging (no user‑specific paths), and tightening documentation (PyPI, setup, uv/uvx, audio deps). No code changes are made in this doc; it is the plan of record.


### References

- `pyproject.toml` – Single declared version, Python requirement, dependencies, wheel includes
- `healthyselfjournal/__init__.py` – Env autoload; exposes `__version__` used in frontmatter
- `healthyselfjournal/cli.py` – Typer root; dependency checks; contains a hard‑coded venv path
- `healthyselfjournal/session.py` – Writes `app_version` into session frontmatter
- `healthyselfjournal/llm.py` – Anthropic thinking support and budgets
- `healthyselfjournal/config.py` – Defaults, env reading, user_config overrides
- `docs/reference/PYPI_PUBLISHING.md` – Publishing steps (should reference single source of version)
- `docs/reference/SETUP_USER.md` – User setup; should emphasize uv/uvx and platform notes
- `docs/reference/CONFIGURATION.md`, `docs/reference/INIT_FLOW.md` – Precedence and wizard behavior
- `AGENTS.md` and other docs that may include user‑specific paths


### Principles, key decisions

- Single source of truth for version
  - Version only declared in `pyproject.toml`. Runtime value read via `importlib.metadata`.
  - Keep `__version__` as a computed alias (no hardcoded string) or call `importlib.metadata` at use sites.

- CWD‑first configuration for CLI/init
  - Default persistence to CWD `.env.local` (works well with `uvx` and is unsurprising for non‑technical users).
  - Optional XDG flag for power users; desktop can continue to use XDG settings.

- Defaults and UX
  - Anthropic “thinking” enabled by default for questions; document budget semantics.
  - Windows supported out‑of‑the‑box; Linux support notes as a nice‑to‑have.

- Compatibility and tooling
  - Python >=3.10 (add `tomli` for <3.11 TOML parsing if needed).
  - Prefer `uvx`/`uv run` for users and developers (venv‑less dev path). Provide neutral venv example if desired; never use user‑specific paths.


### Stages & actions

#### Versioning: single source of truth and runtime access
- [x] Compute `__version__` from installed package metadata
  - In `healthyselfjournal/__init__.py`:
    - `from importlib.metadata import PackageNotFoundError, version as _pkg_version`
    - `try: __version__ = _pkg_version("healthyselfjournal")`
      `except PackageNotFoundError: __version__ = "0.0.0+local"`
  - Acceptance: No hardcoded version anywhere in code; `session` frontmatter shows the correct version when installed; dev checkout uses `0.0.0+local`.
- [x] Align any divergent versions
  - Ensure `pyproject.toml` `[project].version` matches release intent.
  - Acceptance: no mismatch between wheel metadata, runtime `__version__`, and docs.
- [x] Test coverage
  - Add a small test asserting frontmatter `app_version` is non‑empty and uses fallback when distribution metadata is absent.
- [x] Docs: publishing guidance
  - In `docs/reference/PYPI_PUBLISHING.md`, replace duplicated versioning guidance with: “Version lives in `pyproject.toml`; code reads it at runtime via `importlib.metadata` (computed `__version__`).”

#### Init wizard and configuration precedence (CWD‑first)
- [x] Keep default writes to CWD `.env.local` in the wizard
  - Acceptance: Running `uvx healthyselfjournal -- init` writes to the working directory.
- [x] Optional `--xdg` flag
  - Implement an opt‑in flag to persist keys to `~/.config/healthyselfjournal/.env.local`.
  - Acceptance: With `--xdg`, file is created under XDG; precedence remains CLI > OS env > CWD `.env.local` > `.env` > defaults (desktop additionally honors XDG settings).
- [x] Docs: update precedence narratives
  - `CONFIGURATION.md` and `INIT_FLOW.md` emphasize CWD‑first for CLI; mention optional XDG flag and desktop’s XDG settings.
- [x] Tests
  - Cover reading from CWD `.env.local` and ignoring XDG unless requested.

#### CLI messaging: remove user‑specific venv path; promote uv/uvx
- [x] Replace hard‑coded venv path in `healthyselfjournal/cli.py`
  - Show `sys.executable`, suggest `uvx healthyselfjournal -- journal cli`, and include a neutral venv example (no user paths).
  - Acceptance: Error copy is portable and actionable on Windows/macOS/Linux.
- [x] `SETUP_USER.md`
  - Emphasize `uvx` as default; add minimal venv snippet as optional for devs.

#### LLM defaults: thinking enabled by default
- [x] Ensure the default LLM model spec in `config.py` has “thinking” enabled for Anthropic
  - Acceptance: Out‑of‑box question generation uses thinking; budget semantics documented; non‑Anthropic providers do not claim thinking.
- [ ] Docs
  - Add a short note describing the thinking budget behavior and how to override.

#### Python version floor and dependencies
- [x] Relax Python to `>=3.10`
  - Update `pyproject.toml` `requires-python = ">=3.10"`.
  - Add `tomli>=2.0; python_version < "3.11"` if TOML parsing via stdlib isn’t available.
  - Acceptance: Test suite passes on Python 3.10 and 3.12; `uvx` smoke tests succeed.

#### Platform support and setup notes
- [x] Add concise Windows (and Linux) audio notes in `SETUP_USER.md`
  - Windows: typical “it just works” guidance; mention granting mic permissions.
  - Linux: note PortAudio/libsndfile availability and succinct install hints.
  - Acceptance: Users can resolve basic audio prereqs quickly.

#### Venv‑less development via uv
- [ ] Document venv‑less dev
  - Examples: `uv run --active pytest -q tests/test_*.py`, `uv run --active healthyselfjournal journal cli`, `uvx healthyselfjournal -- --help`.
  - Acceptance: Devs can work without manually activating a venv.

#### Sweep docs for user‑specific paths and pin drift
- [x] Remove or neutralize any `/Users/greg/...` paths in docs
  - Replace with neutral examples or `uvx` equivalents.
- [ ] Align dependency pin narratives
  - Ensure docs reflect actual pins (e.g., `python-fasthtml` range) in `pyproject.toml`.

#### Release validation and smoke tests
- [ ] Local build and uvx smoke tests
  - `uv build`; `uvx --from dist/*.whl healthyselfjournal -- --help`.
  - Prompt asset load check as documented.
- [ ] Clean‑machine smoke test (optional)
  - `uvx healthyselfjournal -- --help` and one short run of `journal cli`.


### Unresolved questions / concerns

- Optional XDG flag for CLI
  - Expose `--xdg` now vs. defer until a user asks? (Default remains CWD‑first.)
- CI matrix
  - Do we want a 3.10/3.12 matrix to enforce the relaxed floor in CI?
- Extras
  - Consider extras for local STT or desktop (e.g., `[local]`, `[desktop]`) to simplify opt‑in installs later.


### Notes / rationale

- `importlib.metadata.version()` keeps runtime version aligned with installed distribution; a local fallback avoids breakage in editable/dev contexts.
- CWD‑first `.env.local` fits `uvx` mental model and keeps behavior transparent for non‑technical users; advanced users can opt into XDG.
- Enabling thinking by default improves question quality and aligns with the product’s goals; we clearly document budgets and overrides.
- Relaxing to Python 3.10 lowers friction; adding conditional `tomli` guards TOML parsing on <3.11.
- Promoting `uvx`/`uv run` enables venv‑less dev and reduces environment pitfalls; docs should never show user‑specific paths.


