### Goal, context

Rename the entire project (package name, CLI command, imports, branding, docs, and GitHub repo) with no backwards compatibility. Perform work directly on `main` in a single cohesive change, ensuring tests pass post-rename.


### References

- docs/reference/COMMAND_LINE_INTERFACE.md — CLI usage and flags to update
- docs/reference/SETUP.md — install/run instructions to update
- docs/reference/RECORDING_CONTROLS.md — user-visible CLI text to update
- docs/planning/250917c_publish_to_pypi.md — release steps for new dist name
- gjdutils/docs/instructions/RENAME_OR_MOVE.md — safe rename/move process
- gjdutils/docs/reference/SD_STRING_DISPLACEMENT_FIND_REPLACE.md — safe find/replace tool
- pyproject.toml — project name, scripts, package discovery
- healthyselfjournal/ — current package to be renamed
- tests/ — imports and CLI references


### Principles, key decisions

- Work on `main` directly (solo project; no BC required).
- No aliases or shims. Old names are removed entirely.
- Use `git mv` for directories/files; use `sd` for string replacements with preview first.
- One atomic commit containing the complete rename; then push to origin.
- Rename GitHub repo after local tests pass and changes are pushed.
- Keep environment variables unchanged unless they encode the old brand (they do not).


### Stages & actions

#### Stage: Choose final names and confirm scope
- [ ] Decide final names:
  - NEW_DIST_NAME (PyPI/distribution, e.g. `new-name`)
  - NEW_PACKAGE_NAME (Python import/package dir, e.g. `newname`)
  - NEW_CLI_NAME (CLI command, e.g. `newname`)
  - NEW_REPO_NAME (GitHub repo slug)
- [ ] Confirm we will rewrite all references (imports, CLI text, docs, badges).
- [ ] Ensure working tree is clean (commit/stash any incidental edits).

Acceptance: Names are fixed and documented here; `git status` is clean.


#### Stage: Update project metadata (pyproject)
- [ ] Edit `pyproject.toml`:
  - [project].name = NEW_DIST_NAME
  - [project.scripts] = { NEW_CLI_NAME = "NEW_PACKAGE_NAME.__main__:app" }
  - [tool.setuptools.packages.find].include = ["NEW_PACKAGE_NAME*"]
- [ ] Remove old script entry `healthyselfjournal`.

Acceptance: `uv sync --active` completes without errors.


#### Stage: Rename package directory and update imports
- [ ] Rename the package directory:
```bash
git mv healthyselfjournal NEW_PACKAGE_NAME
```
- [ ] Update imports, strings, and paths across repo using `sd` with preview then apply:
```bash
# Preview everywhere
sd -ps "healthyselfjournal" "NEW_PACKAGE_NAME" .
# Apply
sd -s "healthyselfjournal" "NEW_PACKAGE_NAME" .
```
- [ ] Update user-facing brand strings (panels, help text) from "Healthy Self Journal" to the new brand using `sd` preview then apply.

Acceptance: `rg NEW_PACKAGE_NAME` shows imports updated; no lingering `healthyselfjournal` imports.


#### Stage: Update docs and examples
- [ ] Replace CLI command and name in `docs/reference/*.md` and `README.md`:
  - `healthyselfjournal` → NEW_CLI_NAME
  - "Healthy Self Journal" → New brand
- [ ] Update any code blocks and example commands.

Acceptance: `rg -n "healthyselfjournal|Healthy Self Journal" docs README.md` returns no results.


#### Stage: Clean environment and verify locally
- [ ] Remove build metadata and re-sync deps:
```bash
rm -rf healthyselfjournal.egg-info
uv sync --active
```
- [ ] Run tests (offline set):
```bash
uv run --active pytest -q
```
- [ ] Sanity-check CLI:
```bash
uv run --active NEW_CLI_NAME --help | cat
```

Acceptance: Tests pass; CLI help displays under NEW_CLI_NAME without old branding.


#### Stage: Commit and push on main
- [ ] Commit using the project convention:
```bash
git reset && git add -A && git commit -m "refactor(rename): rename project to NEW_REPO_NAME"
```
- [ ] Push to origin:
```bash
git push origin main
```

Acceptance: CI (if any) passes on main.


#### Stage: Rename GitHub repository and update remotes
- [ ] In GitHub Settings → General, rename repo to NEW_REPO_NAME.
- [ ] Update local `origin` remote URL:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/NEW_REPO_NAME.git
git remote -v | cat
```
- [ ] Update badges/links in `README.md` (if any), issues templates, and project description.

Acceptance: `git fetch` works; README links are correct.


#### Stage: Package publishing (if applicable)
- [ ] Follow `docs/planning/250917c_publish_to_pypi.md` to publish NEW_DIST_NAME.
- [ ] Do not publish to the old dist name again; optionally release a deprecation README there.

Acceptance: `pip install NEW_DIST_NAME` installs the new CLI name.


#### Stage: Post-rename hygiene
- [ ] Final sweep for stragglers:
```bash
rg -n "healthyselfjournal|Healthy Self Journal" .
```
- [ ] Update any external scripts or dotfiles that reference the old CLI.
- [ ] Announce change in CHANGELOG / release notes (optional).

Acceptance: No references to the old name remain in code or docs.


### Notes on tooling

- Prefer `git mv` for directory/file renames for better diffs (per RENAME_OR_MOVE.md).
- Use `sd --preview --string-mode` first to review replacements, then apply with `-s` (per SD_STRING_DISPLACEMENT_FIND_REPLACE.md).
- If you don’t have `sd` installed: `brew install sd` (macOS).


