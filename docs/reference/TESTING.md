# Testing

Use pytest for tests. Write a small number of high-level, high-coverage tests that cover the most important edge cases. Avoid mocking where possible.

### Introduction

This document describes how we run and structure tests for Healthyself Journal: how to run them reliably, what they cover, and common pitfalls.

### See also

- `../reference/SETUP_DEV.md` - Development environment setup and venv activation
- `../reference/COMMAND_LINE_INTERFACE.md` - CLI commands referenced by tests
- `../reference/WEB_RECORDING_INTERFACE.md` - Context for web app tests and endpoints
- `../reference/ARCHITECTURE.md` - High-level architecture referenced by flow tests

### Principles

- Prefer black-box, flow-level tests that exercise the real stack (record → transcribe → prompt → write files) using temp dirs.
- Keep the suite small and high-signal; test critical paths and edge cases.
- Minimize mocking; only stub external services when needed for determinism.

### How to run

Activate the preferred venv and sync dependencies:

```
python -m venv .venv && source .venv/bin/activate
uv sync
```

Load `.env.local` (contains API keys used by integration tests), then run pytest. Due to a `tests` package present in some site-packages, always invoke tests by explicit file paths to avoid import shadowing:

```
set -a; [ -f .env.local ] && source .env.local; set +a
uv run --active pytest -q tests/test_*.py
```

Run a single test module or case:

```
uv run --active pytest -q tests/test_session.py::test_session_complete_updates_frontmatter -q -s -vv
```

### Notes on environments

- The suite passes offline; networked paths are gated by `pytest.importorskip`.
- Web app tests spin up a FastHTML app and use `TestClient`; no external network calls.
- `.env.local` is respected via the shell when running integration tests locally.

### Troubleshooting

- Import errors like `ModuleNotFoundError: No module named 'tests.test_…'` usually mean a site-packages `tests` package is shadowing repo tests.
  - Workarounds:
    - Run by explicit file paths: `pytest tests/test_*.py`
    - Or set `PYTHONPATH=$(pwd)` when invoking pytest.
    - Avoid bare `pytest` in environments that include a top-level `tests` package.


