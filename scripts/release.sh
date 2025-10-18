#!/usr/bin/env bash
set -euo pipefail

# Simple release helper for healthyselfjournal
#
# Usage:
#   scripts/release.sh -v <version> [-r pypi|testpypi] [--validate] [--no-git] [--skip-clean-check]
#
# Examples:
#   scripts/release.sh -v 0.2.9 --validate
#   scripts/release.sh -v 0.2.9 -r testpypi --skip-clean-check

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VERSION=""
REPO_NAME="pypi"   # or testpypi
VALIDATE=false
SKIP_CLEAN_CHECK=false
DO_GIT=true

print_usage() {
  echo "Usage: $0 -v <version> [-r pypi|testpypi] [--validate] [--no-git] [--skip-clean-check]" >&2
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found on PATH" >&2
    exit 1
  fi
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)
      VERSION="$2"; shift 2 ;;
    -r|--repo)
      REPO_NAME="$2"; shift 2 ;;
    --validate)
      VALIDATE=true; shift ;;
    --no-git)
      DO_GIT=false; shift ;;
    --skip-clean-check)
      SKIP_CLEAN_CHECK=true; shift ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "Error: version is required (-v)" >&2
  print_usage
  exit 1
fi

# Requirements
require_cmd python
require_cmd uv
require_cmd git

# Optional: unzip is not required (we use Python to inspect wheels)

echo "==> Releasing healthyselfjournal v$VERSION (repo: $REPO_NAME)"

if [[ "$SKIP_CLEAN_CHECK" != true ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree not clean. Commit or stash changes, or pass --skip-clean-check." >&2
    git status --short
    exit 1
  fi
fi

# Check version matches pyproject
PY_VERSION=$(grep -E '^version\s*=\s*"[^"]+"' pyproject.toml | sed -E 's/.*"([^"]+)"/\1/')
if [[ "$PY_VERSION" != "$VERSION" ]]; then
  echo "Error: pyproject.toml version ($PY_VERSION) != provided version ($VERSION)" >&2
  exit 1
fi

# Check changelog has the version header
if ! grep -q "^## $VERSION -" CHANGELOG.md; then
  echo "Error: CHANGELOG.md missing '## $VERSION - ...' section" >&2
  exit 1
fi

echo "==> Building distributions"
uv build

WHEEL="dist/healthyselfjournal-$VERSION-py3-none-any.whl"
SDIST="dist/healthyselfjournal-$VERSION.tar.gz"
if [[ ! -f "$WHEEL" || ! -f "$SDIST" ]]; then
  echo "Error: expected artifacts not found for $VERSION" >&2
  ls -la dist || true
  exit 1
fi

echo "==> Verifying wheel contents and runtime prompt loading"
python - <<PY
import sys, zipfile
w = "$WHEEL"
req = {
  'healthyselfjournal/prompts/question.prompt.md.jinja',
  'healthyselfjournal/prompts/summary.prompt.md.jinja',
  'healthyselfjournal/prompts/insights.prompt.md.jinja',
  'healthyselfjournal/static/css/app.css',
  'healthyselfjournal/static/js/app.js',
  'healthyselfjournal/web/templates/session_shell.html.jinja',
  'healthyselfjournal/web/templates/settings.html.jinja',
  'healthyselfjournal/web/templates/setup.html.jinja',
  'healthyselfjournal/__main__.py',
  'healthyselfjournal/cli.py',
}
with zipfile.ZipFile(w) as z:
  names = set(z.namelist())
  missing = sorted([p for p in req if p not in names])
  if missing:
    raise SystemExit(f"Missing required files in wheel: {missing}")

# Runtime prompt loading smoke test (import directly from wheel)
sys.path.insert(0, "$WHEEL")
import healthyselfjournal.llm as m
import healthyselfjournal.cli as _cli
snippet = m._load_prompt('question.prompt.md.jinja')[:40]
print('PROMPT_OK:', snippet and len(snippet) > 0)
print('CLI_IMPORT_OK:', bool(_cli))
PY

echo "==> Uploading to $REPO_NAME"
if [[ "$REPO_NAME" == "testpypi" ]]; then
  uvx twine upload -r testpypi "$WHEEL" "$SDIST"
else
  uvx twine upload "$WHEEL" "$SDIST"
fi

if [[ "$VALIDATE" == true ]]; then
  echo "==> Validating from PyPI via uvx"
  uvx healthyselfjournal=="$VERSION" -- --help || true
fi

if [[ "$DO_GIT" == true ]]; then
  echo "==> Tagging and pushing"
  # Push main first (assumes commit already made)
  git push origin main
  if git tag -l | grep -qx "v$VERSION"; then
    echo "Tag v$VERSION already exists; skipping tag creation"
  else
    git tag "v$VERSION"
  fi
  git push origin "v$VERSION"
fi

echo "==> Done. View at: https://pypi.org/project/healthyselfjournal/$VERSION/"

