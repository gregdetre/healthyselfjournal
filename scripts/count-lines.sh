#!/usr/bin/env bash

# Count lines in healthyselfjournal (excluding deps and sessions)

exec "$(dirname "$0")/../gjdutils/src/ts/scripts/count-lines.ts" \
    --exclude-dirs gjdutils,.venv,node_modules,sessions,experim_sessions "$@"