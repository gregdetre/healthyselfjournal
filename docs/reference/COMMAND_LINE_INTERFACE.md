# Command Line Interface (index)

This page is a high-level index. For detailed usage, see:

- `CLI_COMMANDS.md` – Command index and examples
- `CLI_RECORDING_INTERFACE.md` – Terminal recording (`journal cli`)
- `WEB_RECORDING_INTERFACE.md` – Web recording (`journal web`)
- `DESKTOP_APP_PYWEBVIEW.md` – PyWebView desktop shell (`healthyselfjournal journal desktop`)
- `SESSIONS.md` – Session utilities (`session list`, `session merge`, `session summaries`)
- `CLI_COMMANDS.md` – Command index and examples
- `PRIVACY.md` – Privacy controls and how to operate fully local

 Diagnostics:

- `diagnose mic` – Interactive microphone and STT check
- `diagnose local` – Diagnostics group (shows help if no subcommand)
- `diagnose local stt|llm|privacy` – Run individual local checks
- `diagnose cloud stt|llm|tts` – Cloud key presence checks; optional probes

Repair utilities:

- `fix stt` – Backfill missing STT and replace placeholders
- `fix backfill` – Generate summaries where missing
- `fix regenerate <file>` – Regenerate a summary for one session

See also:
- `CLI_COMMANDS.md` – Command catalogue
- `CLI_COMMANDS.md` – Command catalogue
- `RESILIENCE.md` – Detect-and-suggest behaviour, placeholders, and fix guidance
