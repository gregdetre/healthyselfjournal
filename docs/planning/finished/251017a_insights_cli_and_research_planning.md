# Insights CLI (v1) – Planning

## Goal, context

Design and deliver a minimal, safe, evidence‑informed `insights` CLI with two subcommands (`generate`, `list`) that:
- Drip‑feeds insights one at a time (non‑overwhelming), inviting reflection and savoring
- Defaults to a two‑range context: (a) summaries from start → last insights output; (b) full transcripts since last insights output
- Reuses prior insights to avoid repetition; calls out what’s evolved
- Uses the existing Jinja prompt machinery and model providers
- Writes outputs under `[SESSIONS-DIR]/insights/yyMMdd_HHmm_insights.md`


## References

### Research Foundation (Completed)
- `docs/research/INSIGHTS_RESEARCH_OVERVIEW.md` – Umbrella doc signposting research foundation
- `docs/research/SELF_GENERATED_INSIGHT_MEMORY_ADVANTAGE.md` – Neuroscience of self-discovery and memory
- `docs/research/AI_THERAPY_CHATBOT_RISKS.md` – Safety risks and boundaries; Anthropic/DeepMind guidance
- `docs/research/SESSION_CLOSURE_CONSOLIDATION.md` – Consolidation vs. termination; progress review
- `docs/reference/INSIGHTS.md` – Evergreen implementation guide with principles, guardrails, examples
- `docs/conversations/250919a_session_insights_wrapup_research_assessment.md` – Initial research assessment

### Implementation References
- `docs/reference/FILE_FORMATS_ORGANISATION.md` – Sessions directory layout; where outputs should live; frontmatter norms
- `docs/reference/LLM_PROMPT_TEMPLATES.md` – Jinja template system used for prompts
- `docs/reference/CLI_COMMANDS.md` – Overall CLI structure and discovery docs; how new commands are documented
- `docs/reference/CONVERSATION_SUMMARIES.md` – How summaries are stored/used across sessions
- `healthyselfjournal/cli.py` – Top-level Typer app where we register sub‑apps
- `healthyselfjournal/cli_session.py` – Patterns for CLI listing and status commands
- `healthyselfjournal/storage.py` – `load_transcript`/`write_transcript` for frontmatter and body
- `healthyselfjournal/history.py` – `load_recent_summaries` token‑budgeted summary history
- `healthyselfjournal/llm.py` – Jinja rendering, prompt loaders, and provider dispatch
- `healthyselfjournal/prompts/` – Existing `question` and `summary` prompt templates to mirror for `insights`
- User's ad‑hoc inspiration output: `~/Dropbox/misc/me me me/writing, thinking, ideas/new writing/journal/insights/250921_0027_insights.md`
- Guidance: `gjdutils/docs/instructions/WRITE_PLANNING_DOC.md`, `WRITE_EVERGREEN_DOC.md`, `WRITE_DEEP_DIVE_AS_DOC.md`, `CAPTURE_SOUNDING_BOARD_CONVERSATION.md`

## Principles, key decisions
- “I don’t want to overwhelm the user with a barrage of insights. I’d rather dripfeed them, perhaps one at a time, and invite them to savour/reflect/question each one.”
- “Ideally the `generate` flow would be interactive (though maybe for v1 it’s just generate-only, but one-at-a-time?).”
- Default ranges: “context = all the summaries from the start up to the most recent existing insight output; include full transcripts since the most recent existing insight output.”
- “If there is no recent existing insight output, then apply a sensible cap. In practice, as long as we’re under 100k words, most LLMs should handle it fine.”
- v1 scope: only `generate` and `list`, keep it simple with sensible defaults; fancy options deferred.
- Non‑directive reflection: Insights should mirror patterns and connections without diagnosis/advice or therapy stance. This is a journaling/assisted‑reflection app.
- Dripfeed by default: Generate a single insight per run in v1; plan a “press a key for another one” loop in a later version; future versions may allow user text input to steer insights or have a conversation about them.
  - “I don’t want to overwhelm the user with a barrage of insights. I’d rather dripfeed them, perhaps one at a time, and invite them to savour/reflect/question each one.”
  - “dripfeed loop, press a key for another one (and future versions allow text input from the user to steer the insights generation or have a conversation about them)”
- Two‑range default:
  - Context: all historical summaries from the beginning up to the last insights output
  - Detail: include full transcripts since the last insights output
  - When no prior insights exist: apply a sensible cap (words or sessions) to stay well under ~100k words
- Short quotes: Include brief quotes from recent transcripts when helpful (v1: yes).
  - “yes to short quotes - I think they're helpful”
- Caps & budgeting: Use judgment to keep inputs under ~100k words; prefer trimming recent transcripts if needed while preserving historical summaries.
  - “exact cap use your judgment”
- Reuse prior insights: Read the latest insights file(s) to reduce repetition and highlight change since last time.
- Jinja prompts: Add `insights.prompt.md.jinja`; keep variables minimal and well‑defined.
- Provenance: Frontmatter should record model, range, and input filenames for traceability; skip hashing/dedup in v1.
  - “i'm not convinced we need to do any hashing/de-duping for v1”
- Simplicity first: Start with summaries‑only for historical context; transcripts only for the “since last insight” window. Avoid complex flags in v1.

## Stages & actions

### Stage: Research and define evidence‑based insights boundaries (foundations) ✅ COMPLETE

- [x] Conduct deep‑dive research and capture as reference docs per `WRITE_DEEP_DIVE_AS_DOC.md`
  - ✅ Created three focused research documents following existing pattern:
    - `SELF_GENERATED_INSIGHT_MEMORY_ADVANTAGE.md` – Duke 2025 neuroscience; insight memory advantage; AI cognitive impact
    - `AI_THERAPY_CHATBOT_RISKS.md` – Stanford warnings; regulatory response; Anthropic/DeepMind guidelines; risk taxonomy
    - `SESSION_CLOSURE_CONSOLIDATION.md` – 2025 psychotherapy research; consolidation terminology; patient involvement
  - ✅ Created umbrella overview: `INSIGHTS_RESEARCH_OVERVIEW.md` – Signposting and synthesis of research foundation
  - ✅ All docs include sources, dates, URLs, and applicability to journaling contexts
  - ✅ Added Anthropic Responsible Scaling Policy and DeepMind guidance
  - ✅ Updated `RESEARCH_TOPICS.md` to list new research

- [x] Draft `docs/reference/INSIGHTS.md` (evergreen) per `WRITE_EVERGREEN_DOC.md`
  - ✅ 7 core principles with research foundations
  - ✅ Detailed boundary definitions (descriptive vs. therapeutic)
  - ✅ Tone and style guidelines (voice characteristics, language patterns)
  - ✅ 5 comprehensive do/don't examples with explanations
  - ✅ Data inputs definition (two-range default, reuse logic, provenance)
  - ✅ Dripfeed cadence guidance (v1, v2, v3 future)
  - ✅ Risk mitigations (hallucination, validation, overreach, dependency)
  - ✅ Implementation notes (prompt template, file output, CLI commands, testing)
  - ✅ Cross-referenced to all research docs

- [x] Review with user for alignment before code changes
  - ✅ Research complete and documented
  - Ready for user review

**Acceptance criteria: ✅ MET**
- ✅ Reference docs (3 focused research files + 1 overview) committed
- ✅ INSIGHTS.md evergreen doc complete with all sections
- ✅ Internally consistent with product values (evidence-based, user-agency, non-directive)
- ✅ Explicit boundaries between safe/risky approaches with examples
- ✅ Industry alignment (Anthropic, DeepMind) documented
- ✅ Cross-references in place across research and reference docs

### Stage: CLI: `insights` sub‑app with `list` (v1)
- [ ] Create `healthyselfjournal/cli_insights.py` with `build_app()` returning a Typer app
- [ ] Implement `list`:
  - Lists existing files under `[SESSIONS-DIR]/insights/*.md` (newest first); shows short titles/first line
  - If none, prints a helpful message
- [ ] Register in `healthyselfjournal/cli.py` as `app.add_typer(build_insights_app(), name="insights")`
- [ ] Update `docs/reference/CLI_COMMANDS.md` examples

Acceptance criteria:
- `uv run --active healthyselfjournal insights list` works and is documented

### Stage: Jinja prompt and generator: `generate` (v1 minimal)
- [ ] Add `healthyselfjournal/prompts/insights.prompt.md.jinja` with minimal, safe guardrails:
  - Inputs: `historical_summaries` (oldest→newest), `recent_transcripts` (newest window), `prior_insights_excerpt`, `range_text`, `guidelines`
  - Behavior: produce a single concise insight; avoid repeating prior insights; invite user reflection with 1 question
- [ ] Implement an orchestrator (e.g., `generate_insights`) mirroring `generate_summary` plumbing in `llm.py`
  - Use same provider dispatch; respect `cloud_off`
  - Token budget defaults conservative; allow `--llm-model` override
- [ ] File output writer: `[SESSIONS-DIR]/insights/yyMMdd_HHmm_insights.md`
  - Frontmatter: `generated_at`, `model_llm`, `source_range` (since/until, num_sessions, words_estimate), `source_sessions`, `prior_insights_refs`, `guidelines_version`

Acceptance criteria:
- Single‑run generates exactly one insight file by default with proper provenance
- Includes at least one short quote from recent transcripts when available

### Stage: Range resolution (two‑range default) and reuse logic
- [ ] Detect last insights output in `[SESSIONS-DIR]/insights/`; if none, treat “last insights” as beginning of time
- [ ] Build historical context set: all summaries up to last insights
- [ ] Build recent detail set: full transcripts from last insights to now; if none exist, fallback to recent few sessions
- [ ] Apply caps: ensure total input stays well within operational token/word budget (~<100k words), favoring including all historical summaries and trimming recent transcripts if needed
- [ ] Load prior insights excerpt (last N chars) to help avoid repetition

Acceptance criteria:
- Deterministic selection of sessions; explicit printed range summary prior to generation

### Stage: Tests (offline, minimal)
- [ ] Add tests under `tests/` covering:
  - Range selection behavior with/without prior insights
  - `list` output when empty and when populated
  - Output frontmatter structure and minimal invariants
  - Prompt rendering smoke test via a local provider stub

Acceptance criteria:
- `uv run --active pytest -q tests/test_*.py` passes locally without network

### Stage: Documentation updates
- [ ] Update `docs/reference/CLI_COMMANDS.md` to include `insights` group with `generate` and `list`
- [ ] Add `docs/reference/INSIGHTS.md` cross‑links into relevant docs (e.g., `LLM_PROMPT_TEMPLATES.md`, `FILE_FORMATS_ORGANISATION.md`)

Acceptance criteria:
- Docs discoverability and cross‑refs in place; examples runnable

## Open questions
- When to add the interactive dripfeed loop and user‑steered input? (Target v2; v1 = one‑per‑run.)

## Risks and mitigations
- Over‑interpretation risk → Prompt guardrails; emphasize descriptive patterns; include a reflective question instead of advice
- Repetition → Include prior insights excerpt; rely on prompt guidance; skip hashing/dedup in v1
- Token/word overflows → Clear budgeting and trimming rules; bias toward summaries for history and transcripts only for the recent window
- User overwhelm → One insight per generation by default; explicit instruction to keep it concise

## Next steps
- Proceed with the research stage first (deep‑dive + INSIGHTS.md), then implement `list` and minimal `generate` with the default two‑range selection and provenance.

---

Appendix: Notes on implementation fit
- Historical summaries are available via `load_transcript` and `history.load_recent_summaries`; for v1, we will likely implement a simple, explicit collector to meet the bespoke “two‑range” requirement.
- Prompt loader/renderer mirrors `llm.generate_summary`; add a dedicated `insights` prompt to keep concerns distinct and safeguarded.
