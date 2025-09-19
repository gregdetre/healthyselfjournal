# Documentation Organisation

## Quick Start

- New to the project? Start here: `README.md` and `AGENTS.md`.
- Setting up development? See: `docs/reference/SETUP_DEV.md` (⭐ START HERE).
- Running the app? See: `docs/reference/COMMAND_LINE_INTERFACE.md` and `docs/reference/RECORDING_CONTROLS.md`.
- Understanding the product? See: `docs/reference/PRODUCT_VISION_FEATURES.md`.
- Data and files? See: `docs/reference/FILE_FORMATS_ORGANISATION.md`.
- Voice recognition? See: `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md`.
- Speech output (TTS)? See: `docs/reference/AUDIO_SPEECH_GENERATION.md`.

## By Category

### Setup & Infrastructure
Guides for getting your environment ready and understanding core tooling.

- **SETUP** (⭐ START HERE): `docs/reference/SETUP_DEV.md` — Install tooling, use the preferred external venv, and run with `uv`.
- THIRD-PARTY LIBRARIES: `docs/reference/THIRD_PARTY_LIBRARIES_NEEDED.md` — External deps and rationale.
- AUDIO & WHISPER: `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` — Speech-to-text setup, formats, and caveats.
- SPEECH (TTS): `docs/reference/AUDIO_SPEECH_GENERATION.md` — Speech synthesis and playback options.
- FILE FORMATS: `docs/reference/FILE_FORMATS_ORGANISATION.md` — Where files live and how they’re structured.

### Architecture & Design
Conceptual docs, flows, prompts, and product framing.

- **DIALOGUE FLOW** (⭐): `docs/reference/DIALOGUE_FLOW.md` — Conversation flow design and key states.
- CONVERSATION SUMMARIES: `docs/reference/CONVERSATION_SUMMARIES.md` — How session summaries are produced.
- OPENING QUESTIONS: `docs/reference/OPENING_QUESTIONS.md` — Seed prompts and intent.
- LLM PROMPTS: `docs/reference/LLM_PROMPT_TEMPLATES.md` — Prompt templates used by the system.
- PRODUCT VISION: `docs/reference/PRODUCT_VISION_FEATURES.md` — Goals, features, and scope.

### Development Workflows
Day-to-day commands and operational guides.

- **CLI** (⭐): `docs/reference/COMMAND_LINE_INTERFACE.md` — Commands to run and manage sessions.
- RECORDING CONTROLS: `docs/reference/RECORDING_CONTROLS.md` — Recording UX and control flow.

### Research Evidence
Key evidence-based references informing design. See more in `docs/research/`.

- FRICTION REDUCTION: `docs/research/OPENING_QUESTIONS_FRICTION_REDUCTION.md` — Lowering activation energy.
- STRUCTURED REFLECTION: `docs/research/STRUCTURED_REFLECTION_VS_RUMINATION.md` — Avoiding rumination loops.
- SOCRATIC TECHNIQUES: `docs/research/SOCRATIC_QUESTIONING_TECHNIQUES.md` — Question patterns.
- REDEMPTIVE NARRATIVE: `docs/research/REDEMPTIVE_NARRATIVE_CONSTRUCTION.md` — Narrative reframing.
- PROGRESS & STREAKS: `docs/research/PROGRESS_TRACKING_STREAK_DESIGN.md` — Motivation via progress.
- GRATITUDE PRACTICE: `docs/research/GRATITUDE_PRACTICE_OPTIMIZATION.md` — Evidence and parameters.
- SELF-DISTANCING: `docs/research/SELF_DISTANCING_TECHNIQUES.md` — Perspective shifting.
- SESSION TIMING: `docs/research/OPTIMAL_SESSION_TIMING.md` — Best times and durations.
- COGNITIVE–EMOTIONAL: `docs/research/COGNITIVE_EMOTIONAL_INTEGRATION.md` — Integration strategies.
- HABITS & INTENTIONS: `docs/research/IMPLEMENTATION_INTENTIONS_HABITS.md` — Habit scaffolding.
- MINDFUL REFLECTION: `docs/research/MINDFUL_REFLECTION_PLUM_VILLAGE.md` — Thich Nhat Hanh's contemplative practices for journaling.
- BEGINNING ANEW: `docs/research/BEGINNING_ANEW_PRACTICE.md` — Plum Village conflict resolution and relationship healing practice.
- TOPICS INDEX: `docs/research/POTENTIAL_RESEARCH_TOPICS.md` — Future research map.
- DEEP OVERVIEW: `docs/research/deep_research_overview/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` — Literature overview.

### Decisions & Planning
Design discussions, decisions, and forward plans.

- UI TECH DECISIONS: `docs/conversations/250917a_journaling_app_ui_technical_decisions.md` — UI/tech choices.
- DIALOGUE DESIGN: `docs/conversations/250916a_journaling_app_dialogue_design.md` — Conversation design notes.
- RESEARCH PLANNING: `docs/conversations/250917b_evidence_based_journaling_research_planning.md` — Evidence planning.
- SHORT AUDIO FLOW: `docs/conversations/250917c_short_audio_quit_flow_decision.md` — Quit flow decision.
- PRODUCT V1 PLAN: `docs/planning/250917a_voice_journaling_app_v1.md` — First release planning.

### Meta & Repo Docs
Project-level docs at the root.

- AGENTS: `AGENTS.md` — Quick guidance for agents working in this repo.
- TESTING: `TESTING.md` — Notes on testing approach and usage.
- README: `README.md` — High-level overview and entry points.

## By Persona

- **New Developer**: `docs/reference/SETUP_DEV.md`, `docs/reference/COMMAND_LINE_INTERFACE.md`, `docs/reference/DIALOGUE_FLOW.md`, `docs/reference/FILE_FORMATS_ORGANISATION.md`.
- **AI Agent**: `AGENTS.md`, `docs/reference/LLM_PROMPT_TEMPLATES.md`, `docs/reference/CONVERSATION_SUMMARIES.md`.
- **Maintainer**: `docs/reference/PRODUCT_VISION_FEATURES.md`, `docs/planning/250917a_voice_journaling_app_v1.md`, `docs/conversations/250917a_journaling_app_ui_technical_decisions.md`.


