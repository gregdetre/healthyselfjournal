# Documentation Organisation

## Quick Start

- New to the project? Start here: `README.md` and `AGENTS.md`.
- Setting up development? See: `docs/reference/SETUP_DEV.md` (⭐ START HERE).
- CLI overview and commands? See: `docs/reference/COMMAND_LINE_INTERFACE.md` and `docs/reference/CLI_COMMANDS.md`.
- Running the app? See: `docs/reference/RECORDING_CONTROLS.md`.
- Understanding the product? See: `docs/reference/PRODUCT_VISION_FEATURES.md`.
- Data and files? See: `docs/reference/FILE_FORMATS_ORGANISATION.md`.
- Voice recognition? See: `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md`.
- Speech output (TTS)? See: `docs/reference/AUDIO_SPEECH_GENERATION.md`.
- Privacy & safety? See: `docs/reference/PRIVACY.md` and `docs/reference/SAFEGUARDING.md`.

## By Category

### Setup & Infrastructure
Guides for getting your environment ready and understanding core tooling.

- **SETUP** (⭐ START HERE): `docs/reference/SETUP_DEV.md` — Install tooling, use the preferred external venv, and run with `uv`.
- USER SETUP: `docs/reference/SETUP_USER.md` — Install and quickstart for non-developers.
- THIRD-PARTY LIBRARIES: `docs/reference/THIRD_PARTY_LIBRARIES_NEEDED.md` — External deps and rationale.
- CONFIGURATION: `docs/reference/CONFIGURATION.md` — Configuration model and `user_config.toml` options.
- INIT FLOW: `docs/reference/INIT_FLOW.md` — First-run wizard and initialization behaviour.
- AUDIO & WHISPER: `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` — Speech-to-text setup, formats, and caveats.
- SPEECH (TTS): `docs/reference/AUDIO_SPEECH_GENERATION.md` — Speech synthesis and playback options.
- FILE FORMATS: `docs/reference/FILE_FORMATS_ORGANISATION.md` — Where files live and how they’re structured.
- MODELS (LOCAL LLM): `docs/reference/OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md` — Managing a local LLM model.

### Architecture & Design
Conceptual docs, flows, prompts, and product framing.

- **DIALOGUE FLOW** (⭐): `docs/reference/DIALOGUE_FLOW.md` — Conversation flow design and key states.
- CONVERSATION SUMMARIES: `docs/reference/CONVERSATION_SUMMARIES.md` — How session summaries are produced.
- OPENING QUESTIONS: `docs/reference/OPENING_QUESTIONS.md` — Seed prompts and intent.
- LLM PROMPTS: `docs/reference/LLM_PROMPT_TEMPLATES.md` — Prompt templates used by the system.
- PRODUCT VISION: `docs/reference/PRODUCT_VISION_FEATURES.md` — Goals, features, and scope.
- WEB ARCHITECTURE: `docs/reference/WEB_ARCHITECTURE.md` — Web app structure and security considerations.

### Development Workflows
Day-to-day commands and operational guides.

- **CLI** (⭐): `docs/reference/CLI_COMMANDS.md` — Command catalogue to run and manage sessions.
- CLI OVERVIEW: `docs/reference/COMMAND_LINE_INTERFACE.md` — High-level CLI index and entry points.
- RECORDING CONTROLS: `docs/reference/RECORDING_CONTROLS.md` — Recording UX and control flow.
- TERMINAL RECORDING: `docs/reference/CLI_RECORDING_INTERFACE.md` — Terminal (`journal cli`) recording interface.
- WEB RECORDING: `docs/reference/WEB_RECORDING_INTERFACE.md` — Web (`journal web`) interface and flags.
- DESKTOP APP: `docs/reference/DESKTOP_APP_PYWEBVIEW.md` — PyWebView desktop shell.
- DESKTOP (macOS): `docs/reference/DESKTOP_APP_MAC_APPLE.md` — macOS packaging/signing specifics.
- SESSIONS: `docs/reference/SESSIONS.md` — List, merge, and summary utilities.
- TESTING: `docs/reference/TESTING.md` — Running tests and test layout.
- LOGGING: `docs/reference/LOGGING.md` — Logging approach and locations.
- BACKGROUND PROCESSING: `docs/reference/BACKGROUND_PROCESSING.md` — Background tasks and workers.
- PACKAGING: `docs/reference/PYPI_PUBLISHING.md` — Publishing to PyPI.
- PERFORMANCE: `docs/reference/BACKEND_BENCHMARKS.md` — Backend performance notes.

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
- TOPICS INDEX: `docs/research/RESEARCH_TOPICS.md` — Future research map.
- DEEP OVERVIEW: `docs/research/deep_research_overview/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` — Literature overview.

### Decisions & Planning
Design discussions, decisions, and forward plans.

- UI TECH DECISIONS: `docs/conversations/250917a_journaling_app_ui_technical_decisions.md` — UI/tech choices.
- DIALOGUE DESIGN: `docs/conversations/250916a_journaling_app_dialogue_design.md` — Conversation design notes.
- RESEARCH PLANNING: `docs/conversations/250917b_evidence_based_journaling_research_planning.md` — Evidence planning.
- SHORT AUDIO FLOW: `docs/conversations/250917c_short_audio_quit_flow_decision.md` — Quit flow decision.
- PRODUCT V1 PLAN: `docs/planning/250917a_voice_journaling_app_v1.md` — First release planning.

### Privacy, Safety & Resilience
Safety resources, privacy controls, and repair flows.

- PRIVACY: `docs/reference/PRIVACY.md` — Privacy controls and operating fully locally.
- SAFEGUARDING: `docs/reference/SAFEGUARDING.md` — Approach to user safety and scope.
- CRISIS RESOURCES: `docs/reference/GLOBAL_CRISIS_RESOURCES.md` — Country/state-specific crisis hotlines and services.
- UPDATE RESOURCES: `docs/reference/UPDATE_CRISIS_RESOURCES.md` — How to update the crisis resources catalogue.
- RESILIENCE: `docs/reference/RESILIENCE.md` — Detect-and-suggest behaviour and repair flows.

### Libraries
Documentation for specific libraries used in the project.

- FASTHTML: `docs/reference/libraries/FASTHTML.md` — Notes on using FastHTML.
- QUESTIONARY: `docs/reference/libraries/QUESTIONARY.md` — Notes on interactive CLI prompts.

### Meta & Repo Docs
Project-level docs and top-level guides.

- AGENTS: `AGENTS.md` — Quick guidance for agents working in this repo.
- TESTING: `docs/reference/TESTING.md` — Notes on testing approach and usage.
- README: `README.md` — High-level overview and entry points.

## By Persona

- **New Developer**: `docs/reference/SETUP_DEV.md`, `docs/reference/COMMAND_LINE_INTERFACE.md`, `docs/reference/CLI_COMMANDS.md`, `docs/reference/DIALOGUE_FLOW.md`, `docs/reference/FILE_FORMATS_ORGANISATION.md`.
- **AI Agent**: `AGENTS.md`, `docs/reference/LLM_PROMPT_TEMPLATES.md`, `docs/reference/CONVERSATION_SUMMARIES.md`, `docs/reference/PRIVACY.md`, `docs/reference/SAFEGUARDING.md`.
- **Maintainer**: `docs/reference/PRODUCT_VISION_FEATURES.md`, `docs/reference/TESTING.md`, `docs/planning/250917a_voice_journaling_app_v1.md`, `docs/conversations/250917a_journaling_app_ui_technical_decisions.md`, `docs/reference/PYPI_PUBLISHING.md`.


