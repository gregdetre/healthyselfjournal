# Safeguarding v1: 80-20 Implementation

## Goal

Implement pragmatic, high-impact safeguarding measures for v1 release that address critical risks without over-engineering. Focus on the essentials that beat 90% of mental health apps: clear boundaries, verified crisis resources, anti-rumination (already implemented), and basic deterioration monitoring.

**Core principle:** "Journaling app with duty of care" not "mental health intervention"

## Context

After extensive research into mental health app safeguarding (see `docs/conversations/251017a_safeguarding_crisis_resources_research.md`), we identified that most apps fail at basic safeguarding:
- 93% of suicide prevention apps lack comprehensive evidence-based strategies
- 9% of apps (including those with 1M+ downloads) have ERRONEOUS crisis helpline numbers
- 35% of digital mental health interventions collect NO safety data

Rather than attempting comprehensive safeguarding (pattern-based crisis detection, global crisis catalog, digital safety plans), v1 focuses on high-impact essentials that can be implemented quickly and maintained easily.

**User requirements:**
- Widely available via PyPI/uvx, for free
- Target users: "people struggling with mental health" (not just healthy reflection)
- Global user base (users could be anywhere)
- Hold to higher standard despite "journaling" classification
- NICE as baseline standard, but draw from international best practices
- No company structure or insurance (personal liability exposure)

## References

**Primary safeguarding docs:**
- `docs/reference/SAFEGUARDING.md` - Current safeguarding principles (needs updating with v1 implementation)
- `docs/conversations/251017a_safeguarding_crisis_resources_research.md` - Comprehensive research findings on crisis detection, safety protocols, and evidence-based strategies
- `docs/planning/251017d_global_crisis_resources_catalogue.md` - Deferred comprehensive global catalog (too complex for v1)

**Related docs:**
- `docs/reference/PRODUCT_VISION_FEATURES.md` - Product values (evidence-based, long-term wellbeing)
- `docs/reference/DIALOGUE_FLOW.md` - Question sequencing and session flow
- `healthyselfjournal/prompts/question.prompt.md.jinja` - Already contains anti-rumination guardrails (lines 48-58, Three P's awareness, concrete action redirection)
- `docs/research/AUTONOMY_SUPPORT_MI_SDT_FOR_JOURNALING.md` - MI/SDT autonomy support principles
- `docs/research/ANTI_SYCOPHANCY_AND_PARASOCIAL_RISK_GUARDRAILS.md` - Anti-sycophancy implementation

**Implementation files:**
- `healthyselfjournal/cli.py` - CLI entry points for adding resources command
- `healthyselfjournal/session.py` - Session management for deterioration checks
- `healthyselfjournal/config.py` - Configuration for break nudges and pacing
- `healthyselfjournal/storage.py` - Persistence for session counting and deterioration checks
- `README.md` - User-facing documentation for disclaimers and exclusion criteria

## Principles & Key Decisions

**80-20 approach:**
- Focus on high-impact, low-complexity interventions that address critical risks
- Defer comprehensive solutions (global catalog, pattern detection, safety plans) until post-v1
- Aim to beat 90% of apps by doing the basics WELL (correct crisis numbers, clear boundaries, rumination awareness)

**Classification:**
- "Journaling app with duty of care" (user's explicit framing)
- NOT positioned as mental health intervention
- BUT hold to higher standard because users will treat it as support regardless of disclaimers

**Liability awareness:**
- User has no company structure or insurance
- Clear documentation of exclusion criteria and scope provides legal protection
- Disclaimers must be prominent but not intrusive

**What's in v1:**
1. Clear disclaimers about scope and suitability
2. Essential crisis resources (5-10 key international resources)
3. Keep existing anti-rumination prompts (already implemented)
4. Simple session pacing boundaries (20min nudge)
5. Document exclusion criteria in README
6. Simple deterioration check (every 7-10 sessions)

**What's deferred:**
- Comprehensive global crisis resources catalog (251017d planning doc exists but deferred)
- Pattern-based crisis detection (needs testing, false positive risks)
- Digital safety plans feature (feature creep)
- Sophisticated deterioration analytics (over-engineering)
- Professional integration or hybrid monitoring models

## Stages & Actions

### Stage: Review existing anti-rumination implementation
- [ ] Read `healthyselfjournal/prompts/question.prompt.md.jinja` in detail
  - [ ] Confirm anti-rumination guardrails are already present (lines 48-58, Three P's, concrete redirection)
  - [ ] Check if any enhancements needed for v1
  - [ ] Verify autonomy-supportive language is used throughout
- [ ] Document status: confirm this is already done and working

### Stage: Add disclaimers to startup flow
- [ ] Design minimal, non-intrusive disclaimer text
  - Text: "This is a reflective journaling tool, not therapy or mental health treatment. Not suitable for active mental health crises or thoughts of self-harm. If in crisis: [key resources]. Press Enter to continue."
  - Show on: first launch only (track in config/state)
- [ ] Decide where to show disclaimer
  - Option A: Before first session starts (preferred - users see it when relevant)
  - Option B: At CLI startup (too early, may be ignored)
  - Stop and confirm approach with user
- [ ] Implement disclaimer display logic in `healthyselfjournal/cli.py` or `session.py`
- [ ] Track first-launch state in config/storage
- [ ] Add disclaimer to `--help` output (always visible, non-blocking)
- [ ] Test disclaimer appears on first run and not on subsequent runs

### Stage: Create essential crisis resources reference
- [ ] Research and verify 5-10 key international crisis resources
  - [ ] US/Canada: 988 Suicide & Crisis Lifeline (verify current)
  - [ ] US: Crisis Text Line (text HOME to 741741)
  - [ ] US backup: National Suicide Prevention Lifeline 1-800-273-8255
  - [ ] UK/Ireland: Samaritans (116 123, verify number)
  - [ ] Australia: Lifeline (13 11 14, verify number)
  - [ ] International: Befrienders Worldwide (findahelpline.com)
  - [ ] Verify each number is current as of 2025
- [ ] Create `docs/reference/CRISIS_RESOURCES.md` with verified resources
  - Include: number/URL, countries served, availability (24/7), verification date
  - Include note: "Verified quarterly. Last verified: [date]"
  - Keep format simple and scannable
- [ ] Add instruction to update verification date to `docs/reference/SAFEGUARDING.md`

### Stage: Add crisis resources to session boundaries
- [ ] Add crisis resources to session dialogue boundaries footer
  - Where: End of each question response from Claude (or periodic reminder)
  - Format: Brief, non-intrusive footer (e.g., "If in crisis: 988 (US/CA), 116 123 (UK), 13 11 14 (AU), or findahelpline.com")
  - Frequency: Decide on appropriate reminder frequency (every response vs every 5-10 mins)
  - Stop and confirm approach with user before implementing
- [ ] Implement in prompt template or session logic
- [ ] Test that footer appears appropriately during sessions

### Stage: Add CLI command for direct crisis resource access
- [ ] Add `resources` or `crisis` command to CLI
  - Shows full crisis resources from `CRISIS_RESOURCES.md`
  - Available mid-session (interrupt flow if needed)
  - Implement in `healthyselfjournal/cli.py`
- [ ] Document command in `--help` and `docs/reference/CLI_COMMANDS.md`
- [ ] Test command works mid-session

### Stage: Document exclusion criteria
- [ ] Draft exclusion criteria list
  - Active suicidal ideation with plan
  - Recent suicide attempt
  - Active psychosis or mania
  - Severe PTSD with flashbacks
  - Substance abuse crisis
  - Under 18 without parental awareness (discuss with user)
- [ ] Add to README.md in clear, prominent section
  - Section: "Who This App Is NOT Suitable For"
  - Include disclaimer: "This app is not suitable if you are experiencing..."
  - Direct to crisis resources if any criteria apply
- [ ] Add to `docs/reference/SAFEGUARDING.md`
- [ ] Stop and review with user before finalizing

### Stage: Implement simple deterioration check
- [ ] Design deterioration check UX
  - Trigger: After every 7-10 sessions (configurable)
  - Question: "How has journaling been for you? [Helpful / Neutral / Making things worse / Skip]"
  - If "Making things worse": Show message suggesting pause, provide alternatives, log event
- [ ] Implement session counter in `storage.py`
  - Track total sessions
  - Track sessions since last deterioration check
- [ ] Implement check prompt in `session.py` or `cli.py`
  - Show at session start or end (decide based on UX)
  - Store response in session metadata
- [ ] Implement event logging for "making worse" responses
  - Add to `events.log` (metadata only, no content)
  - Format: `timestamp,event=deterioration_check,response=worse,session_count=N`
- [ ] Make check frequency configurable in `config.py`
- [ ] Test deterioration check appears after appropriate session count
- [ ] Test logging works for "making worse" responses

### Stage: Implement session time pacing
- [ ] Review existing break nudge implementation (mentioned in SAFEGUARDING.md)
  - Check if already implemented at ~20 minutes
  - If not, implement gentle break reminder at 20 minutes
- [ ] Add optional weekly usage summary
  - "You've journaled X times this week" (neutral tone, no judgment)
  - Configurable on/off
  - Implement in `storage.py` and `session.py`
- [ ] Make pacing features configurable in `config.py`
- [ ] Test break nudge appears at appropriate time
- [ ] Test weekly summary (if implemented)

### Stage: Update documentation
- [ ] Update `docs/reference/SAFEGUARDING.md`
  - Add implementation details for v1 features
  - Update "Implementation notes" section with actual implementation locations
  - Add verification schedule for crisis resources
  - Add note about deferred features (global catalog, pattern detection, safety plans)
- [ ] Update README.md
  - Add disclaimers section
  - Add exclusion criteria
  - Add crisis resources quick reference
  - Link to full `CRISIS_RESOURCES.md`
- [ ] Update `docs/reference/CLI_COMMANDS.md` with crisis resources command
- [ ] Review `docs/reference/SETUP_USER.md` to ensure disclaimers/safety info is visible early

### Stage: Testing and validation
- [ ] Write automated tests for key safety features
  - [ ] Test disclaimer shows on first launch only
  - [ ] Test deterioration check triggers at correct session count
  - [ ] Test crisis resources command works
  - [ ] Test event logging for deterioration responses
- [ ] Manual testing of complete user flow
  - [ ] First launch experience (disclaimer)
  - [ ] Session boundaries (crisis resources footer)
  - [ ] Crisis resources command mid-session
  - [ ] Deterioration check after N sessions
- [ ] Review all user-facing text for tone (neutral, autonomy-supportive, no praise/emojis)

### Stage: Final review and commit
- [ ] Stop and review all changes with user
  - Confirm disclaimer text and placement
  - Confirm crisis resources are correct and sufficient for v1
  - Confirm deterioration check UX is appropriate
  - Discuss any remaining concerns
- [ ] Run type checking: `uv run --active mypy healthyselfjournal` (if project uses mypy)
- [ ] Run tests: `uv run --active pytest -q tests/test_*.py`
- [ ] Git commit changes following `gjdutils/docs/instructions/GIT_COMMIT_CHANGES.md`
  - Use descriptive commit message covering all v1 safeguarding features
  - Include reference to this planning doc

### Stage: Post-v1 planning
- [ ] Document lessons learned from v1 implementation
- [ ] Gather user feedback on safeguarding features
  - Are disclaimers appropriate?
  - Are crisis resources accessible enough?
  - Is deterioration check helpful or intrusive?
- [ ] Decide on priority for deferred features
  - Pattern-based crisis detection (needs research on voice-specific patterns)
  - Expanded global crisis catalog (see 251017d planning doc)
  - Digital safety plans (50% ED reduction evidence)
  - Professional integration options

## Appendix

### Research Summary: What Doesn't Work vs What Works

**What Doesn't Work (avoid these):**
- Simple keyword detection for crisis → Users use indirect language, slang, humor
- "Go call 911" and halt approach → Alienates users, breaks trust
- Erroneous crisis numbers → 9% of apps had WRONG numbers
- No safety monitoring → 35% of DMHIs collect zero safety data

**What Works (v1 focuses on these):**
- Verified crisis resources → Essential baseline, must be correct
- Clear scope boundaries → Legal protection, sets expectations
- Rumination mitigation → Prevents symptom deterioration (most common adverse event)
- Simple deterioration monitoring → Catches worsening early
- Autonomy-supportive language → Respects user agency, reduces dependency risk

### Alternative Approaches Considered and Deferred

**Comprehensive global crisis resources catalog:**
- Desiderata: Cover all countries, verify quarterly, provide local language resources
- Tradeoff: High maintenance burden, diminishing returns for v1 user base
- Decision: Defer to post-v1, start with 5-10 key international resources
- Rationale: Better to have 5 correct numbers than 100 potentially outdated ones
- See: `docs/planning/251017d_global_crisis_resources_catalogue.md`

**Pattern-based crisis detection:**
- Desiderata: Detect crisis indicators across sessions (hopelessness + sleep disruption + social withdrawal)
- Tradeoff: False positives (intrusive interruptions), false negatives (missed crises), needs extensive testing
- Decision: Defer to post-v1 after gathering real usage data
- Rationale: Risk of harm from false positives/negatives in untested system

**Digital safety plans:**
- Desiderata: Evidence-based (50% reduction in ED visits), user-controlled
- Tradeoff: Feature creep for v1, assumes user has supports to list
- Decision: Defer to post-v1 as separate feature
- Rationale: Excellent feature but not critical for v1 launch; needs thoughtful UX design

**Sophisticated deterioration analytics:**
- Desiderata: Track mood trends, detect declining patterns, validated scales (PHQ-2)
- Tradeoff: Over-engineering for v1, requires baseline data to be meaningful
- Decision: Defer to post-v1, start with simple "Helpful/Neutral/Worse" check
- Rationale: Need usage data to build meaningful analytics; simple check covers basics

### Key Statistics Driving Decisions

- **93% of suicide prevention apps** lack comprehensive evidence-based strategies
- **9% of apps (including 1M+ downloads)** provided erroneous crisis helpline numbers
- **35% of digital mental health interventions** collected NO safety data
- **50% reduction in ED visits** when digital safety plans activated (evidence for future feature)
- **Main adverse event** in DMHIs: symptom deterioration from increased rumination

### User's Explicit Decisions (from 251017a conversation)

- Classification: "It's journalling" but holding to "higher standard nonetheless"
- Rationale: "users will ignore the distinction, and to lead by example"
- Distribution: "widely available (e.g. via PyPI, uvx), for free"
- Target users: "including to people struggling with mental health"
- Standards: "I would trust/default to NICE, but we should draw from anything good from anywhere"
- Geographic scope: "Users could be based anywhere in the world"
- Legal structure: "I don't have insurance. At the moment, I'm not releasing this via a company"
- Technical: "we transcribe everything. We're not solely local-first - we also offer cloud models"

### Risks and Mitigations

**Risk: Personal liability exposure (no company structure/insurance)**
- Mitigation: Clear disclaimers, documented exclusion criteria, adherence to best practices
- Ongoing: User should consult solicitor before public release (raised in 251017a conversation)

**Risk: Erroneous crisis resources (like 9% of apps)**
- Mitigation: Verify all numbers before v1, document verification date, add quarterly review reminder
- Ongoing: Need process for user-reported errors

**Risk: Deterioration check feels intrusive or judgmental**
- Mitigation: Neutral tone, "Skip" option, autonomy-supportive framing
- Test: Gather user feedback in v1, adjust as needed

**Risk: Disclaimer ignored/skipped by users in crisis**
- Mitigation: Show at first launch (most salient moment), repeat in help text, add to crisis resources footer
- Limitation: Accept that some users will skip regardless; documented effort is legally protective

**Risk: Voice-specific crisis expression patterns not well understood**
- Mitigation: Don't attempt sophisticated detection in v1; focus on session-level deterioration check
- Future: Research voice-specific patterns before attempting real-time detection
