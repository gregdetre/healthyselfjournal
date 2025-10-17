# Crisis Resources CLI Implementation Design - 2025-10-18

## Context & Goals

Following the implementation of a comprehensive global crisis resources catalogue (`docs/reference/GLOBAL_CRISIS_RESOURCES.md`), this conversation explored how to surface these verified crisis resources to users through the CLI interface. The discussion focused on UX considerations, geographic handling, voice-first interface challenges, and privacy concerns.

## Key Background

The crisis resources catalogue was created in response to research finding that **9% of mental health apps with 1M+ downloads provided erroneous crisis helpline numbers**. We now have 42+ verified services across 19 countries, all documented with authoritative sources.

**User context:**
- "I want to make this widely available (e.g. via PyPI, uvx), for free, including to people struggling with mental health."
- "Users could be based anywhere in the world"
- Voice-first journaling app with CLI interface

## Main Discussion

### User's Question

**User:** "How might the CLI command work?"

### Key Design Questions Identified

**1. Discovery vs. Integration**
Should crisis resources be:
- **A. Standalone command** (`healthyselfjournal crisis` or `healthyselfjournal help crisis`)
- **B. Integrated into session flow** (shown during/after sessions)
- **C. Both** (accessible multiple ways)

**2. Geographic Context**
How do we know which country's resources to show?
- Detect from system locale?
- Ask once at first launch?
- Always show international + let user specify country?
- Show everything and let user scroll?

### Option Analysis

#### Standalone CLI Command

```bash
# Simple discovery
healthyselfjournal crisis

# Or as help subcommand
healthyselfjournal help crisis

# With country override
healthyselfjournal crisis --country=UK
```

**Pros:**
- Always accessible, even outside a session
- Can be shared/documented ("if you need help, run...")
- Clear, dedicated purpose
- Easy to find via `--help`

**Cons:**
- User has to know it exists
- Not surfaced when most needed (mid-crisis during session)
- Requires terminal to be open

#### Session-Integrated Display

**During session:**
- Dialogue detects distress patterns
- Assistant says: "If you'd like to see crisis support resources, I can show them on screen"
- User says "yes" → resources display on terminal
- Voice doesn't read numbers (error-prone), just visual display

**After session:**
- Brief footer in session summary: "Need support? Type `healthyselfjournal crisis` to see verified helplines"

**Pros:**
- Contextual - shown when potentially needed
- Non-intrusive (offer, don't force)
- Respects voice-first (visual display, not spoken)

**Cons:**
- Requires active session
- Pattern detection could be wrong/miss cases
- User might not see terminal if voice-focused

### Concerns Raised

**Concern 1: Voice-first conflict**
Reading "one-eight-six-zero-two-six-six-two-three-four-five" is terrible UX and error-prone.

**Suggestion:** Resources should **always** display visually on terminal, never spoken. Voice interface could say "I'm showing crisis resources on your screen now" but not read numbers.

**Concern 2: False negatives in crisis detection**
Pattern-based detection will miss people in crisis who don't use obvious language.

**Suggestion:** Make resources **discoverable** via:
1. Mentioned in welcome message first time app runs
2. `--help` text includes crisis command
3. Brief non-intrusive footer after sessions
4. Dedicated command for direct access

**Concern 3: Geography**
Users travel, use VPNs, or system locale might be wrong.

**Suggestion:** Tiered approach:
```
1. First run: "Where are you located? [UK/US/CA/AU/Other]"
   - Save to config
   - Can change later with `healthyselfjournal config set location=UK`

2. Show resources:
   - International (always)
   - User's country (from config)
   - "For other countries: healthyselfjournal crisis --all"
```

### Trade-off: Accessibility vs. Overwhelming

**The tension:**
- Make resources **very** accessible → risk seeming alarmist, "this app thinks I'm in crisis"
- Make resources **subtle** → risk people not finding them when needed

**Recommendation:** Lean toward **accessible but not alarming**
- Mentioned once at welcome (normalizing, not scary)
- Available via clear command
- Offered contextually if distress detected
- But NOT pushed aggressively

### Proposed Implementation Sketch

```python
# In cli.py or new cli/crisis.py

@app.command()
def crisis(
    country: Optional[str] = typer.Option(None, "--country", "-c", help="Country code (UK, US, CA, AU, etc.)"),
    all: bool = typer.Option(False, "--all", "-a", help="Show all countries"),
):
    """Display verified crisis helpline numbers and resources.

    Free, confidential, 24/7 support available worldwide.
    """
    user_country = country or config.get("location") or detect_locale_country()

    if all:
        display_all_crisis_resources()
    else:
        display_crisis_resources_for(user_country, include_international=True)
```

**Display format example:**
```
Crisis Support Resources (Free, Confidential, 24/7)

🌍 International:
  Find A Helpline: https://findahelpline.com
  (Covers 150+ countries with local services)

🇬🇧 United Kingdom:
  • Samaritans: 116 123 (call anytime)
  • SHOUT: Text 85258
  • NHS Mental Health: Call 111

⚠️  Life-threatening emergency? Call 999

For other countries: healthyselfjournal crisis --all
To set your location: healthyselfjournal config set location=UK
```

## Open Questions

**Two questions posed to user:**

1. **When is the earliest/best time to make users aware this command exists?** During first-run setup wizard? In the help text? After first session?

2. **Privacy concern**: Should the app log when someone accesses crisis resources? (Even anonymously: "crisis command used at 2025-10-17 14:30"). Could be useful for understanding usage patterns, but could feel invasive?

## Implementation Considerations

### Voice-First Interface Challenges
- Long phone numbers are error-prone when spoken aloud
- Terminal display must be primary; voice only announces "showing resources on screen"
- Resources must be visible even during voice recording

### Discovery Mechanisms
Multiple paths to find crisis resources needed:
- Standalone CLI command (always available)
- Welcome message (normalize awareness)
- Help text (discoverability)
- Session footer (gentle reminder)
- Crisis pattern detection (contextual offer)

### Geographic Handling
- Config-driven location (saved once, changeable)
- System locale detection as fallback
- Always show international resources (Find A Helpline)
- Manual country override via `--country` flag

### Privacy and Logging
- Question whether to log crisis resource access
- If logging, use anonymous counts only
- Consider auto-redaction options
- Balance safety monitoring vs. user privacy

## Related Work

**Documents Created in This Session:**
- `docs/reference/GLOBAL_CRISIS_RESOURCES.md` - Comprehensive verified crisis resources catalogue (42+ services, 19 countries)
- `gjdutils/docs/instructions/UPDATE_CRISIS_RESOURCES.md` - Quarterly verification procedures
- `docs/reference/SAFEGUARDING.md` - Updated to reference crisis catalogue

**Related Documentation:**
- `docs/planning/251017d_global_crisis_resources_catalogue.md` - Planning and research methodology
- `docs/conversations/251017a_safeguarding_crisis_resources_research.md` - Research findings
- `docs/reference/PRODUCT_VISION_FEATURES.md` - Evidence-based product vision
- `docs/reference/DIALOGUE_FLOW.md` - Session flow and boundaries

## Next Steps

**Pending User Decisions:**
1. Choose timing for introducing crisis command to users (first run vs. help text vs. after session)
2. Decide on logging approach for crisis resource access
3. Determine priority regions for first implementation

**Future Implementation:**
1. Create `cli/crisis.py` or add to `cli.py`
2. Implement geographic detection and config storage
3. Design terminal display format
4. Add crisis pattern detection to dialogue flow (optional)
5. Update welcome message/help text
6. Add session footer reference to crisis command
7. Consider privacy logging approach
8. Test with various locales and countries

## Sources & References

**Internal:**
- Crisis resources catalogue: `docs/reference/GLOBAL_CRISIS_RESOURCES.md`
- Safeguarding principles: `docs/reference/SAFEGUARDING.md`
- Planning doc: `docs/planning/251017d_global_crisis_resources_catalogue.md`

**Research:**
- BMC Medicine study: 9% of apps provided erroneous crisis numbers
- 42+ verified services researched from official government and NGO sources
- IASP (International Association for Suicide Prevention)
- Find A Helpline (ThroughLine) - 150+ countries, 1,300+ services

**Design Principles:**
- Voice-first interface constraints
- Local-first, offline-capable
- Privacy-respecting
- Evidence-based safeguarding
- Accessible but not alarmist

---

**Date**: October 18, 2025
**Type**: Design discussion
**Status**: Awaiting user decisions on open questions
**Next Action**: User to answer timing and privacy questions
