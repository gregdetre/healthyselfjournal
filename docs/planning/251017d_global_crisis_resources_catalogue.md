# Global Crisis Resources Catalogue - Research & Documentation Plan

## Goal

Create a comprehensive, well-cited catalogue of verified crisis helpline numbers, websites, and contact details for mental health and suicide prevention resources across as many countries as possible, to support users of the journaling app regardless of their location.

**Context:**
Following research into safeguarding best practices for mental health/journaling apps, a critical gap was identified: 9% of apps with 1M+ downloads provided erroneous crisis helpline numbers. Given that users could be based anywhere in the world and the app explicitly targets people struggling with mental health, we need accurate, verified, regularly-updated crisis resources.

The app transcribes voice to text, enabling crisis pattern detection, making it essential to have appropriate verified resources to provide when indicators are detected or when users request help.

## References

### Internal Documentation
- `docs/conversations/251017a_safeguarding_crisis_resources_research.md` - Research findings on safeguarding best practices; identified critical gap in crisis resources
- `docs/reference/SAFEGUARDING.md` - Current safeguarding approach; mentions "help resources" but lacks comprehensive verified list
- `docs/reference/PRODUCT_VISION_FEATURES.md` - Product vision prioritizes evidence-based design and long-term wellbeing
- `docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md` - Research standards for documentation quality
- `gjdutils/docs/instructions/WRITE_EVERGREEN_DOC.md` - Documentation structure and quality standards
- `gjdutils/docs/instructions/WRITE_DEEP_DIVE_AS_DOC.md` - Guidelines for comprehensive research documentation

### Key Research Findings
- BMC Medicine study: 6 of 69 apps (9%) provided erroneous crisis helpline numbers
- Evidence-based suicide prevention strategy #6: "Access to Emergency Counseling"
- NICE DHT framework requires demonstrating safety and benefit outweighs risk
- WHO guidelines on digital health interventions

### Existing Crisis Resource Lists (for reference)
- 988 Suicide & Crisis Lifeline (US) - current as of 2025
- National Suicide Prevention Lifeline (US): 1-800-273-8255
- Crisis Text Line (US): Text HOME to 741741
- Samaritans (UK): 116 123
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

## Principles & Key Decisions

### Standards & Requirements

**Verification Priority:**
- Accuracy is paramount - wrong numbers could cost lives
- Must include verification date and source for each entry
- Regular re-verification needed (numbers change)
- Prefer official government/NGO sources over aggregated lists

**Scope:**
- Focus on crisis/suicide prevention helplines first (highest priority)
- Include mental health support lines (secondary priority)
- Include text/chat/online options where available (accessibility)
- Document 24/7 availability vs limited hours
- Note language availability where relevant
- Include both national and local resources where appropriate

**Geographic Coverage:**
- Aim for comprehensive global coverage
- Prioritize countries with significant English-speaking populations initially
- Prioritize UK (developer location) and major population centers
- Include countries from all continents
- Note regional variations within countries where applicable

**Accessibility:**
- Document multiple access methods (phone, text, web chat, email)
- Note availability for specific populations (youth, LGBTQ+, veterans, etc.)
- Include crisis resources for specific situations (domestic violence, substance abuse)
- Consider accessibility for deaf/hard of hearing users

**Documentation Quality (per WRITE_EVERGREEN_DOC.md):**
- Clear structure optimized for lookups
- Comprehensive "See also" references
- Well-sourced with verification dates
- Maintained and updateable
- Practical implementation guidance for app integration

**Legal & Ethical:**
- Make resources accessible mid-session, not just in documentation
- Never diagnose or assess risk level
- Provide resources as options, not commands
- Respect user autonomy while ensuring safety information available

### User Requirements

From conversation:
- "I'm in the UK, and as a rule I would trust/default to NICE, but we should draw from anything good from anywhere"
- "Users could be based anywhere in the world"
- "I want to make this widely available (e.g. via PyPI, uvx), for free"
- "Including to people struggling with mental health"

## Stages & Actions

### Stage: Initial Research Strategy & Verification Standards

- [ ] Review current state of crisis resources in app
  - [ ] Check `docs/reference/SAFEGUARDING.md` for existing mentions of resources
  - [ ] Search codebase for any hardcoded crisis numbers/URLs
  - [ ] Check `healthyselfjournal/prompts/question.prompt.md.jinja` for boundaries footer content

- [ ] Define verification methodology
  - [ ] Document what constitutes a "verified" source (official government sites, established NGOs)
  - [ ] Create verification checklist template
  - [ ] Define frequency for re-verification (suggest quarterly at minimum)
  - [ ] Establish process for user-reported errors

- [ ] Create structured data format
  - [ ] Design YAML/JSON schema for crisis resources (country, service name, phone, text, web, hours, languages, populations served, verified date, source URL)
  - [ ] Consider whether to use structured data file or markdown table
  - [ ] Plan for version control and change tracking

### Stage: Comprehensive Web Research - Priority Regions

Research note: Use parallel web searches for efficiency; focus on authoritative sources

- [ ] Search for authoritative international crisis resource directories
  - [ ] International Association for Suicide Prevention (IASP)
  - [ ] WHO mental health resources
  - [ ] Befrienders Worldwide
  - [ ] Suicide Prevention Resource Center

- [ ] Research UK resources (developer location, NICE standards)
  - [ ] Samaritans (116 123)
  - [ ] SHOUT Crisis Text Line (text 85258)
  - [ ] NHS mental health services and crisis numbers
  - [ ] Regional/devolved nation variations (Scotland, Wales, Northern Ireland)

- [ ] Research US resources (large English-speaking population)
  - [ ] 988 Suicide & Crisis Lifeline (verify current as of 2025)
  - [ ] National Suicide Prevention Lifeline (1-800-273-8255)
  - [ ] Crisis Text Line (HOME to 741741)
  - [ ] Veterans Crisis Line
  - [ ] Trevor Project (LGBTQ+ youth)
  - [ ] State-specific resources

- [ ] Research Canada resources
  - [ ] Canada Suicide Prevention Service
  - [ ] Crisis Services Canada
  - [ ] Kids Help Phone
  - [ ] Provincial variations

- [ ] Research Australia & New Zealand
  - [ ] Lifeline Australia (13 11 14)
  - [ ] Beyond Blue
  - [ ] 1737 (NZ)
  - [ ] Lifeline Aotearoa

- [ ] Research Ireland
  - [ ] Samaritans Ireland
  - [ ] Pieta House
  - [ ] Text About It (50808)

### Stage: Comprehensive Web Research - Additional Regions

- [ ] Research European Union countries
  - [ ] Research EU-wide resources if any
  - [ ] Germany (Telefonseelsorge)
  - [ ] France (SOS Amitié)
  - [ ] Netherlands (113 Zelfmoordpreventie)
  - [ ] Spain (Teléfono de la Esperanza)
  - [ ] Italy, Portugal, Belgium, Nordic countries
  - [ ] Eastern European countries

- [ ] Research Asia-Pacific
  - [ ] India (AASRA, Vandrevala Foundation)
  - [ ] Japan (TELL Lifeline)
  - [ ] South Korea
  - [ ] Singapore (Samaritans of Singapore)
  - [ ] Philippines, Indonesia, Thailand
  - [ ] Hong Kong, Taiwan

- [ ] Research Middle East & Africa
  - [ ] South Africa (SADAG)
  - [ ] Israel (ERAN)
  - [ ] UAE, Saudi Arabia
  - [ ] Kenya, Nigeria, other major population centers

- [ ] Research Latin America
  - [ ] Brazil (CVV)
  - [ ] Mexico
  - [ ] Argentina (Centro de Asistencia al Suicida)
  - [ ] Chile, Colombia, other countries

### Stage: Specialized & Underserved Populations

- [ ] Research resources for specific populations
  - [ ] LGBTQ+ specific crisis lines (e.g., Trevor Project, LGBT National Hotline)
  - [ ] Youth and adolescent specific (e.g., Kids Help Phone)
  - [ ] Veterans and military (e.g., Veterans Crisis Line)
  - [ ] Domestic violence (overlaps with mental health crises)
  - [ ] Substance abuse crisis lines
  - [ ] Postpartum support
  - [ ] Deaf and hard of hearing accessibility (TTY numbers, video relay)

- [ ] Research text-based and online resources
  - [ ] Text crisis lines by country
  - [ ] Web chat services
  - [ ] Email support services
  - [ ] Mobile apps with crisis support

- [ ] Research language-specific resources
  - [ ] Spanish-language resources in US and Latin America
  - [ ] French-language resources in Canada, France, Africa
  - [ ] Multilingual services in multicultural countries

### Stage: Verification & Quality Assurance

- [ ] Verify all phone numbers
  - [ ] Cross-reference with multiple authoritative sources
  - [ ] Check official government/NGO websites
  - [ ] Document source URL and verification date for each entry
  - [ ] Flag any numbers that appear in multiple conflicting sources

- [ ] Verify service characteristics
  - [ ] 24/7 availability vs limited hours
  - [ ] Free vs toll charges
  - [ ] Language availability
  - [ ] Geographic restrictions (e.g., US toll-free not accessible internationally)

- [ ] Test accessibility where possible
  - [ ] Check that websites are live and functional
  - [ ] Verify chat services are operational
  - [ ] Note any services that are temporarily suspended

- [ ] Document gaps and limitations
  - [ ] Countries with no identified crisis resources
  - [ ] Regions with only limited-hour services
  - [ ] Languages with limited support
  - [ ] Populations without specialized resources

### Stage: Write Comprehensive Documentation

Following WRITE_EVERGREEN_DOC.md structure:

- [ ] Create `docs/reference/GLOBAL_CRISIS_RESOURCES.md`
  - [ ] Introduction explaining purpose and scope
  - [ ] How to use this resource (for developers and users)
  - [ ] Verification methodology and update schedule
  - [ ] Important disclaimers about accuracy and emergencies

- [ ] Organize resources by region
  - [ ] International/multi-country resources first
  - [ ] Then by continent and country
  - [ ] Within countries: national resources, then specialized populations, then regional
  - [ ] Clear hierarchy and table of contents for easy navigation

- [ ] Create structured entries with consistent format
  - [ ] Country/Region name
  - [ ] Service name
  - [ ] Phone number(s)
  - [ ] Text/SMS options
  - [ ] Web chat URL
  - [ ] Email (if available)
  - [ ] Hours of operation
  - [ ] Languages supported
  - [ ] Special populations served
  - [ ] Geographic restrictions
  - [ ] Source URL
  - [ ] Last verified date

- [ ] Add implementation guidance section
  - [ ] How to integrate into app boundaries footer
  - [ ] Geographic detection strategies (or manual selection)
  - [ ] When to surface resources during dialogue
  - [ ] Displaying resources without breaking user flow
  - [ ] Accessibility considerations for voice-first interface

- [ ] Include comprehensive references
  - [ ] Links to all source websites
  - [ ] Citations for research on crisis resources
  - [ ] Related internal documentation
  - [ ] Standards and frameworks (NICE, WHO)

- [ ] Add "See also" section linking to:
  - [ ] `SAFEGUARDING.md`
  - [ ] `PRODUCT_VISION_FEATURES.md`
  - [ ] Research on suicide prevention strategies
  - [ ] Implementation planning docs

### Stage: Create Verification Update Instructions

- [ ] Write `gjdutils/docs/instructions/UPDATE_CRISIS_RESOURCES.md`
  - [ ] Purpose: prompt/instructions for checking and updating crisis resources
  - [ ] Verification schedule (suggest quarterly)
  - [ ] Step-by-step verification process
  - [ ] How to check each phone number is still active
  - [ ] How to verify websites are operational
  - [ ] How to identify service changes or discontinuations
  - [ ] How to add newly discovered resources
  - [ ] How to document verification in commit message

- [ ] Include web search strategies
  - [ ] Search patterns for finding updates
  - [ ] Official source websites to check
  - [ ] News sources for service closures/changes
  - [ ] How to cross-reference multiple sources

- [ ] Define quality standards
  - [ ] Minimum number of authoritative sources required
  - [ ] Acceptable source types (government, established NGO, academic)
  - [ ] Unacceptable sources (unverified aggregators, forums)
  - [ ] Documentation requirements for each change

- [ ] Create checklist template for verification runs
  - [ ] List of countries/regions to verify
  - [ ] Checkboxes for each verification step
  - [ ] Space to note changes or issues
  - [ ] Template for commit message

### Stage: Implementation Planning

- [ ] Review current app architecture for resource display
  - [ ] Check how boundaries footer is currently shown
  - [ ] Identify where in dialogue flow resources should surface
  - [ ] Consider CLI command for direct access (e.g., `help crisis`)

- [ ] Design geographic selection approach
  - [ ] Option A: Ask user to select country at first launch
  - [ ] Option B: Auto-detect from system locale (with manual override)
  - [ ] Option C: Always show international + selected country resources
  - [ ] Discuss trade-offs with user before implementing

- [ ] Plan crisis indicator integration
  - [ ] When pattern-based detection triggers, show appropriate resources
  - [ ] Make resources available without requiring crisis detection
  - [ ] Consider user privacy re: logging crisis resource access

- [ ] Consider structured data implementation
  - [ ] Store resources in YAML/JSON for programmatic access
  - [ ] Generate markdown documentation from structured data
  - [ ] Enable filtering by country, language, population, availability

### Stage: Review & Refinement

- [ ] Stop and review with user
  - [ ] Share `GLOBAL_CRISIS_RESOURCES.md` draft
  - [ ] Discuss any surprising findings or gaps
  - [ ] Confirm geographic priorities
  - [ ] Review verification methodology

- [ ] Gather feedback on implementation approach
  - [ ] How to surface resources in CLI (command? automatic?)
  - [ ] Geographic selection method
  - [ ] Frequency of showing resources (every session? on request? when detected?)

- [ ] Discuss ongoing maintenance plan
  - [ ] Who will run quarterly verifications?
  - [ ] Process for user-reported errors
  - [ ] How to handle deprecated resources

### Stage: Final Quality Checks

- [ ] Verify all links are functional (use script or manual check)
- [ ] Check consistent formatting throughout document
- [ ] Ensure all entries have verification dates
- [ ] Confirm all countries have at least one verified resource (or explicitly note gap)
- [ ] Proofread for clarity and accuracy
- [ ] Validate against WRITE_EVERGREEN_DOC.md standards

### Stage: Documentation & Knowledge Sharing

- [ ] Update `SAFEGUARDING.md` to reference new resource catalogue
  - [ ] Add link in "Guardrails (operational)" section
  - [ ] Update "Boundaries footer" to reference verified resources

- [ ] Update `PRODUCT_VISION_FEATURES.md` if relevant
  - [ ] Note enhanced safety features in "Current Implementation"

- [ ] Update `RESEARCH_TOPICS.md` if applicable
  - [ ] Mark "Identifying at-risk users" with progress
  - [ ] Cross-reference new documentation

- [ ] Git commit following project conventions
  - [ ] Use format: `docs: add comprehensive global crisis resources catalogue with verification methodology`
  - [ ] Include detailed commit body explaining scope and sources
  - [ ] Reference issue/planning doc if applicable

### Stage: Long-term Maintenance Setup

- [ ] Schedule first verification check (3 months from completion)
- [ ] Create reminder system for quarterly verifications
- [ ] Document process in project maintenance procedures
- [ ] Consider automation opportunities
  - [ ] Script to check if URLs are still live
  - [ ] Automated reminder for verification schedule
  - [ ] Template generator for new resource entries

## Appendix

### Research Approach Notes

**Parallel Execution:**
Following `SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md`, use parallel web searches for efficiency when researching multiple geographic regions.

**Source Quality Hierarchy:**
1. Official government health/crisis services
2. Established international NGOs (IASP, Befrienders Worldwide)
3. National crisis line organizations
4. Regional/local official services
5. Academic/research institution compilations
6. Vetted aggregator sites (with verification)

**Documentation Quality:**
- Minimum 10-15 references per major region
- URLs for all sources
- Verification dates clearly marked
- Implementation readiness assessed
- Cultural considerations noted

### Key Statistics from Research

From conversation research findings:
- 93% of suicide prevention apps did NOT incorporate all six evidence-based strategies
- 9% of apps provided erroneous crisis helpline numbers
- 6 apps with wrong numbers had 1M+ downloads each
- 35% of DMHI studies collected NO safety data
- Digital safety plans showed 50% reduction in ED visits when activated

### Important Considerations

**Voice-First Interface Challenges:**
- Reading long phone numbers verbally is error-prone
- Consider "text you the number" or "show on screen" options
- Voice command for direct access: "show crisis help"
- CLI should display resources visually even in voice mode

**Privacy & Logging:**
- Accessing crisis resources is sensitive
- Consider whether to log access (for safety monitoring vs privacy)
- Anonymous logging option: "crisis resources accessed" without details
- Clear privacy policy about what is/isn't logged

**Cultural Sensitivity:**
- Crisis service availability varies globally
- Some cultures have stigma around mental health help-seeking
- Language barriers significant in many regions
- Consider phrasing that respects cultural differences

**Edge Cases:**
- User traveling internationally (phone not accessible from abroad)
- User in region with no identified resources
- User in country with government restrictions on mental health services
- Services temporarily unavailable or discontinued

### Related Research

This work connects to:
- Pattern-based crisis detection (future work)
- Safety plan development feature (evidence-based, 50% ED reduction)
- Adverse event monitoring approach
- Exclusion criteria and contraindications documentation
- Terms of service and liability considerations

### Alternative Approaches Considered

**Option A: Use existing aggregator**
- Pros: Less work, maintained by others
- Cons: 9% had wrong numbers; reliability unclear; may go offline
- Decision: Build our own verified list with sources

**Option B: Only provide international resources**
- Pros: Simpler, no geographic detection needed
- Cons: Local resources often better, language-appropriate, culturally aware
- Decision: Provide both international and country-specific

**Option C: Require professional screening before app use**
- Pros: Reduced liability, appropriate triage
- Cons: Major barrier to access; may prevent any help for vulnerable users
- Decision: Self-service with clear boundaries and easy resource access

### User Quotes & Context

"It's journalling. But let's hold ourselves to a higher standard nonetheless, because (as you say) users will ignore the distinction, and to lead by example."

"I want to make this widely available (e.g. via PyPI, uvx), for free, including to people struggling with mental health."

"I'm in the UK, and as a rule I would trust/default to NICE, but we should draw from anything good from anywhere."

"Users could be based anywhere in the world."

"I don't have insurance. At the moment, I'm not releasing this via a company."

These constraints mean:
- High ethical standards required despite no legal requirement
- Global scope essential
- Accuracy critical due to vulnerable user base and personal liability
- Free/open source aligns with accessibility mission
- Risk mitigation through excellent documentation and verified resources
