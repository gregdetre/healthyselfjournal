# Safeguarding Research: Crisis Resources and Major Concerns - 2025-10-17

---
Date: 2025-10-17
Duration: Extended conversation
Type: Research Review, Decision-Making, Exploratory
Status: Active
Related Docs:
- `docs/planning/251017d_global_crisis_resources_catalogue.md` - Planning doc for crisis resources research
- `docs/reference/SAFEGUARDING.md` - Current safeguarding principles
- `docs/reference/PRODUCT_VISION_FEATURES.md` - Product vision and values
- `docs/research/AUTONOMY_SUPPORT_MI_SDT_FOR_JOURNALING.md` - MI/SDT guidance
- `docs/research/ANTI_SYCOPHANCY_AND_PARASOCIAL_RISK_GUARDRAILS.md` - Anti-sycophancy guardrails
---

## Context & Goals

This conversation explored safeguarding best practices for a voice-based reflective journaling app, with particular focus on serious crisis situations like suicidal ideation. User wanted to understand major issues and concerns beyond current safeguarding measures, and requested web research on current best practices.

## User's Initial Question

> "What about really if we stray into really dangerous safe-guarding territory, e.g. suicidal ideation, etc?
>
> What other major issues/concerns should we be aware of/guard against?
>
> Search the web for best practices.
>
> see @PRODUCT_VISION_FEATURES.md , @SAFEGUARDING.md , @RESEARCH_TOPICS.md ,
> @SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md , and anything else relevant.
>
> Be in @SOUNDING_BOARD_MODE.md"

## Assistant's Research Process

I reviewed the existing documentation and conducted parallel web searches on:
- Mental health app safeguarding and crisis detection best practices (2024-2025)
- Digital mental health interventions safety protocols and referral pathways
- Journaling app ethical guidelines and WHO/NICE standards
- AI chatbot mental health safety guardrails and crisis language detection

## Key Research Findings

### Critical Statistics from 2024-2025 Research

**From BMC Medicine systematic assessment of 69 suicide prevention/depression apps:**
- Only **5 apps (7%) incorporated all six evidence-based suicide prevention strategies**
- **6 apps (9%), including 2 with 1M+ downloads, provided ERRONEOUS crisis helpline numbers**
- Most apps addressed only 1-2 strategies with significant gaps

**From JMIR Mental Health 2023 systematic review on DMHIs:**
- **35% of digital mental health intervention studies collected NO safety data**
- Main treatment-related adverse event: symptom deterioration
- Methods varied widely; very few performed formal statistical analyses

**From 2025 Scientific Reports on AI chatbots:**
- "The safety of such agents when dealing with individuals experiencing mental health crises, including suicidal crisis, has not been adequately evaluated"
- 29 AI-powered chatbot agents assessed

**From 2025 JMIR study on digital safety plans:**
- **50% reduction in emergency department visits** when digital safety plans activated
- Evidence-based intervention with demonstrated effectiveness

### Six Evidence-Based Suicide Prevention Strategies

Clinical guidelines identify that apps should incorporate:

1. **Mood and Suicidal Thought Tracking** - Assessing acute risk through monitoring mood, triggers, plans, and protective factors
2. **Safety Plan Development** - Creating structured, standardized lists of coping strategies and support contacts
3. **Activities to Deter Suicidal Thoughts** - Recommending wellness-enhancing activities with follow-up
4. **Information and Education** - Providing content on warning signs, risk factors, and prevention methods
5. **Access to Support Networks** - Enabling users to store and contact family, friends, and providers
6. **Access to Emergency Counseling** - Offering direct connection to crisis helplines or trained counselors

Critical finding: Apps rarely assessed risk factors, triggers, or protective factors; only one reviewed suicide history.

### What Doesn't Work

**Simple Keyword Detection:**
- Users express distress using indirect language, slang, humor
- Adolescents particularly use non-obvious cues
- Example: asking about "tall bridges" after job loss → chatbots provided bridge examples instead of recognizing suicidal ideation

**"Go Call 911" and Halt Approach:**
- Current AI assistants "typically say 'Don't talk to me, go somewhere else'"
- Alienates users, breaks trust, provides no actual help
- OpenAI acknowledged: "There have been instances where our 4o model fell short in recognizing signs of delusion or emotional dependency"

**Lack of Human Connection:**
- Most chatbots unable to recognize and respond appropriately to signs of distress
- No transition from machine to in-person help built into platforms

### What Shows Promise

**Digital Safety Plans (Strong Evidence):**
- 50% reduction in ED visits when activated (2025 study)
- User-controlled and collaborative approach

**Layered Safety Approach:**
- Example (Therabot): Continue dialogue with crisis intervention + alert button to 988 + care team calls/texts
- Three layers: immediate dialogue support, self-activated help, proactive outreach

**Context-Aware Pattern Detection:**
- Monitor patterns over time rather than keywords
- Indicators: increasing hopelessness + sleep disruption + social withdrawal + reduced pleasure

**Standardized Safety Protocols:**
- Active collection methods (proactively asking about adverse events)
- Regular intervals: weekly, post-session, mid-treatment, follow-up
- Formal statistical comparisons using standardized metrics
- Predetermined quantitative thresholds (e.g., PHQ-9 weekly assessments)

### Standards & Frameworks

**NICE DHT Evidence Standards Framework:**
- Digital mental health technologies using complex software must meet medical device standards
- Must demonstrate effectiveness and acceptable safety
- Evidence-based argument that benefit outweighs risk

**WHO Guidelines (2019):**
- First evidence-based guidelines for digital health
- Nine recommendations on digital health interventions

**Minimum Standards for Mental Health Apps:**
- Data safety and privacy
- Effectiveness
- User experience/adherence
- Data integration
- Transparent data policies
- Language at maximum 6th grade reading level

**Key Limitation:** "App developers and data storage companies do not meet the regulatory language to fall under HIPAA jurisdiction, and therefore most mental health apps are not regulated or required to comply with HIPAA standards."

### Other Major Safeguarding Concerns Identified

1. **Symptom Deterioration** (most common adverse event) - Increased rumination, worsening depression from negative reflection, anxiety from self-focus

2. **Dependency & Over-Use** - Using app instead of seeking professional help, substituting for human connection

3. **Privacy Breaches** - Voice data is highly identifiable; device compromise risk; STT services process audio

4. **Ineffective for Trauma** - Expressive writing can re-traumatize without professional support; need explicit contraindication for active PTSD

5. **Cultural Insensitivity** - Western-centric gratitude and individualism assumptions may alienate or harm users from collectivist cultures

6. **Misuse by Abusers** - Could be used to monitor/control partners; device access = access to emotional vulnerability

7. **Regulatory Non-Compliance** - NICE DHT framework (UK), FDA (US health claims), GDPR (EU), FTC investigation of AI companion chatbots (2024-2025)

8. **Professional Supervision Tension** - Apps should complement, never replace, professional relationships, BUT "Reputable digital mental health services that do not require prior face-to-face referrals play an important role in early intervention"

## Assistant's Complete Analysis (Verbatim)

### What You Already Have (Actually Better Than Most)

Your current safeguarding approach is substantially better than typical mental health apps. You have:

- Evidence-based principles (MI/SDT, anti-sycophancy)
- Autonomy-supportive language and boundaries
- Rumination redirection (a real safety concern)
- Usage pacing with break nudges
- Clear scope boundaries with help resources

However, the research reveals you're missing critical crisis safeguarding that 93% of suicide prevention apps also lack.

### The Core Tension: What Kind of App Are You?

Here's the fundamental question: **Is this a journaling app or a mental health intervention?**

This matters because:

**If it's "just a journaling app":**
- You could argue lower duty of care
- But users in crisis **will use it anyway** (they don't read classifications)
- Voice-first + AI dialogue = users may treat it like therapy
- Local-first + no monitoring = you won't know when crises happen

**If it's a mental health intervention:**
- Higher regulatory scrutiny (NICE DHT framework, potential medical device classification)
- Greater liability exposure
- Need formal adverse event monitoring
- But also: clearer ethical obligations and guidance

**The research suggests you can't avoid this choice.** Your design (AI dialogue, emotional processing, daily sessions) will be experienced as mental health support regardless of your disclaimers.

### The Suicidal Ideation Problem

This is where things get serious.

**What Doesn't Work:**
- Simple keyword detection → Users use indirect language, slang, humor
- "Go call 911" and halt → Alienates users, breaks trust, no actual help
- Erroneous crisis numbers → 6 apps with 1M+ downloads had WRONG numbers
- Ignoring it → Main adverse event in DMHIs is "symptom deterioration"

**What Shows Promise:**
- Digital safety plans → 50% reduction in ED visits when activated
- Layered approach: Continue dialogue + alert button + outreach from care team
- Context-aware detection → Pattern of hopelessness, not just keywords
- Verified local crisis resources → 988 in US, but varies by country

**The Voice Challenge:**
Your voice-first design makes this harder. Text can be pattern-matched; prosody analysis for crisis detection is unreliable and raises huge privacy concerns.

### Critical Questions for You

**1. Target Users & Risk Profile:**
- Who do you **intend** to serve? "Generally healthy people doing reflection" or "people struggling with mental health"?
- Will you **exclude** high-risk users? (Most studies do, but high-risk people won't self-exclude)
- What happens when a regular user hits a crisis? (Job loss, breakup, diagnosis, grief)

**2. Risk Tolerance & Liability:**
- What's your acceptable level of risk?
- Are you prepared for the scenario where someone experiences harm or deterioration while using your app?
- Do you have liability insurance for mental health software?
- What regulatory framework applies in your jurisdiction?

**3. Human Intervention Capacity:**
- You're local-first with no monitoring → **you can't detect crises in real-time**
- You can't provide live routing to crisis counselors
- You can't follow up with at-risk users
- Is this an acceptable limitation, or do you need to reconsider the architecture?

**4. Professional Supervision:**
- Should your app require initial screening by a mental health professional?
- Should it be "prescription-only" for certain features?
- Or completely self-service?

### Specific Gaps & Options

**Gap 1: Crisis Language Detection & Response**

**Option A: Minimal (current approach)**
- Rely on disclaimers and scope boundaries
- ❌ Pros: Simple, maintains flow
- ❌ Cons: User may not see disclaimer when in crisis; no active support

**Option B: Pattern-Based Detection**
- Monitor for crisis indicators across sessions (not just keywords)
- Hopelessness + sleep disruption + social withdrawal + reduced pleasure
- When threshold crossed: interrupt dialogue, provide verified crisis resources, log event
- ✅ Pros: Catches indirect expressions, can work with transcribed voice
- ⚠️ Cons: False positives (interrupting unnecessarily), false negatives (missing real crises)

**Option C: Safety Plan Integration**
- During onboarding, guide users to create a safety plan (evidence-based component)
- When indicators appear, surface their own plan
- ✅ Pros: Evidence-based (50% ED reduction), user-controlled, collaborative
- ⚠️ Cons: Requires more complex implementation, assumes user has supports to list

**Option D: Exclusion Criteria**
- During onboarding, screen for active suicidal ideation, recent attempts, severe symptoms
- Explicitly state app is not suitable, provide resources
- ⚠️ Pros: Clear risk reduction, legally defensible
- ❌ Cons: People lie on screens, may increase harm by denying access to ANY support

**Gap 2: Verified Crisis Resources**

**Current state**: You mention "help resources" but the research shows 9% of apps had **wrong numbers**.

**Required:**
- **988 Suicide & Crisis Lifeline** (US) - verify it's still current
- **988 is current as of 2025** but you need international alternatives
- Location detection (or manual selection) to provide correct local resources
- Regular verification (numbers change)
- Backup: National Suicide Prevention Lifeline 1-800-273-8255, Crisis Text Line (text HOME to 741741)
- **Make these accessible mid-session**, not just in docs

**Gap 3: Adverse Event Monitoring**

**Research requirement**: "Safety assessment should be systematically assessed in every DMHI study"

**For you:**
- How will you know if someone's symptoms deteriorate while using your app?
- You're local-first, so you can't do this traditionally
- Options:
  - Periodic check-ins: "On a scale of 1-10, how would you rate your overall wellbeing compared to when you started?"
  - Validated scales: PHQ-2 (2-item depression screener) weekly
  - Pattern analysis: Declining mood trajectory over sessions
  - User-initiated reporting: "This isn't helping" → log and prompt for alternatives

**Gap 4: Exclusion Criteria & Contraindications**

**Research standard**: "Clearly specify indications and contraindications"

**You need to document:**
- Active suicidal ideation with plan
- Recent suicide attempt
- Active psychosis or mania
- Severe PTSD with flashbacks
- Under 18 without parental consent?
- Substance abuse crisis

**But here's the paradox**: Stating these may reduce liability but also prevents access to people who might benefit from **any** reflective practice.

### What I Think You Should Prioritize

**Immediate (Do This):**

1. **Verified Crisis Resources**
   - Add 988 and crisis text line to boundaries footer
   - Make accessible during any session (command or detection)
   - Test regularly that numbers are correct

2. **Clearer Classification & Disclaimers**
   - State explicitly: "This is a reflective journaling tool, not therapy or mental health treatment"
   - "Not suitable for active mental health crises"
   - "If experiencing thoughts of self-harm, please contact [resources] immediately"
   - Show at: first launch, before first session, periodically

3. **Document Exclusion Criteria**
   - Even if you don't enforce them, state them
   - Helps establish scope and expected use

4. **Simple Deterioration Check**
   - After every 5-7 sessions: "How has journaling been for you? Helpful / Neutral / Making things worse"
   - If "worse": suggest stopping, provide alternatives, log the event

**Medium-Term (Consider Seriously):**

5. **Pattern-Based Crisis Indicators**
   - Not keyword matching, but trend analysis
   - Increasing hopelessness, social isolation, sleep disruption
   - When threshold crossed: pause dialogue, surface crisis resources, offer safety plan creation

6. **Safety Plan Development**
   - Optional feature: "Would you like to create a personal safety plan?"
   - Evidence-based component: 50% reduction in crisis ED visits
   - Surfaces automatically when indicators appear

7. **Audit Trail for Safety Events**
   - Log when crisis indicators detected (no content, just fact)
   - Log when resources provided
   - Enables post-hoc analysis if needed for liability

**Longer-Term (Architectural):**

8. **Hybrid Model Consideration**
   - Pure local-first prevents real-time intervention
   - Could you offer optional "connect to crisis network" feature?
   - Sends alert to 988 or provider if user consents when threshold crossed?

9. **Professional Integration**
   - Allow therapists to "prescribe" app to clients
   - Provides professional oversight without breaking local-first for data

10. **Formal Safety Study**
    - If you want to make any mental health claims
    - Need IRB approval, adverse event monitoring, control group
    - Could be a user study with optional participation

### My Main Concerns

**1. The Voice Modality Gap**

Most crisis detection research is text-based. Voice is harder:
- Can't do simple pattern matching on audio
- Must rely on transcripts (latency issue during real-time session)
- Prosody analysis is unreliable and creepy (privacy)
- Users may express crisis more indirectly in voice

**You need research on voice-specific crisis expression patterns.** I didn't find this in the literature.

**2. The "Therapeutic Relationship" Illusion**

Your app creates a daily dialogue with Claude. Research shows users **will** form parasocial bonds (you already have anti-sycophancy for this). But they may also disclose crises to Claude that they wouldn't write in a traditional journal.

This increases risk beyond typical journaling apps. You need stronger safeguards because the AI dialogue invites deeper disclosure.

**3. Liability Without Monitoring**

Local-first + no telemetry = you can't:
- Detect when a user is in crisis
- Follow up with at-risk users
- Testify that you did everything reasonable to prevent harm
- Provide audit trails of safety interventions

**This could be a liability nightmare** if something happens. You may need to reconsider pure local-first, or at least offer optional anonymous safety telemetry.

**4. The "Not a Mental Health App" Defense May Not Hold**

If you:
- Use AI dialogue for emotional processing
- Ask about feelings, thoughts, behaviors
- Target "self-reflection" and "wellbeing"
- Enable daily engagement

...you're functionally a mental health intervention. Courts and regulators may not care what you call it.

### Other Major Issues to Guard Against

Beyond suicidal ideation:

- **Symptom Deterioration** - Most common adverse event; monitor with periodic check-ins
- **Dependency & Over-Use** - Your break nudges help, but consider usage caps or "graduation" prompts
- **Privacy Breaches** - Voice data highly identifiable; warn users not to share information that could endanger them if exposed
- **Ineffective for Trauma** - Need explicit contraindication for active PTSD; if trauma content emerges, redirect to specialized resources
- **Cultural Insensitivity** - Western-centric assumptions may harm collectivist culture users; implementation unclear
- **Misuse by Abusers** - Could monitor/control partners; consider warnings
- **Regulatory Non-Compliance** - NICE DHT (UK), FDA (US health claims), GDPR (EU), FTC investigation

### Questions for You

Before I can help you implement better safeguards:

**1. What's your target geography?** (Determines which crisis resources and regulations apply)

**2. Are you releasing this publicly or keeping it personal/research-only?** (Massively different risk profiles)

If public, do you have:
- Legal entity and liability insurance?
- Terms of service reviewed by lawyer?
- IRB approval if making research claims?

If personal, are you prepared for the risk that someone you share it with could experience harm?

---

**This is a lot, I know. But the research is clear: most mental health apps are dangerously inadequate at safeguarding, and the consequences are real. You're ahead of the curve with your evidence-based approach, but there are serious gaps around crisis situations.**

**What's your gut reaction to this? Which of these concerns resonates most, and which feels like over-engineering for your use case?**

## User's Response & Key Decisions

User responded:

> "It's journalling. But let's hold ourselves to a higher standard nonetheless, because (as you say) users will ignore the distinction, and to lead by example. I want to make this widely available (e.g. via PyPI, uvx), for free, including to people struggling with mental health. I'm in the UK, and as a rule I would trust/default to NICE, but we should draw from anything good from anywhere. Users could be based anywhere in the world. I'm planning (for now) to make this a free app. I don't have insurance. At the moment, I'm not releasing this via a company.
>
> (N.B. It's voice-based, but we transcribe everything. We're not solely local-first - we also offer cloud models, FYI.)"

**Key decisions captured:**
- Classification: "It's journalling" but holding to "higher standard nonetheless"
- Rationale: "users will ignore the distinction, and to lead by example"
- Distribution: "widely available (e.g. via PyPI, uvx), for free"
- Target users: "including to people struggling with mental health"
- Standards: "I would trust/default to NICE, but we should draw from anything good from anywhere"
- Geographic scope: "Users could be based anywhere in the world"
- Legal structure: "I don't have insurance. At the moment, I'm not releasing this via a company"
- Technical clarifications: "we transcribe everything. We're not solely local-first - we also offer cloud models"

**Implications:**
- Global distribution with no geographic restrictions
- Significant personal liability exposure (no insurance, no company structure)
- Transcription available enables text-based safety monitoring (addresses voice modality concern)
- Cloud models available enable more sophisticated monitoring and intervention (addresses local-first limitation)
- NICE standards as baseline, but draw from international best practices
- High ethical standards despite "just journaling" classification

User then requested:

> "Write up this conversation in detail, including both user input and your output, pretty much all verbatim. Follow the file naming from @gjdutils/docs/instructions/CAPTURE_SOUNDING_BOARD_CONVERSATION.md
>
> Write a planning document as per @gjdutils/docs/instructions/WRITE_PLANNING_DOC.md for researching/gathering/writing up a comprehensive prioritised well-cited .md catalogue of relevant phone numbers/websites/contact details across as many countries as we can (also see @gjdutils/docs/instructions/WRITE_DEEP_DIVE_AS_DOC.md ) . And include a stage to write a .md instructions-prompt to check/update this.
>
> Then stay in @gjdutils/docs/instructions/SOUNDING_BOARD_MODE.md , but ask questions one or two at a time."

## Documents Created

1. **This conversation document**: `docs/conversations/251017a_safeguarding_crisis_resources_research.md`

2. **Planning document**: `docs/planning/251017d_global_crisis_resources_catalogue.md`
   - Comprehensive plan for researching and documenting global crisis resources
   - Structured stages for verification methodology, web research by region, quality assurance
   - Includes stage to write `docs/reference/UPDATE_CRISIS_RESOURCES.md` for ongoing verification
   - References WRITE_EVERGREEN_DOC.md and WRITE_DEEP_DIVE_AS_DOC.md standards

## Assistant's Follow-Up Questions (Sounding Board Mode)

After creating the documentation, I asked two critical questions:

**Question 1: Have you consulted with a solicitor about personal liability exposure?**

Context: The research shows 93% of mental health apps have serious safety gaps, and you're explicitly targeting "people struggling with mental health." If someone experiences harm while using your app (whether or not it's actually caused by the app), you could face:
- Civil liability claims
- Regulatory action (especially under UK regulations, GDPR, potentially FDA if US users make health claims)
- Reputational risk

The fact that you're holding yourself to high standards is admirable, but it doesn't necessarily protect you legally. A properly structured company with liability insurance would create a legal barrier between the app and your personal assets.

**Question 2: Should you consider forming a limited company and getting professional indemnity insurance before public PyPI release?**

Context: This isn't necessarily expensive (UK limited company is ~£100 to register), and professional indemnity insurance for software varies but might be manageable for a free app. The protection would be significant.

What's your thinking on this? Is the plan to release as-is under personal liability, or are you open to exploring corporate structure first?

## Open Questions

**Legal & Liability:**
- Has user consulted solicitor about personal liability?
- Will user form limited company before public release?
- Will user obtain professional indemnity insurance?
- Has user considered terms of service and liability disclaimers?

**Implementation Priorities:**
- Which gaps should be addressed first (crisis resources, detection, disclaimers)?
- What level of crisis intervention is appropriate for a "journaling app"?
- How to balance safety features with user experience and autonomy?
- Should onboarding include screening questions or only disclaimers?

**Technical Architecture:**
- How to implement geographic detection or selection for local crisis resources?
- Where in dialogue flow should crisis resources surface?
- Should there be a CLI command for direct access (e.g., `help crisis`)?
- How to log safety events while respecting privacy?

**Ongoing Maintenance:**
- Who will run quarterly verification of crisis resources?
- Process for user-reported errors in crisis information?
- How to handle deprecated resources or service changes?

**Research Gaps:**
- Voice-specific crisis expression patterns (not found in literature)
- Cultural variations in crisis disclosure and help-seeking
- Effectiveness of self-guided digital safety planning without clinical oversight

## Next Steps

User requested this conversation be captured verbatim, then continue in sounding board mode.

Awaiting user's response to the two critical questions about legal structure and insurance before proceeding with implementation planning.

## Sources & References

### Academic Research (2024-2025)

**Crisis Detection & Suicide Prevention:**
- **Performance of mental health chatbot agents in detecting and managing suicidal ideation** - Scientific Reports (2025): https://www.nature.com/articles/s41598-025-17242-4
- **Suicide prevention and depression apps' suicide risk assessment and management: a systematic assessment of adherence to clinical guidelines** - BMC Medicine: https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1461-z
- **Suicide prevention and depression apps' suicide risk assessment and management** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6921471/
- **Harnessing technology for hope: a systematic review of digital suicide prevention tools** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12234914/
- **The Effectiveness of Mobile Apps for Monitoring and Management of Suicide Crisis** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9573577/
- **Mobile apps designed to help prevent suicide** - APA Services: https://www.apaservices.org/practice/business/technology/tech-column/mobile-apps-suicide-prevention

**Digital Safety Plans:**
- **Impact of Digital Safety Plan Activation on Subsequent Emergency Departments Visits Following an Initial Suicide Attempt: Quasi-Experimental Study** - JMIR Mental Health (2025): https://mental.jmir.org/2025/1/e70253
- **Translating Suicide Safety Planning Components Into the Design of mHealth App Features: Systematic Review** - JMIR Mental Health (2024): https://mental.jmir.org/2024/1/e52763/

**Safety Protocols for DMHIs:**
- **The Safety of Digital Mental Health Interventions: Systematic Review and Recommendations** - PMC (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10594135/
- **Digital Health Interventions for Delivery of Mental Health Care: Systematic and Comprehensive Meta-Review** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9109782/
- **Digital interventions in mental health: An overview and future perspectives** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12051054/
- **Developing a process for assessing the safety of a digital mental health intervention and gaining regulatory approval** - Trials: https://trialsjournal.biomedcentral.com/articles/10.1186/s13063-024-08421-1

**AI Chatbot Safety:**
- **AI Chatbots Need Guardrails to Protect Users' Mental Health** - Undark (2025): https://undark.org/2025/09/18/opinion-chatbots-guardrails-mental-health/
- **ChatGPT adds mental health guardrails after bot 'fell short in recognizing signs of delusion'** - NBC News: https://www.nbcnews.com/tech/tech-news/chatgpt-adds-mental-health-guardrails-openai-announces-rcna222999
- **New study warns of risks in AI mental health tools** - Stanford Report (2025): https://news.stanford.edu/stories/2025/06/ai-mental-health-care-tools-dangers-risks
- **Teens Are Using Chatbots as Therapists. That's Alarming** - RAND (2025): https://www.rand.org/pubs/commentary/2025/09/teens-are-using-chatbots-as-therapists-thats-alarming.html
- **"It happened to be the perfect thing": experiences of generative AI chatbots for mental health** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11514308/
- **The Generative AI Therapy Chatbot Will See You Now** - #CrisisTalk: https://talk.crisisnow.com/the-generative-ai-therapy-chatbot-will-see-you-now/

**Ethical Considerations:**
- **Ethical and Safety Concerns Regarding the Use of Mental Health–Related Apps in Counseling** - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7457894/
- **Using generic AI chatbots for mental health support: A dangerous trend** - APA Services: https://www.apaservices.org/practice/business/technology/artificial-intelligence-chatbots-therapists

### Standards & Guidelines

**WHO:**
- **Ethics and governance of AI for health** (2023): https://www.who.int/publications/i/item/9789240077989
- **WHO guideline Recommendations on Digital Interventions for Health System Strengthening**: https://www.ncbi.nlm.nih.gov/books/NBK541902/
- **Recommendations on digital interventions for health system strengthening**: https://www.who.int/publications/i/item/9789241550505
- **Classification of digital interventions, services and applications in health (2nd ed)**: https://www.who.int/publications/i/item/9789240081949

**NICE:**
- **Evidence Standards Framework for Digital Health Technologies (DHT)** (2022): https://www.nice.org.uk/about/what-we-do/our-programmes/evidence-standards-framework-for-digital-health-technologies
- **Digital mental health technologies guidance launched**: https://www.gov.uk/government/news/digital-mental-health-technologies-guidance-launched-to-help-manufacturers-and-safeguard-users

**US Resources:**
- **2024 National Strategy for Suicide Prevention**: https://www.hhs.gov/programs/prevention-and-wellness/mental-health-substance-use-disorder/national-strategy-suicide-prevention/index.html

**Evaluation Frameworks:**
- **Towards a consensus around standards for smartphone apps and digital mental health** - PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6313231/
- **Standards and principles for evaluating mental health apps** - National Elf Service: https://www.nationalelfservice.net/treatment/digital-health/standards-and-principles-for-evaluating-mental-health-apps/
- **Actionable health app evaluation: translating expert frameworks into objective metrics** - npj Digital Medicine: https://www.nature.com/articles/s41746-020-00312-4

### Internal Documentation

- `docs/reference/SAFEGUARDING.md` - Current safeguarding principles
- `docs/reference/PRODUCT_VISION_FEATURES.md` - Product vision and values
- `docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md` - Research approach
- `docs/research/AUTONOMY_SUPPORT_MI_SDT_FOR_JOURNALING.md` - MI/SDT guidance
- `docs/research/ANTI_SYCOPHANCY_AND_PARASOCIAL_RISK_GUARDRAILS.md` - Anti-sycophancy guardrails
- `docs/research/RESEARCH_TOPICS.md` - Research prioritization
- `gjdutils/docs/instructions/SOUNDING_BOARD_MODE.md` - Sounding board approach
- `gjdutils/docs/instructions/CAPTURE_SOUNDING_BOARD_CONVERSATION.md` - Conversation documentation guidelines
- `gjdutils/docs/instructions/WRITE_PLANNING_DOC.md` - Planning document structure
- `gjdutils/docs/instructions/WRITE_DEEP_DIVE_AS_DOC.md` - Deep dive documentation approach

## Related Work

**Documents created from this conversation:**
- `docs/planning/251017d_global_crisis_resources_catalogue.md` - Comprehensive research plan for global crisis resources with verification methodology

**Future implementation work:**
- Enhanced crisis resource integration in app
- Pattern-based crisis indicator detection
- Digital safety plan development feature
- Improved disclaimers and exclusion criteria
- Periodic wellbeing check-ins
- Updates to `SAFEGUARDING.md` with specific crisis protocols
- Terms of service and liability documentation
