# Research Topics for Evidence-Based Journaling

see also:
- `docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE.md`
- `../../healthyselfjournal/prompts/question.prompt.md.jinja`

## Research Agent Instructions

When researching topics from this list:

1. **Search comprehensively** for trustworthy sources using detailed web searches
2. **Focus on evidence** from meta-analyses, RCTs, and systematic reviews (2019-2025 preferred for digital interventions)
3. **Write evergreen documentation** following `@gjdutils/docs/instructions/WRITE_EVERGREEN_DOC.md` structure
4. **Save to** `docs/research/TOPIC_NAME.md` with descriptive filenames
5. **Emphasize practical application** for:
   - LLM prompt design for dialogue generation (see `../../healthyselfjournal/prompts/question.prompt.md.jinja` for implementation)
   - Voice-based CLI implementation
   - Safety features and risk mitigation
6. **Include extensive references** with URLs for all claims
7. **Cross-reference** other docs in the research folder

## Completed Research

The following topics have been researched and documented:

### Tier 1 (Completed)
- ✅ Self-distancing techniques → `SELF_DISTANCING_TECHNIQUES.md`
- ✅ Optimal session timing → `OPTIMAL_SESSION_TIMING.md`
- ✅ Structured reflection vs rumination → `STRUCTURED_REFLECTION_VS_RUMINATION.md`
- ✅ Cognitive-emotional integration → `COGNITIVE_EMOTIONAL_INTEGRATION.md`
- ✅ Implementation intentions → `IMPLEMENTATION_INTENTIONS_HABITS.md`

### Tier 2 (Completed)
- ✅ Gratitude practice optimization → `GRATITUDE_PRACTICE_OPTIMIZATION.md`
- ✅ Redemptive narrative construction → `REDEMPTIVE_NARRATIVE_CONSTRUCTION.md`
- ✅ Opening questions → `OPENING_QUESTIONS_FRICTION_REDUCTION.md`
- ✅ Progress tracking → `PROGRESS_TRACKING_STREAK_DESIGN.md`
- ✅ Socratic questioning → `SOCRATIC_QUESTIONING_TECHNIQUES.md`

### Contemplative Practices (Completed)
- ✅ Mindful reflection (Plum Village tradition) → `MINDFUL_REFLECTION_PLUM_VILLAGE.md`
- ✅ Beginning Anew practice → `BEGINNING_ANEW_PRACTICE.md`

### Cognitive Frameworks (Completed)
- ✅ Explanatory style and Three P's (Seligman) → `EXPLANATORY_STYLE_THREE_PS.md`

### Coaching Questions (Completed)

#### Research-Backed (Strong Evidence)
- ✅ Coaching frameworks and evidence → `COACHING_FRAMEWORKS_EVIDENCE.md`
  - ICF competencies research, Socratic questioning RCTs, GROW model studies
  - Solution-Focused Brief Therapy evidence, motivational interviewing data

- ✅ MI/SDT autonomy support for journaling → `AUTONOMY_SUPPORT_MI_SDT_FOR_JOURNALING.md`
  - Autonomy-supportive language patterns; avoid cheerleading; permission-based advice

#### Expert-Endorsed (Moderate Evidence)
- ✅ Daily reflection questions → `DAILY_REFLECTION_QUESTIONS.md`
  - Neil Pasricha framework, Rangan Chatterjee protocol, morning/evening routines
  - Some research backing, widely used in practice
- ✅ Values and self-discovery → `VALUES_SELF_DISCOVERY_QUESTIONS.md`
  - Business Model You framework, identity exploration, strengths assessment
  - Based on positive psychology research, needs more RCT validation

#### Popular/Practical (Variable Evidence)
- ✅ Famous coach questions → `FAMOUS_COACH_QUESTIONS.md`
  - Tim Ferriss (17 questions, fear-setting), Tony Robbins, Marshall Goldsmith
  - Martha Beck, Arthur Brooks failure journal, Jeff Bezos framework
  - Expert credibility but limited controlled studies
- ✅ Annual and quarterly reviews → `ANNUAL_QUARTERLY_REVIEW_QUESTIONS.md`
  - Debbie Millman 5-year vision, career planning, life design questions
  - Anecdotal success stories, needs systematic research

## Prioritization Criteria
- **Effect size**: Clinical significance (d>0.3 high, 0.2-0.3 medium, <0.2 small)
- **Implementation ease**: Natural fit for voice-based CLI interface
- **User engagement**: Likelihood of sustained practice, low friction
- **Evidence quality**: Meta-analyses, RCTs, replication robustness
- **Risk mitigation**: Avoiding harmful patterns and iatrogenic effects
- **Quick wins**: Benefits achievable in <7 days
- **Cultural robustness**: Cross-cultural effectiveness

## Tier 1: High Priority (Strong evidence, large effects, easy implementation)

1. **Self-distancing techniques for emotional regulation**
   - Effect size: Strong neurological evidence (reduced amygdala, increased PFC)
   - Third-person pronouns, temporal distancing, observer perspective
   - Perfect for voice input, immediate benefits

2. **Optimal session duration and timing patterns**
   - 15-20 minute sweet spot preventing rumination
   - Morning vs evening effects on different outcomes
   - Critical for CLI timer features

3. **Structured reflection vs destructive rumination**
   - Distinguishing adaptive vs maladaptive repetitive thinking
   - Concrete/action-focused vs abstract/analytical patterns
   - Core safety feature to prevent harm

4. **Cognitive-emotional integration prompts**
   - Combined processing outperforms emotion-only expression
   - Question sequences that balance feelings and thoughts
   - Essential for dialogue design

5. **Implementation intentions and behavioral triggers**
   - "When-then" planning for habit formation
   - Environmental cues and routine stacking
   - Key for 7% retention problem

## Tier 2: Medium Priority (Good evidence, moderate effects, valuable features)

6. **Gratitude practice optimization**
   - Weekly > daily (avoiding hedonic adaptation)
   - 3-5 elaborated items vs simple lists
   - Cultural considerations for collectivist users

7. **Redemptive narrative construction**
   - Moving from negative to positive story arcs
   - Meaning-making and growth narratives
   - McAdams' narrative identity framework

8. **Opening questions that reduce friction**
   - Lowering activation energy for starting
   - Voice-specific prompts vs writing prompts
   - Chronotype-matched timing suggestions

9. **Progress tracking and streak design**
   - Avoiding toxic gamification
   - Loss aversion vs gain framing
   - Variable reinforcement schedules

10. **Socratic questioning techniques**
    - Evidence from CBT and motivational interviewing
    - Adaptive question branching
    - Avoiding leading or assumptive prompts

## Tier 3: Important but Complex (Strong evidence but implementation challenges)

11. **Voice-specific emotional processing**
    - Unique benefits of speaking vs writing
    - Prosodic features and emotional tone
    - Stream-of-consciousness facilitation

12. **Cultural adaptation strategies**
    - Collectivist vs individualist approaches
    - Avoiding Western-centric assumptions
    - Alternative frameworks to gratitude

13. **Micro-interventions for busy users**
    - 2-3 minute effective practices
    - Single-question reflections
    - Emergency/crisis moment tools

14. **Breaking negative thought patterns**
    - Cognitive restructuring techniques
    - Thought-stopping vs acceptance approaches
    - Rumination circuit breakers

15. **Goal-setting and values clarification**
    - Implementation planning vs outcome goals
    - Values-based vs achievement-based framing
    - Progress celebration patterns

## Tier 4: Specialized Topics (Niche but valuable for specific users)

16. **Sleep quality improvement protocols**
    - Evening gratitude for better sleep
    - Worry postponement techniques
    - Pre-sleep mental clearing

17. **Stress inoculation through writing**
    - Pre-event anxiety management
    - Future-self letters
    - Scenario planning exercises

18. **Social connection through journaling**
    - Gratitude letters (sent and unsent)
    - Relationship reflection prompts
    - Avoiding co-rumination traps

19. **Creative expression integration**
    - Metaphor and imagery use
    - Poetry and free association
    - Non-linear narrative techniques

20. **Trauma-informed approaches**
    - Safety protocols for difficult content
    - Grounding techniques
    - Professional referral triggers

## Tier 5: Emerging/Experimental (Limited evidence but promising)

21. **AI dialogue personalization**
    - Adaptive questioning based on user patterns
    - Style matching and pacing
    - Personality-informed approaches

22. **Mindfulness integration** → `MINDFUL_REFLECTION_PLUM_VILLAGE.md` ✅
    - Present-moment awareness prompts
    - Body scan check-ins
    - Breath-based transitions

23. **Humor and playfulness in reflection**
    - Cognitive flexibility through levity
    - Perspective-taking through humor
    - Avoiding forced positivity

24. **Habit stacking with existing routines**
    - Coffee/commute/bedtime integration
    - Trigger identification algorithms
    - Context-dependent reminders

25. **Multi-modal future features**
    - Voice input with visual feedback
    - Sketch/doodle integration
    - Music/mood tracking

## Tier 6: Risk Mitigation Research (Preventing harm)

- ✅ Anti-sycophancy and parasocial risk guardrails → `ANTI_SYCOPHANCY_AND_PARASOCIAL_RISK_GUARDRAILS.md`
  - Neutral language, anti-praise, break nudges, boundaries; WHO/NICE-aligned

26. **Identifying at-risk users**
    - Crisis detection without diagnosis
    - Referral protocols
    - Liability considerations

27. **Avoiding performative positivity**
    - Authentic vs forced gratitude
    - Negative emotion validation
    - Balanced emotional expression

28. **Privacy and data security**
    - Encryption for voice/text
    - Local-first architecture
    - Consent and transparency

29. **Addiction and over-dependence**
    - Healthy usage patterns
    - Tool vs crutch framing
    - Graduation strategies

30. **Age-appropriate adaptations**
    - Adolescent vs adult approaches
    - Developmental considerations
    - Parental involvement guidelines

## Tier 7: Miscellaneous Suggestions

31. **Clean Language techniques** → `CLEAN_LANGUAGE_TECHNIQUES.md` ✅
    - Non-directive questioning using client's exact words
    - Metaphor exploration and preservation
    - David Grove's 12 basic questions framework
    - Avoiding therapist contamination of client's experience

## Research Methodology Notes

For each area, we should investigate:
- Meta-analyses and systematic reviews (highest priority)
- Individual RCTs with pre-registration
- Longitudinal studies for retention/engagement
- Cross-cultural validation studies
- Neuroscience evidence where applicable
- Qualitative user experience research

Focus on research from 2019-2025 for digital intervention studies, while including foundational earlier work for core psychological principles.

## Next Steps

1. Select top 10-15 areas for immediate deep research
2. Create standardized research template for each area
3. Launch parallel research agents for efficiency
4. Synthesize findings into implementation guidelines
5. Create testable hypotheses for A/B testing

## Appendix: Open Questions, Suggestions, and Concerns

### Open Questions
- How do we balance scientific rigor with user accessibility in prompt design?
- Should we create user profiles/personas to customize approaches?
- What's the optimal way to detect and adapt to cultural differences in real-time?
- How do we measure effectiveness without being intrusive?

### Suggestions
- Consider creating a "research synthesis" document that pulls together key findings across all topics
- Develop a decision tree for selecting appropriate techniques based on user state/needs
- Create a "contraindications matrix" showing which techniques to avoid in specific situations
- Build a prompt library organized by emotional states, goals, and user characteristics

### Concerns
- **Over-reliance on Western research**: Most studies come from WEIRD populations (Western, Educated, Industrialized, Rich, Democratic)
- **Individual variation**: Effect sizes are averages; what works for one person may harm another
- **Implementation fidelity**: Translating research protocols to voice-based CLI may change effectiveness
- **Ethical considerations**: Need clear boundaries about when to refer users to professional help
- **Privacy implications**: Voice data is particularly sensitive and identifiable
- **Risk of harm**: Some techniques (e.g., trauma processing) require professional guidance