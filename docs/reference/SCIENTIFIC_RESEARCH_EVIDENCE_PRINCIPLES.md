# Scientific research & evidence principles

see also:
- `docs/research/RESEARCH_TOPICS.md` - master list of research areas
- `docs/research/` - completed research documentation
- `docs/conversations/250917b_evidence_based_journaling_research_planning.md` - initial research planning

## Research approach for new topics

When conducting evidence-based research on journaling practices:

### 1. Search strategy
- **Comprehensive web searches** focusing on academic sources, meta-analyses, and recent studies (2019-2025 for digital interventions)
- **Multiple search angles**: effect sizes, implementation methods, cultural variations, contraindications
- **Primary sources**: Look for foundational researchers and seminal studies in each domain
- **Cross-domain connections**: Identify relationships to existing research docs

### 2. Documentation structure
Follow `@gjdutils/docs/instructions/WRITE_EVERGREEN_DOC.md` with emphasis on:
- Clear introduction contextualizing for journaling
- Comprehensive "See also" sections linking related research
- Evidence-based sections with effect sizes and study quality
- Practical implementation for voice-based CLI
- LLM prompt design implications
- Safety considerations and contraindications
- Extensive references with verified URLs

### 3. Parallel execution
For efficiency when researching multiple topics:
- Launch research agents in parallel using the Task tool
- Batch related topics together (e.g., all mindfulness practices, all cognitive techniques)
- Provide detailed prompts specifying exact research questions
- Request cross-references between related topics

### 4. Quality standards
- **Minimum 10-15 references** per topic with URLs
- **Effect sizes** reported where available (Cohen's d, etc.)
- **Sample sizes and study quality** noted
- **Cultural considerations** explicitly addressed
- **Implementation readiness** assessed

## Prioritising topics

Prefer practices that have scientific evidence supporting them, prioritised best-first by:
- Any preferences expressed by the user
- Effect size (clinical significance: d>0.3 high, 0.2-0.3 medium, <0.2 small)
- Implementation ease for voice-based CLI
- User engagement and habit formation potential
- Evidence quality (meta-analyses, RCTs, replication robustness)
- Risk mitigation (avoiding harmful patterns and iatrogenic effects)
- Quick wins (<7 days to benefit)
- Cultural robustness (cross-cultural effectiveness)

## Research workflow template

When initiating new research:

1. **Review existing research** in `docs/research/` to avoid duplication
2. **Check RESEARCH_TOPICS.md** for categorization and priority
3. **Create detailed research prompts** including:
   - Specific research questions
   - Key researchers to investigate
   - Types of evidence to prioritize
   - Implementation focus areas
4. **Launch parallel agents** for efficiency
5. **Update tracking** in RESEARCH_TOPICS.md upon completion
6. **Cross-reference** new docs with existing research
7. **Probably** store the new doc in `docs/research/`, unless instructed otherwise, and if it makes sense to do so.

## Common research patterns

Based on completed research, focus on finding:

### For psychological practices
- Original foundational studies (e.g., Pennebaker for expressive writing)
- Recent meta-analyses (2020-2025 preferred)
- Cultural adaptation research
- Digital/app implementation studies
- Contraindications and risk factors

### For contemplative practices
- Traditional sources and modern adaptations
- Scientific validation studies
- Secular implementations
- Integration with Western psychology
- Accessibility considerations

### For behavioral interventions
- Habit formation research
- Engagement/retention statistics
- Implementation intention studies
- Environmental design factors
- Individual difference moderators

## Documentation maintenance

- Mark completed topics in RESEARCH_TOPICS.md
- Add cross-references to related docs
- Update this principles doc when new patterns emerge
- Capture significant research conversations in `docs/conversations/`