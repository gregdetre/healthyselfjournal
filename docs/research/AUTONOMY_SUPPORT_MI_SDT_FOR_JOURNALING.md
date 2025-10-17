# Autonomy support for journaling: Motivational Interviewing (MI) and Self‑Determination Theory (SDT)

## Introduction

This document summarizes how MI and SDT can guide the app’s questioning style to support user autonomy, reduce dependency, and improve wellbeing. It translates core principles into prompt rules for a voice‑first CLI, with emphasis on minimal effusiveness, anti‑sycophancy, and rumination safeguards.

## See also

- `../reference/PRODUCT_VISION_FEATURES.md` – product values and feature direction
- `../reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md` – research standards and workflow
- `../reference/DIALOGUE_FLOW.md` – question sequencing and session management
- `../../healthyselfjournal/prompts/question.prompt.md.jinja` – prompt rules implemented in code
- `RESEARCH_TOPICS.md` – prioritization and completed evidence topics

## Principles and design decisions (MI + SDT)

- Autonomy‑supportive stance (SDT): acknowledge perspective, offer choices, invite self‑initiation; avoid controlling language.
- Evocation over persuasion (MI): draw out the user’s own reasons and ideas; avoid the “righting reflex” and cheerleading.
- Brief reflective summaries, not praise: reflect content neutrally; avoid “I’m proud of you,” superlatives, or emojis.
- Ask‑permission before advice: “Would you like a brief suggestion?”; keep suggestions optional and specific.
- Action focus without pressure: pivot from abstract loops to one small, user‑chosen next step when rumination is detected.

## Implementation guidance for the voice‑first CLI

- Acknowledgement pattern: one neutral line reflecting what was said, then a single, concrete follow‑up question.
- Language patterns to prefer: “if you want…”, “you might consider…”, “would you like…?”, “it’s up to you…”.
- Language to avoid: praise/approval, exclamation points, superlatives, role claims (“as your therapist”), flattery.
- Rumination pivot: add time‑bound or specificity constraints (e.g., next 24 hours; one small action; one concrete example).
- Consent check: before suggestions, ask permission; if declined, return to exploratory questioning.

## Mapping to prompt rules (already implemented or planned)

- Anti‑praise/anti‑sycophancy lines in `question.prompt.md.jinja` enforce neutrality and avoid agreement by default on evaluative claims.
- Autonomy‑supportive phrasing is explicitly encouraged; advice is permission‑based.
- Rumination detection → concrete pivot and short‑horizon framing.

## Risks and safeguards

- Over‑directive tone can reduce intrinsic motivation (SDT): keep choices salient and pressure low.
- Excessive validation can increase dependency (MI): prefer neutral reflection to approval.
- Long abstract processing can drift into co‑rumination: use specificity/time bounds and action anchors.

## References (accessed 2025‑10‑17)

- Miller, W. R., & Rollnick, S. (2013). Motivational Interviewing: Helping People Change (3rd ed.). Oxford University Press. `https://global.oup.com/academic/product/motivational-interviewing-9781609182274`
- Lundahl, B. et al. (2010). A meta‑analysis of motivational interviewing: Twenty‑five years of empirical studies. Journal of Consulting and Clinical Psychology, 78(6), 868–884. PubMed: `https://pubmed.ncbi.nlm.nih.gov/21114301/`
- Lundahl, B. et al. (2013). Motivational interviewing in medical care settings: A systematic review and meta‑analysis. Patient Education and Counseling, 93(2), 157–168. PubMed: `https://pubmed.ncbi.nlm.nih.gov/24001658/`
- Deci, E. L., & Ryan, R. M. (2000). The “what” and “why” of goal pursuits: Human needs and the self‑determination of behavior. American Psychologist, 55(1), 68–78. DOI: `https://doi.org/10.1037/0003-066X.55.1.68`
- Ng, J. Y. Y. et al. (2012). Self‑Determination Theory applied to health contexts: A meta‑analysis. Perspectives on Psychological Science, 7(4), 325–340. PubMed: `https://pubmed.ncbi.nlm.nih.gov/26168370/`
- Su, Y.‑L., & Reeve, J. (2011). A meta‑analysis of the effectiveness of intervention programs designed to support autonomy. Educational Psychology Review, 23, 159–188. DOI: `https://doi.org/10.1007/s10648-010-9142-7`
- Teixeira, P. J. et al. (2020). A conceptual model for fostering the adoption of health behaviors via autonomy support. Health Psychology Review, 14(3), 235–252. DOI: `https://doi.org/10.1080/17437199.2019.1689411`
- Ryan, R. M., Patrick, H., Deci, E. L., & Williams, G. C. (2008). Facilitating health behaviour change and its maintenance: Interventions based on SDT. The European Health Psychologist, 10(1), 2–5. PDF: `https://selfdeterminationtheory.org/SDT/documents/2008_RyanPatrickDeciWilliams.pdf`
