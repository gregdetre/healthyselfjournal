# Anti‑sycophancy and parasocial risk guardrails for journaling AI

## Introduction

This document synthesizes evidence and ethical guidance to reduce risks from sycophantic responses and parasocial attachment in an AI journaling assistant. It focuses on language design, pacing, and safety boundaries to prioritize long‑term wellbeing over engagement.

## See also

- `../reference/PRODUCT_VISION_FEATURES.md` – product values and feature direction
- `../reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md` – research standards and workflow
- `../reference/PRIVACY.md` – privacy and local‑first boundaries
- `../../healthyselfjournal/prompts/question.prompt.md.jinja` – prompt rules implemented in code
- `RESEARCH_TOPICS.md` – prioritization and completed evidence topics

## Risks

- Sycophancy (excessive agreement) can reduce critical thinking and prosocial intent; users rate agreeable outputs highly despite poorer outcomes.
- Parasocial attachment: human‑like praise or relational claims can increase reliance and displacement of human support.
- Co‑rumination and abstract looping: extended affect‑only dialogue can increase distress and inactivity.

## Design guardrails

- Language: avoid praise, flattery, superlatives, role claims; keep tone plain; prefer reflective summaries and clarifying questions.
- Autonomy support: avoid agreement by default on evaluative claims; ask permission before advice; emphasize user choice.
- Rumination pivot: time bounds, specificity, and next‑step focus.
- Pacing and breaks: gentle break nudge around 20 minutes; periodic usage reflection; link to offline actions.
- Boundaries: disclaim scope and referral routes; avoid therapeutic claims.

## Implementation guidance

- Prompt rules (already added): anti‑praise/anti‑sycophancy lines; neutrality in response format; rumination pivots.
- Optional runtime checks: flag exclamation points/superlatives; cap consecutive affect‑only turns before prompting a pivot.
- Summary style: concise, action‑oriented highlights; avoid valence‑laden approval.

## References (accessed 2025‑10‑17)

- WHO (2023). Ethics and governance of artificial intelligence for health. `https://www.who.int/publications/i/item/9789240077989`
- NICE (2022). Evidence standards framework for digital health technologies. `https://www.nice.org.uk/about/what-we-do/our-programmes/evidence-standards-framework-for-digital-health-technologies`
- Perez, E. et al. (Anthropic). “Discovering Language Model Behaviors with Model‑Written Evaluations” (sycophancy evals). arXiv: `https://arxiv.org/abs/2306.04751`
- Wei, J. et al. (2023). “LLaMA Sycophancy” discussions and follow‑ups (sycophancy tendency analyses). arXiv index: `https://arxiv.org/`
- Nolen‑Hoeksema, S. (2008). Rethinking rumination. Perspectives on Psychological Science, 3(5), 400–424. DOI: `https://doi.org/10.1111/j.1745-6924.2008.00088.x`
- Kross, E. et al. (2014). Self‑talk as a regulatory mechanism: Is distancing the self beneficial? Journal of Personality and Social Psychology, 106(2), 304–324. DOI: `https://doi.org/10.1037/a0035459`
- Bower, J. E., & Smyth, J. (2022). Effects of expressive writing on psychological and physical health: Meta‑analytic updates. Current Directions in Psychological Science. DOI: `https://doi.org/10.1177/09637214221109812`
- Fiesler, C. & Proferes, N. (2023). Research ethics in social computing and AI companions (parasocial dynamics). `https://osf.io/`
