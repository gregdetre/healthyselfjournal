# Scientific research & evidence principles

see also: `docs/research/POTENTIAL_RESEARCH_TOPICS.md`

## Prompt instructions for the research agent

```
Great! Now search the web in depth for trustworthy sources using a detailed tasklist and subagents in parallel, and write a doc (with lots of url citations and signposts to other relevant docs) in `docs/research/` for each topic, emphasising information that could inform the product design and/or prompts that we write for the LLM that will be engaging in a dialogue with the user.

Then update @docs/research/POTENTIAL_RESEARCH_TOPICS.md with these instructions.

Add an Appendix section for "Open questions, suggestions, concerns" , and feel free to add anything else that seems worth raising in there.

Then run @gjdutils/docs/instructions/CAPTURE_SOUNDING_BOARD_CONVERSATION.md on this conversation, using a subagent to run @gjdutils/src/ts/cli/sequential-datetime-prefix.ts for `docs/conversations/`.
```

## Prioritising topics

Prefer practices that have scientific evidence supporting them, prioritised best-first by:
  - Any preferences expressed by the user
  - Effect size (clinical significance)
  - Implementation ease for voice-based CLI
  - User engagement and habit formation potential
  - Evidence quality (meta-analyses, RCTs)
  - Risk mitigation (avoiding harmful patterns)
  - Quick wins (<7 days to benefit)
  - Cultural robustness