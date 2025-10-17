# LLM Prompt Templates

## Overview

Jinja2 templates for flexible prompt generation with context and conditionals.

## See also

- `ARCHITECTURE.md` - LLM integration and template engine design
- `OPENING_QUESTIONS.md` - Embedded example questions and variety strategies
- `CONVERSATION_SUMMARIES.md` - Context from previous sessions
- `DIALOGUE_FLOW.md` - When different prompts are used

## Template Structure

Templates include:
- Current session transcript
- Recent session summaries for continuity
- Embedded example questions for inspiration
- Override parameters for custom openers

## Key Variables

- `override_opener` - Custom first question if provided
- `recent_summaries` - Previous session context
- `question_examples` - Bank of varied questions
- `current_session` - Transcript so far

## Anthropic “thinking” note

When using Anthropic models with the `:thinking` suffix in `LLM_MODEL` (e.g., `anthropic:claude-sonnet-4:20250514:thinking`), the app enables the API’s thinking mode and budgets a portion of tokens for reasoning. Budget is clamped to Anthropic’s minimum (>=1024) and below the response `max_tokens`. You can disable thinking by omitting `:thinking` in `LLM_MODEL`.