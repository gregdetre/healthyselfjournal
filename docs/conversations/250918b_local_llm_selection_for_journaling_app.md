---
Date: September 18, 2025
Duration: ~30 minutes
Type: Decision-making
Status: Resolved
Related Docs: docs/reference/OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md
---

# Local LLM Selection for Journaling App - September 18, 2025

## Context & Goals

User needed to select a local LLM to replace Claude for their voice-first reflective journaling app that uses dialogue-based questioning. The model needed to run on a 24GB RAM MacBook M3 without consuming all memory, prioritizing intelligence and instructability over speed.

## Key Background

User requirements:
- "Doesn't matter so much if it's slow, but it just needs to run without using up all the RAM"
- "Be as smart/wise/instructable as possible"
- "I'm fine with going with Ollama"
- "I would prefer to avoid Langchain if possible"
- Must handle complex prompt templates for Socratic questioning, emotional intelligence, and reflective dialogue
- Follow selection criteria from third-party library selection guidelines

## Main Discussion

### Initial Framework Analysis

Started by examining the journaling app's requirements through the question prompt template. The LLM needs sophisticated capabilities:
- Emotional intensity assessment
- Pattern recognition (rumination, contamination narratives)
- Socratic questioning techniques
- Clean language preservation
- Session phase awareness (0-5min, 5-15min, 15-20min, 20min+)
- Change talk detection

### Platform Selection: Ollama

Ollama emerged as the clear winner for the framework due to:
- Massive community with extensive documentation
- Trivial installation process
- Stable Python API without requiring LangChain
- Direct HTTP REST endpoint support
- Strong ecosystem integration with 2024-2025 models

### Model Evolution (2024-2025)

Initial research focused on common models (Llama 3.1, Mistral, DeepSeek), but user requested focus on models from the last 6 months only. This led to discovery of newer 2025 releases:

**Google Gemma 3 (2025)** - Major breakthrough:
- 27B model beats Llama 3 405B and Qwen2.5-70B despite being much smaller
- Quantization Aware Training (QAT) reduces memory from 54GB to 14.1GB with minimal quality loss
- 54% less perplexity drop with Q4 quantization compared to standard approaches

**Other 2025 contenders:**
- Qwen 2.5 series (including QwQ reasoning model)
- Phi-4 14B (Microsoft's latest reasoning model)
- DeepSeek-R1 distilled versions

## Alternatives Considered

### Models Evaluated
1. **Llama 3.3/3.1 8B** - Initially recommended for balance
2. **DeepSeek-R1 14B** - Strong reasoning but higher memory
3. **Qwen 2.5 14B** - Excellent reasoning, lighter weight
4. **Phi-4 14B** - Microsoft's complex reasoning specialist
5. **Gemma 3 27B** - Winner: best performance with Q4 quantization

### Framework Options
- LangChain integration: Rejected due to "unnecessary complexity and overhead"
- Direct Ollama Python library: Considered but not required
- Pure HTTP REST API: Selected for simplicity and control

## Decisions Made

User accepted recommendation for:
- **Framework**: Ollama with direct HTTP REST API (no LangChain)
- **Primary Model**: Gemma 3 27B with Q4 quantization
- **Rationale**: "Gemma 3 27B offers the best balance of capability and efficiency for your journaling app on 24GB hardware"

## Implementation Approach

Simple Python client without dependencies:
```python
class OllamaClient:
    def __init__(self, model="gemma3:27b-instruct-q4_K_M"):
        self.base_url = "http://localhost:11434"
        self.model = model

    def chat(self, messages, temperature=0.7):
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 8192,
                    "repeat_penalty": 1.1
                },
                "stream": False
            }
        )
        return response.json()["message"]["content"]
```

## Open Questions

- Specific performance benchmarks for emotional intelligence tasks
- Optimal quantization level (Q4_K_M vs Q4_0 vs Q5_K_M) for journaling use case
- Fine-tuning potential for therapy/journaling-specific responses

## Next Steps

1. Install Ollama and pull Gemma 3 27B quantized model
2. Test integration with existing Jinja prompt templates
3. Benchmark response quality against Claude baseline
4. Document deployment process across different platforms

## Sources & References

### Key Research Sources
- **Best Local LLM Models 2025** ([Klu AI](https://klu.ai/blog/open-source-llm-models)) - Overview of 2025 open source models
- **Gemma 3 QAT Models** ([Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)) - Quantization breakthrough details
- **Ollama Python Library** ([GitHub](https://github.com/ollama/ollama-python)) - Official Python integration
- **Local AI with Ollama** ([Devtutorial](https://devtutorial.io/local-ai-with-ollama-using-python-to-call-ollama-rest-api-p3817.html)) - Direct REST API examples

### Internal References
- Original prompt template: `healthyselfjournal/prompts/question.prompt.md.jinja`
- Selection criteria: `gjdutils/docs/instructions/THIRD_PARTY_LIBRARY_SELECTION.md`
- Product vision: `docs/reference/PRODUCT_VISION_FEATURES.md`

## Related Work
- Implementation guide: `docs/reference/OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md` (to be created)