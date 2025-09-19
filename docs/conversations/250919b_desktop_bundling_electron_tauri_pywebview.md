# Desktop Bundling Options for Python/FastHTML Apps - 250919

---
Date: 2025-09-19
Duration: ~30 minutes
Type: Research Review, Decision-making
Status: Resolved
Related Docs: `docs/reference/PRODUCT_VISION_FEATURES.md`
---

## Context & Goals

Exploring cross-platform desktop bundling options for the voice journaling FastHTML app, with particular focus on handling complex Python dependencies including local AI models (Ollama, llama-cpp-python, Whisper).

## Key Background

From the user's requirements:
- "We have a Python/FastHTML app that we'd like to bundle cross-platform as an app"
- Considering three main options: Electron, Tauri, PyWebView
- Critical constraint: "We may want to use local AI models, e.g. Ollama or Llama-python-cpp, Gemma, Whisper"
- Important clarification: "Let's assume the user will download the LLM model (e.g. Gemma) when they first start the app, i.e. that model won't be part of the bundle"
- Main concern: "I'm most worried about the complexity that Ollama/Llama-cpp-python and Whisper will add"

The app is a voice-first reflective journaling application using:
- Voice recording with Whisper STT
- Claude LLM for dialogue
- FastHTML for web interface
- Local-first data handling

## Main Discussion

### Electron + Python Assessment

**Architecture Approaches Found:**
- Most common: PyInstaller to create Python executable, then bundle with Electron
- Communication via ZeroRPC or REST APIs (Flask/FastAPI/FastHTML)
- Full Python runtime bundling possible (e.g., Datasette Desktop bundles Python 3.9)

**Key Challenges:**
- Complex process management - must carefully handle Python subprocess lifecycle
- Path resolution issues between Electron and Python environments
- Platform-specific packaging challenges
- Adds ~150MB for Chromium + ~50MB for Python runtime

**Real-world Implementation (2024):**
- Successfully used in production (Datasette Desktop)
- Electron-builder can package everything together
- Requires careful cleanup to avoid orphaned Python processes

### Tauri + Python Assessment

**Architecture:**
- Uses native OS webview (WebKit/Edge WebView2) instead of Chromium
- Python runs as "sidecar" process managed by Rust core
- ~10MB base size vs Electron's ~150MB

**Developer Experience (2024):**
- "Caused some pain" - actual developer quote about Tauri+Python
- Requires "most hacky way of copy and pasting" for bundling
- Complex PyInstaller configurations with multiple collect-all flags
- Platform-specific binary suffixes needed

**AI Model Integration:**
- Active examples of bundling llama.cpp and Whisper
- Community building Ollama integrations
- Sidecar pattern well-documented but adds complexity

### PyWebView Assessment

**Architecture:**
- Native Python with embedded webview
- Single process - no IPC overhead
- Uses OS native rendering (GTK/QT on Linux, WebKit on macOS, WebView2 on Windows)

**Developer Experience:**
- "A fantastic tool for building awesome desktop apps around the Python stack" (Feb 2025)
- Developer chose it after "wasting tens of hours" with alternatives
- Simpler than Tauri: avoided having to "learn Rust"
- Direct Python integration without subprocess management

## Alternatives Considered

### Complexity Comparison for AI Dependencies

**llama-cpp-python bundling:**
- **PyWebView**: Direct Python module, single process, PyInstaller handles it
- **Tauri**: Must bundle as separate sidecar, manage IPC, platform-specific builds
- **Electron**: Similar complexity to Tauri, subprocess management required

**Whisper integration:**
- **PyWebView**: faster-whisper bundles FFmpeg via PyAV (no external deps), direct calls
- **Tauri/Electron**: Audio streaming across process boundary adds latency

**Ollama management:**
- All approaches require Ollama as separate service
- PyWebView has advantage of same-process Python for client libraries

## Decisions Made

User's decision criteria: "Which of Tauri/PyWebView do you think will handle those best?"

**Recommendation: PyWebView** for the following reasons:

1. **Lower complexity** - Single Python process, no IPC overhead
2. **Better AI integration** - Direct memory access to models, native Python bindings
3. **Simpler bundling** - PyInstaller handles binary wheels better in single process
4. **Faster development** - Stay in Python ecosystem, avoid Rust learning curve

The user implicitly accepted this recommendation by not pushing back or requesting alternatives.

## Open Questions

- Specific PyInstaller configuration for llama-cpp-python cross-platform builds
- Best practices for downloading and managing LLM models on first run
- Performance comparison of PyWebView vs Tauri for resource-intensive AI workloads
- Long-term maintenance considerations for each approach

## Next Steps

Potential follow-up actions identified:
- Create proof-of-concept with PyWebView + FastHTML + local Whisper
- Test PyInstaller bundling with all AI dependencies
- Benchmark memory usage with various model sizes

## Sources & References

### Official Documentation
- **[Tauri Sidecar Documentation](https://v2.tauri.app/develop/sidecar/)** - Official guide for embedding external binaries
- **[PyWebView 5 Release](https://simonwillison.net/2024/Mar/13/pywebview-5/)** (March 2024) - Latest features and capabilities
- **[llama-cpp-python PyPI](https://pypi.org/project/llama-cpp-python/)** - Python bindings for llama.cpp
- **[faster-whisper PyPI](https://pypi.org/project/faster-whisper/)** - CTranslate2-based Whisper implementation

### Implementation Examples
- **[Datasette Desktop](https://til.simonwillison.net/electron/python-inside-electron)** - Simon Willison's approach to bundling Python in Electron
- **[electron-python-example](https://github.com/fyears/electron-python-example)** - Reference implementation
- **[example-tauri-python-server-sidecar](https://github.com/dieharders/example-tauri-python-server-sidecar)** - Tauri + Python template

### Community Discussions
- **[Building Local LM Desktop Applications with Tauri](https://medium.com/@dillon.desilva/building-local-lm-desktop-applications-with-tauri-f54c628b13d9)** (2024) - Real-world Tauri + LLM experience
- **[How to write and package desktop apps with Tauri + Vue + Python](https://hamza-senhajirhazi.medium.com/how-to-write-and-package-desktop-apps-with-tauri-vue-python-ecc08e1e9f2a)** - Detailed Tauri + Python guide
- **[PyWebView Hacker News Discussion](https://news.ycombinator.com/item?id=39665828)** (2024) - Community feedback on v5

### Research on Bundling Challenges
- **[Whisper & Faster-Whisper standalone executables](https://github.com/Purfview/whisper-standalone-win)** - Pre-built Windows binaries approach
- **[whisper-cpp-python](https://pypi.org/project/whisper-cpp-python/)** - C++ bindings for Whisper
- **[Stack Overflow: llama-cpp-python installation errors](https://stackoverflow.com/questions/77267346/error-while-installing-python-package-llama-cpp-python)** - Common bundling issues

### Web Search Queries Used
- "Electron bundle Python backend 2024 best practices"
- "Electron Python FastHTML 2024 example implementation"
- "PyWebView bundle llama-cpp-python whisper PyInstaller 2024"
- "Tauri Python sidecar llama whisper bundling 2024"
- "llama-cpp-python PyInstaller binary wheels platform specific bundling issues 2024"
- "Ollama integration desktop app PyWebView vs Tauri bundling 2024"
- "PyWebView vs Tauri Python bundling complexity real experience 2024"

## Related Work

- Current implementation: `healthyselfjournal/web/app.py` - FastHTML web interface
- Audio handling: `healthyselfjournal/audio.py`, `healthyselfjournal/transcription.py`
- Future work: Desktop bundling implementation based on PyWebView recommendation

## Key Insights & Patterns

### Bundling Philosophy Trade-offs

**Electron approach**: "Bundle everything" - includes full Chromium, leads to 150MB+ apps but maximum compatibility

**Tauri approach**: "Use native components" - tiny 10MB core but complexity when adding Python/AI

**PyWebView approach**: "Python-first with native UI" - middle ground, good for Python-heavy apps

### AI Model Bundling Complexity Hierarchy

1. **Simplest**: Cloud APIs only (no bundling needed)
2. **Moderate**: Python-only models (transformers, etc.)
3. **Complex**: C++ compiled models (llama-cpp-python, Whisper)
4. **Most Complex**: Separate services (Ollama, local model servers)

### Critical Success Factors for Python + AI Desktop Apps

1. **Binary dependency management** - C++ extensions must be handled correctly
2. **Process lifecycle** - Clean startup/shutdown without orphans
3. **Memory management** - AI models compete with app for RAM
4. **Cross-platform building** - Each OS needs specific handling
5. **Model downloading** - First-run experience for large models

### Developer Pain Points (2024)

Most frustration comes from:
- Subprocess communication complexity (IPC)
- Platform-specific build configurations
- Missing dependencies in bundled apps
- Binary wheel compatibility issues
- Process cleanup and orphaned services

The consensus from real-world experience suggests that for Python-heavy applications with AI components, minimizing process boundaries (PyWebView approach) significantly reduces complexity compared to sidecar architectures (Tauri/Electron).