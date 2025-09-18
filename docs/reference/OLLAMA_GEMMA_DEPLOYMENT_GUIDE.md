# Ollama & Gemma Deployment Guide

*Last updated: September 18, 2025*

## Overview

Ollama is an open-source platform for running Large Language Models (LLMs) locally with minimal setup complexity. Combined with Google's Gemma 3 models (released 2025), it provides a powerful solution for private, offline AI capabilities. This guide covers installation, configuration, and troubleshooting across multiple platforms.

## Why Ollama + Gemma 3

### Ollama Advantages
- **Simplicity**: One-command installation, automatic GPU detection
- **Ecosystem**: Massive community, extensive documentation, active development
- **API Design**: Clean REST API, official Python library, no dependency bloat
- **Performance**: Automatic GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
- **Model Management**: Easy model pulling, quantization support, automatic memory management

### Gemma 3 27B Advantages (2025)
- **Superior Performance**: Outperforms Llama 3 405B and Qwen2.5-70B despite being smaller
- **Memory Efficiency**: Q4 quantization reduces from 54GB to 14.1GB with minimal quality loss
- **QAT Technology**: 54% less perplexity drop with quantization vs standard approaches
- **Practical Size**: Fits comfortably in 24GB VRAM with room to spare

## Platform-Specific Installation

### Apple Silicon (M1/M2/M3/M4) - Priority Platform

#### Installation
```bash
# Option 1: Official installer (recommended)
curl -fsSL https://ollama.com/install.sh | sh

# Option 2: Homebrew
brew install ollama
```

#### Key Features
- **Automatic Metal acceleration**: Up to 40% better performance vs x86 emulation
- **Unified Memory Architecture**: Leverages Apple's shared memory design
- **Neural Engine support**: Dedicated ML hardware acceleration
- **No configuration needed**: GPU acceleration works out of the box

#### Performance Expectations
- Base M1 (8GB): Can run 3B models, ~40 tokens/s
- M3 (24GB): Comfortably runs Gemma 3 27B Q4, ~50-70 tokens/s
- M2/M3 Ultra (64GB+): Can run 110B parameter models with 4-bit quantization

#### Verification
```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep Metal

# Verify GPU usage (look for Metal buffer allocation)
OLLAMA_DEBUG=1 ollama serve
```

### Windows 11

#### Prerequisites
- NVIDIA GPU with compute capability 5.0+ (check [NVIDIA docs](https://developer.nvidia.com/cuda-gpus))
- Latest NVIDIA drivers (version 560+)
- CUDA Toolkit 11.8+ (for GPU acceleration)

#### Installation
```powershell
# Download from https://ollama.com/download/windows
# Or use winget (if available)
winget install Ollama.Ollama
```

#### Common Issues & Solutions

**GPU Not Detected**
```powershell
# Set environment variables
$env:CUDA_VISIBLE_DEVICES="0"
$env:OLLAMA_GPU_MEMORY_FRACTION="0.8"

# For troubleshooting, disable GPU
$env:OLLAMA_NO_GPU="1"
```

**System Crashes with NVIDIA**
- Update to latest NVIDIA drivers (570+)
- Check for driver/CUDA version mismatch
- Add to Ollama service: `Environment="OLLAMA_FLASH_ATTENTION=1"`

**DirectML/AMD Support**
- Limited support as of 2025
- Primarily optimized for CUDA/NVIDIA

### Linux (Ubuntu/Debian)

#### NVIDIA Setup
```bash
# Add NVIDIA repository (Ubuntu)
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install drivers and CUDA
sudo apt install nvidia-driver-560 cuda-toolkit-12-0

# Verify installation
nvidia-smi
nvcc --version

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

#### AMD Setup
```bash
# Install ROCm (for AMD GPUs)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev

# Verify with
rocm-smi
```

#### Network Configuration
```bash
# Edit systemd service for network access
sudo systemctl edit ollama

# Add:
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"

# Restart
sudo systemctl restart ollama
```

#### Troubleshooting Linux GPU Issues
```bash
# Debug GPU detection
OLLAMA_DEBUG=1 ollama serve

# Check loaded models and GPU usage
ollama ps

# For AMD GPUs with recognition issues
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU
```

### macOS Intel

```bash
# Installation same as Apple Silicon
curl -fsSL https://ollama.com/install.sh | sh

# Note: CPU-only, no GPU acceleration
# Performance will be significantly slower
```

## Gemma 3 Model Setup

### Pulling Models
```bash
# Recommended: 27B with Q4 quantization (14.1GB)
ollama pull gemma3:27b-instruct-q4_K_M

# Alternative quantizations
ollama pull gemma3:27b-instruct-fp16    # Highest quality (54GB)
ollama pull gemma3:27b-instruct-q8_0    # Balanced (27GB)
ollama pull gemma3:27b-instruct-q2_K    # Extreme compression (7GB)

# Smaller models for limited hardware
ollama pull gemma3:12b-instruct-q4_K_M  # 6.6GB
ollama pull gemma3:4b-instruct-q4_K_M   # 2.6GB
```

## Known Issues & Gotchas (2025)

### Gemma 3 Specific Issues

#### 1. High VRAM Usage
- **Problem**: Unusually high VRAM consumption vs similar models
- **Solution**:
  ```bash
  export OLLAMA_FLASH_ATTENTION=1
  export OLLAMA_KV_CACHE_TYPE=q8_0
  ```

#### 2. Random Crashes
- **Problem**: Gemma3:27b crashes randomly while other models work
- **Solution**: Consider using Ollama v0.6.8 instead of v0.7
- **Alternative**: Use Gemma 2 27B or Qwen2.5 32B

#### 3. Incorrect Model Size Detection (v0.7)
- **Problem**: Reports 26GB instead of 24GB, fewer GPU layers
- **Solution**: Manually set GPU layers:
  ```bash
  export OLLAMA_GPU_LAYERS=62
  ```

#### 4. Poor Long Context Performance
- **Problem**: Model loses track with contexts >48K tokens
- **Solution**: Limit context to 32K or less:
  ```python
  options={"num_ctx": 32768}
  ```

#### 5. Segmentation Violations
- **Problem**: SIGSEGV after multiple API calls
- **Solution**: Implement retry logic, consider memory limits:
  ```bash
  export OLLAMA_MAX_MEMORY=32g
  export OLLAMA_GPU_MEMORY=16g
  ```

### Platform-Specific Gotchas

#### Apple Silicon
- **Docker limitation**: No GPU acceleration in Docker containers
- **Solution**: Run Ollama natively, not in containers

#### Windows
- **WSL complications**: GPU passthrough issues
- **CUDA version mismatches**: nvidia-smi version lag
- **Solution**: Run natively on Windows, keep drivers updated

#### Linux
- **AMD GPU recognition**: Often not detected despite ROCm
- **Multi-GPU confusion**: Wrong GPU selected
- **Solution**: Explicitly set GPU device:
  ```bash
  export CUDA_VISIBLE_DEVICES=0  # NVIDIA
  export HIP_VISIBLE_DEVICES=0   # AMD
  ```

## Python Integration (Without LangChain)

### Direct REST API
```python
import requests
import json

class OllamaClient:
    def __init__(self, model="gemma3:27b-instruct-q4_K_M", base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = model

    def chat(self, messages, temperature=0.7, num_ctx=8192):
        """Direct chat API call"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "repeat_penalty": 1.1,
                    "top_p": 0.95
                },
                "stream": False
            }
        )
        return response.json()["message"]["content"]

    def generate(self, prompt, system=None):
        """Generate completion with optional system prompt"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        if system:
            data["system"] = system

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data,
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_response += chunk["response"]

        return full_response
```

### Official Python Library
```python
from ollama import Client

client = Client(host='http://localhost:11434')

# Chat completion
response = client.chat(
    model='gemma3:27b-instruct-q4_K_M',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': 'Hello!'}
    ]
)

print(response['message']['content'])
```

### Migration from Claude
```python
# Before (Claude via Anthropic SDK)
from anthropic import Anthropic
client = Anthropic(api_key="...")
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Hello"}]
)

# After (Ollama)
from ollama import Client
client = Client()
response = client.chat(
    model='gemma3:27b-instruct-q4_K_M',
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Performance Optimization

### Memory Management
```bash
# Set memory limits
export OLLAMA_MAX_MEMORY=24g      # System RAM limit
export OLLAMA_GPU_MEMORY=20g      # VRAM limit
export OLLAMA_GPU_MEMORY_FRACTION=0.8  # Use 80% of VRAM

# Enable new memory estimator (2025)
export OLLAMA_NEW_ESTIMATES=1
```

### Context Window Tuning
```python
# For journaling/dialogue (prioritize quality)
options = {
    "num_ctx": 8192,        # Reasonable context
    "num_batch": 512,       # Larger batches
    "num_gpu": 62,         # Max GPU layers for 27B
    "main_gpu": 0,         # Primary GPU
    "low_vram": False      # Full GPU usage
}

# For quick responses (prioritize speed)
options = {
    "num_ctx": 2048,       # Smaller context
    "num_batch": 128,      # Smaller batches
    "temperature": 0.5,    # Less randomness
    "top_k": 10           # Limit token choices
}
```

### Model-Specific Settings for Journaling
```python
# Optimized for reflective dialogue
journaling_options = {
    "temperature": 0.7,      # Balanced creativity
    "top_p": 0.95,          # Nucleus sampling
    "repeat_penalty": 1.1,   # Reduce repetition
    "repeat_last_n": 64,    # Context for penalty
    "mirostat": 2,          # Coherent long responses
    "mirostat_tau": 5.0,    # Target perplexity
    "num_ctx": 8192,        # Full conversation context
    "seed": -1              # Random seed
}
```

## Troubleshooting Guide

### Diagnostic Commands
```bash
# Check Ollama version
ollama --version

# List installed models
ollama list

# Check running models and resource usage
ollama ps

# Test model
ollama run gemma3:27b-instruct-q4_K_M "Hello, how are you?"

# Verbose logging
OLLAMA_DEBUG=1 ollama serve

# Check API health
curl http://localhost:11434/api/tags
```

### Common Problems & Solutions

| Problem | Solution |
|---------|----------|
| Model runs slowly | Check GPU detection with `ollama ps`, verify quantization level |
| Out of memory | Use more aggressive quantization (Q2_K), reduce context window |
| GPU not detected | Update drivers, check CUDA/ROCm installation, verify with nvidia-smi/rocm-smi |
| API connection refused | Check if Ollama service is running: `systemctl status ollama` |
| Inconsistent responses | Adjust temperature and repeat_penalty parameters |
| Crashes during generation | Reduce batch size, check memory limits, try different quantization |

## Migration Path

### From Cloud APIs
1. Install Ollama and pull Gemma model
2. Update API endpoints from cloud URLs to `localhost:11434`
3. Adjust authentication (remove API keys)
4. Update response parsing (similar JSON structure)
5. Add retry logic for local resource constraints
6. Test performance and adjust quantization as needed

### Package Version Compatibility
```python
# Recommended versions (Sep 2025)
ollama==0.5.4           # Latest stable
requests>=2.31.0        # For direct API
jinja2>=3.1.0          # For templates

# Avoid
langchain<0.3.1         # Deprecated Ollama integration
ollama<0.4.0           # Breaking changes with newer models
```

## Best Practices

### Production Deployment
1. **Model Selection**: Start with Q4 quantization, adjust based on quality needs
2. **Resource Monitoring**: Track VRAM usage, implement graceful degradation
3. **Fallback Strategy**: Keep smaller model (12B/4B) as backup
4. **Health Checks**: Regular API pings, automatic restart on failure
5. **Logging**: Enable debug logs for troubleshooting, rotate regularly

### Development Workflow
1. **Local Testing**: Use smaller models (4B) for rapid iteration
2. **Quality Validation**: Test with target model before deployment
3. **Version Control**: Pin model versions in configuration
4. **Documentation**: Document model choice rationale and settings

## Resources

### Official Documentation
- **Ollama**: https://ollama.com/library
- **Gemma Models**: https://ollama.com/library/gemma3
- **API Reference**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Python Library**: https://github.com/ollama/ollama-python

### Community Resources
- **GitHub Issues**: https://github.com/ollama/ollama/issues
- **Model Benchmarks**: https://artificialanalysis.ai/models
- **Performance Guides**: https://markaicode.com/google-gemma-3-27b-ollama-installation-guide/

### Related Internal Documentation
- Selection process: `docs/conversations/250918b_local_llm_selection_for_journaling_app.md`
- Product requirements: `docs/reference/PRODUCT_VISION_FEATURES.md`
- Prompt templates: `examinedlifejournal/prompts/`

---

*This guide focuses on practical deployment with emphasis on Apple Silicon as the primary platform, while providing comprehensive coverage for Windows and Linux environments. Regular updates recommended as Ollama and Gemma models evolve rapidly.*