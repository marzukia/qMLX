# vLLM-MLX

**Apple Silicon MLX Backend for vLLM alike** - GPU-accelerated LLM inference on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow.svg)](https://github.com/waybarrios/vllm-mlx)
[![GitHub](https://img.shields.io/badge/GitHub-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)

> **🚧 Work in Progress**: This project is under active development. Core functionality is complete, but optimizations are ongoing.
>
> **Repository**: [https://github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## Overview

vllm-mlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Vision-language models for multimodal inference

## Features

- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **Unified memory** - no CPU↔GPU data transfers
- **4-bit quantization** - run large models on limited memory
- **Vision-language models** - image, video, and audio understanding
- **vLLM API compatible** - same OpenAI-compatible interface
- **Optimized by default** - mlx-lm includes Flash Attention and optimized Metal kernels
- **Paged KV Cache** - memory-efficient caching with prefix sharing for concurrent users

## Project Status

### ✅ What's Complete (Phases 1-3)

**Phase 1: Core LLM Support**
- ✅ MLXPlatform integration with vLLM
- ✅ Basic LLM inference using mlx-lm
- ✅ Model loading from HuggingFace
- ✅ Text generation with streaming support
- ✅ Chat completion interface

**Phase 2: OpenAI-Compatible Server**
- ✅ FastAPI server with OpenAI-compatible endpoints
- ✅ `/v1/chat/completions` endpoint
- ✅ `/v1/completions` endpoint
- ✅ Streaming responses (SSE)
- ✅ Full OpenAI Python SDK compatibility

**Phase 3: Multimodal Support (MLLM)**
- ✅ mlx-vlm integration for vision-language models
- ✅ Image understanding (URLs, base64, local files)
- ✅ Video understanding (URLs, base64, local files)
- ✅ Multi-image support
- ✅ OpenAI-compatible multimodal API
- ✅ Support for Qwen-VL, LLaVA, Idefics, PaliGemma, Pixtral, Molmo, DeepSeek-VL
- ✅ Gradio chat UI with text/image/video support
- ✅ Performance benchmarking tools

**Phase 4: Optimizations (Complete)**
- ✅ Continuous batching for higher throughput (Phase 4.1)
- ✅ KV cache / prefix caching for repeated prompts (Phase 4.2)
- ✅ Low-latency streaming with RequestOutputCollector pattern (Phase 4.3)
- ✅ Dual server modes: Simple (max throughput) and Continuous Batching (multi-user)
- ✅ Qwen3 tokenizer fix (eos_token handling)
- ✅ Special token filtering in output
- ✅ Paged KV Cache for memory-efficient prefix sharing (Phase 4.4)

**Phase 5: Integrations**
- ✅ Open WebUI compatibility
- ✅ Configurable max_tokens via CLI
- ⏳ LangChain integration
- ⏳ LlamaIndex integration

**Advanced Features**
- ⏳ Structured output (JSON mode, grammar constraints)
- ✅ Function calling / tool use with MCP integration
- ⏳ Vision-language reasoning chains
- ⏳ Fine-tuning support

**Current Limitations:**
- Limited to models available on mlx-community

**Want to contribute?** See [Contributing](#contributing) section below.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX and dependencies

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

# Install the package (this installs all dependencies automatically)
pip install -e .
```

Optional (transformers video processors / torchvision):
```bash
pip install -e ".[vision]"
```

This will install:
- `mlx`, `mlx-lm`, `mlx-vlm` - MLX framework and model libraries
- `transformers`, `tokenizers` - HuggingFace libraries
- `opencv-python` - Video processing
- `gradio` - Chat UI
- `psutil` - Resource monitoring

### Verify Installation

```bash
# Check that CLI commands are available
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help

# Test with a small model
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 1
```

## Quick Start

### Option 1: OpenAI-Compatible Server

Start the server:
```bash
# Start server (simple mode - maximum throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# For multiple concurrent users, use continuous batching mode
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

Use with OpenAI client:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or use curl:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### OpenAI Python SDK - Complete Examples

vllm-mlx is fully compatible with the OpenAI Python SDK for text, images, and video.

#### Text Chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Simple text chat
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short story"}],
    max_tokens=200,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Image Analysis (with VLM model)

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Option 1: Image from URL
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}}
        ]
    }],
    max_tokens=256
)
print(response.choices[0].message.content)

# Option 2: Base64 encoded image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image("photo.jpg")
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }],
    max_tokens=512
)
print(response.choices[0].message.content)
```

#### Video Analysis (with VLM model)

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Option 1: Video from URL
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }],
    max_tokens=512
)
print(response.choices[0].message.content)

# Option 2: Base64 encoded video
def encode_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_video = encode_video("video.mp4")
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what's happening in this video"},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}}
        ]
    }],
    max_tokens=512
)
print(response.choices[0].message.content)
```

### MLLM Server (Multimodal Language Models)

Start the server with a MLLM model (auto-detected):
```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

Use with OpenAI client for multimodal content:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Qwen3-VL-4B-Instruct-3bit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    max_tokens=256
)
print(response.choices[0].message.content)
```

Or use curl with multimodal content:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-VL-4B-Instruct-3bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 256
  }'
```

### Option 2: Direct Python API

```python
from vllm_mlx.models import MLXLanguageModel

# Load a quantized model
model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
model.load()

# Generate text
output = model.generate("What is the capital of France?", max_tokens=100)
print(output.text)

# Streaming generation
for chunk in model.stream_generate("Tell me a story about a robot"):
    print(chunk.text, end="", flush=True)
```

### Chat Interface

```python
messages = [
    {"role": "user", "content": "Hello, who are you?"}
]
response = model.chat(messages)
print(response.text)
```

### Multimodal Language Models

```python
from vllm_mlx.models import MLXVisionLanguageModel

# Load a vision model
mllm = MLXVisionLanguageModel("mlx-community/Qwen2-VL-2B-Instruct-4bit")
mllm.load()

# Describe an image
description = mllm.describe_image("photo.jpg")
print(description)

# Answer questions about images
answer = mllm.answer_about_image("photo.jpg", "What color is the car?")
print(answer)

# Multi-image understanding
output = mllm.generate(
    prompt="Compare these two images",
    images=["image1.jpg", "image2.jpg"]
)
```

### Video Understanding

```python
# From local file
output = mllm.generate(
    prompt="What is happening in this video?",
    videos=["video.mp4"],
    video_fps=1.0,  # Extract 1 frame per second
    video_max_frames=16
)
print(output.text)

# From URL (auto-downloads)
output = mllm.generate(
    prompt="Describe this video",
    videos=["https://example.com/video.mp4"],
    video_fps=2.0
)

# Convenience method
description = mllm.describe_video("video.mp4", fps=2.0)
```

### Video via OpenAI-Compatible API

Send video content using the familiar OpenAI format:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Using video_url (similar to image_url)
response = client.chat.completions.create(
    model="mlx-community/Qwen3-VL-4B-Instruct-3bit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }],
    max_tokens=256
)
print(response.choices[0].message.content)
```

Or with curl:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-VL-4B-Instruct-3bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this video"},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
      ]
    }],
    "max_tokens": 256,
    "video_fps": 2.0,
    "video_max_frames": 16
  }'
```

**Supported video formats:**
- Local files: `{"type": "video", "video": "/path/to/video.mp4"}`
- URLs: `{"type": "video_url", "video_url": {"url": "https://..."}}`
- Base64: `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`

## Example Scripts

The `examples/` directory contains ready-to-run scripts demonstrating different use cases:

### Direct Python API Examples

#### `simple_generate.py` - Basic LLM Inference

Demonstrates simple text generation, streaming, and chat with the direct Python API.

```bash
python examples/simple_generate.py
```

What it shows:
- Loading a quantized model
- Simple text generation
- Streaming generation
- Chat interface

#### `mllm_example.py` - Multimodal Language Models

Shows image understanding and visual question answering.

```bash
# With an image file
python examples/mllm_example.py path/to/image.jpg

# Without image (text-only mode)
python examples/mllm_example.py
```

What it shows:
- Loading a multimodal model
- Image description
- Visual question answering
- Custom prompts with images

### OpenAI API Examples

These examples require a running server. Start one first:

```bash
# For text examples
vllm-mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# For image/video examples
vllm-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

#### `demo_openai_text.py` - Text Chat

Complete examples using the OpenAI Python SDK for text chat.

```bash
python examples/demo_openai_text.py
```

What it shows:
- Simple chat completion
- System messages and roles
- Streaming responses
- Multi-turn conversations
- Temperature control

#### `demo_openai_image.py` - Image Analysis

Image understanding using the OpenAI API format.

```bash
python examples/demo_openai_image.py
```

What it shows:
- Images from URLs
- Base64 encoded images
- Visual question answering
- Follow-up questions

#### `demo_openai_video.py` - Video Analysis

Video understanding using the OpenAI API format.

```bash
python examples/demo_openai_video.py
```

What it shows:
- Videos from URLs
- Base64 encoded videos
- Video description and analysis
- Specific questions about video content
- Follow-up questions

### Benchmark Examples

Run performance benchmarks to measure inference speed:

#### Text-Only LLM Benchmarks

```bash
# Run LLM benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 5 --max-tokens 256
```

**Real Performance - LLM Models (M4 Max, 128GB):**

| Model | Gen Speed | TTFT* | Memory |
|-------|-----------|-------|--------|
| Qwen3-0.6B-8bit | 402.3 tok/s | 58.6 ms | 0.68 GB |
| Llama-3.2-1B-Instruct-4bit | 463.6 tok/s | 49.2 ms | 0.69 GB |
| Qwen2.5-1.5B-Instruct-4bit | 308.5 tok/s | 86.2 ms | 0.84 GB |
| Llama-3.2-3B-Instruct-4bit | 200.1 tok/s | 81.4 ms | 1.79 GB |
| Qwen3-30B-A3B-4bit | 123.9 tok/s | 126.9 ms | 16.05 GB |

*TTFT = Time to First Token (latency until the model starts generating)

#### Multimodal Image Benchmarks

```bash
# Full image benchmark (10 resolutions)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Quick image benchmark (4 resolutions)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --quick
```

**Real Performance - Qwen3-VL-8B-Instruct-4bit (M4 Max, 128GB):**

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.04s | 78 | 74.8 tok/s |
| 336x336 | 113K | 0.94s | 64 | 68.3 tok/s |
| 448x448 | 201K | 1.45s | 70 | 48.1 tok/s |
| 512x512 | 262K | 1.58s | 99 | 62.8 tok/s |
| 672x672 | 452K | 1.83s | 83 | 45.3 tok/s |
| 768x768 | 590K | 2.05s | 91 | 44.3 tok/s |
| 896x896 | 803K | 2.61s | 90 | 34.5 tok/s |
| 1024x1024 | 1.0M | 2.79s | 76 | 27.2 tok/s |
| 1280x720 | 922K | 2.97s | 96 | 32.4 tok/s |
| 1920x1080 | 2.1M | 6.30s | 89 | 14.1 tok/s |

**Summary:** Average 45.2 tok/s across all resolutions. Fastest at 224x224 (74.8 tok/s), slowest at 1920x1080 (14.1 tok/s)

#### Multimodal Video Benchmarks

```bash
# Full video benchmark (8 configurations)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video

# Quick video benchmark (3 frame counts)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video --quick
```

**Real Performance - Qwen3-VL-8B-Instruct-4bit (M4 Max, 128GB):**

| Configuration | Frames | Time | Tokens | Speed | Memory |
|---------------|--------|------|--------|-------|--------|
| 2 frames @ 0.5fps | 2 | 4.46s | 256 | 57.4 tok/s | 6.4 GB |
| 4 frames @ 1fps | 4 | 4.69s | 256 | 54.5 tok/s | 6.4 GB |
| 6 frames @ 1fps | 6 | 5.16s | 197 | 38.2 tok/s | 6.6 GB |
| 8 frames @ 2fps | 8 | 6.72s | 240 | 35.7 tok/s | 6.8 GB |
| 12 frames @ 2fps | 12 | 8.91s | 256 | 28.7 tok/s | 7.1 GB |
| 16 frames @ 2fps | 16 | 11.46s | 256 | 22.3 tok/s | 7.6 GB |
| 24 frames @ 4fps | 24 | 16.13s | 226 | 14.0 tok/s | 8.4 GB |
| 32 frames @ 4fps | 32 | 21.34s | 256 | 12.0 tok/s | 9.2 GB |

**Summary:** Average 24.6 tok/s across all configurations. Fastest at 2 frames (57.4 tok/s), slowest at 32 frames (12.0 tok/s)

#### Continuous Batching & Prefix Cache

vllm-mlx includes optimizations for handling multiple concurrent requests efficiently.

**Run the tests:**
```bash
# Continuous batching benchmark
python tests/test_continuous_batching.py

# Prefix cache test
python tests/test_prefix_cache.py
```

**Continuous Batching Results (M4 Max, 128GB):**

| Model | Single Request | Batch (5 req) | Speedup |
|-------|----------------|---------------|---------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*Batching 5 concurrent requests shows 1.5-3x throughput improvement depending on model size.*

**Streaming Performance (M4 Max, 128GB):**

| Model | TTFT | Generation Speed |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

*TTFT = Time to First Token. Streaming delivers tokens as they're generated with low latency.*

**Streaming Configuration:**

Use `--stream-interval` to control streaming behavior:

```bash
# Smooth streaming (default) - send every token immediately
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --stream-interval 1

# Batch tokens for higher throughput (useful for high-latency networks)
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --stream-interval 5

# Non-streaming mode: set stream=false in API request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'

# Streaming mode (default): set stream=true in API request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

| `--stream-interval` | Behavior |
|---------------------|----------|
| `1` (default) | Send every token immediately - smoothest streaming |
| `2-5` | Batch tokens before sending - reduces network overhead |
| `10+` | Higher batching - maximum throughput, chunkier output |

**Prefix Cache Results - Qwen3-0.6B-8bit (M4 Max, 128GB):**

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================

  Model: mlx-community/Qwen3-0.6B-8bit
  Test: Verify KV cache reuse for repeated prompts
  Expected behavior:
    - Same prompt → cache HIT (skip prompt processing)
    - Different prompt → cache MISS (process from scratch)

----------------------------------------------------------------------
  Loading Model
----------------------------------------------------------------------
    Model loaded in 0.70s

----------------------------------------------------------------------
  TEST 1: First Request (Cache Miss Expected)
----------------------------------------------------------------------
    Prompt: "What is 2+2?"
    Tokens: 15

    Cache Statistics:
    Metric        | Value
    --------------+------
    Hits          | 0
    Misses        | 1
    Hit Rate      | 0.0%
    Tokens Saved  | 0
    Total Queries | 1

----------------------------------------------------------------------
  TEST 2: Same Prompt Again (Cache Hit Expected)
----------------------------------------------------------------------
    Prompt: "What is 2+2?" (same as TEST 1)
    Tokens: 15
    Speedup: 1.26x faster

    Cache Statistics:
    Metric        | Value
    --------------+------
    Hits          | 1
    Misses        | 1
    Hit Rate      | 50.0%
    Tokens Saved  | 15
    Total Queries | 2

----------------------------------------------------------------------
  TEST 3: Different Prompt (Cache Miss Expected)
----------------------------------------------------------------------
    Prompt: "What is the capital of France?" (different from TEST 1)
    Tokens: 15

    Cache Statistics:
    Metric        | Value
    --------------+------
    Hits          | 1
    Misses        | 2
    Hit Rate      | 33.3%
    Tokens Saved  | 15
    Total Queries | 3

======================================================================
  TEST RESULTS SUMMARY
======================================================================

    Test Results:
    Test   | Description          | Expected | Actual | Time   | Status
    -------+----------------------+----------+--------+--------+-------
    TEST 1 | First request        | MISS     | MISS   | 84.3ms | ✓
    TEST 2 | Same prompt (cached) | HIT      | HIT    | 66.9ms | ✓
    TEST 3 | Different prompt     | MISS     | MISS   | 65.2ms | ✓

    Final Cache Statistics:
    Metric           | Value
    -----------------+------
    Total Requests   | 3
    Cache Hits       | 1
    Cache Misses     | 2
    Hit Rate         | 33.3%
    Tokens Saved     | 15
    Speedup (cached) | 1.26x

======================================================================
  ✓ ALL TESTS PASSED - Prefix cache working correctly!
======================================================================
```

*Prefix caching saves computation when the same prompt prefix is repeated (e.g., system prompts, chat history).*

#### Paged KV Cache (Memory Efficiency)

Paged KV Cache provides memory-efficient caching with block-based allocation and prefix sharing for concurrent users. This implementation follows **vLLM's architecture** (`vllm/v1/core/block_pool.py`) adapted for MLX on Apple Silicon.

**Key Advantages:**

| Feature | Benefit |
|---------|---------|
| **1.14x Speedup** | Faster inference by reusing cached KV computations |
| **80% Memory Savings** | Share system prompt blocks across concurrent users |
| **vLLM Architecture** | FreeKVCacheBlockQueue, BlockHashToBlockMap, chain hashing |
| **Real Tensor Storage** | Extracts actual KV data using `.state`, survives BatchGenerator recreation |
| **Block Deduplication** | Hash-based detection prevents duplicate storage |
| **Copy-on-Write (COW)** | Shared blocks only copied when modified |
| **O(1) LRU Eviction** | Doubly linked list for efficient cleanup under memory pressure |

**Architecture (vLLM-style):**

```
┌─────────────────────────────────────────────────────────────────┐
│                      PagedCacheManager                          │
├─────────────────────────────────────────────────────────────────┤
│  FreeKVCacheBlockQueue     │  BlockHashToBlockMap               │
│  (O(1) doubly linked list) │  (hash → block for prefix caching) │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐  │  {hash_0: block_5}                 │
│  │ 3 │↔│ 7 │↔│ 2 │↔│ 9 │  │  {hash_1: block_12}                │
│  └───┘ └───┘ └───┘ └───┘  │  {hash_2: block_5}  (shared!)      │
│   LRU ───────────▶ MRU    │                                     │
├─────────────────────────────────────────────────────────────────┤
│  CacheBlock[0..N]:                                              │
│  - block_id, ref_count, block_hash                              │
│  - prev_free_block, next_free_block (doubly linked)             │
│  - cache_data: List[(keys, values)] per layer                   │
└─────────────────────────────────────────────────────────────────┘
```

**How It Works:**

The Paged KV Cache extracts actual tensor data from mlx-lm's KVCache using the `.state` property, storing real KV tensor slices in 64-token blocks with chain hashing (each block's hash depends on its parent).

```
Request Completion                    Cache Storage
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌─────────────────────┐
│ response.cache() │ ───────────▶ │ Extract .state      │
│ (KVCache objects)│              │ (keys, values)      │
└──────────────────┘              └─────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ Slice into 64-token │
                                  │ blocks + chain hash │
                                  └─────────────────────┘
                                            │
       New Request                          ▼
       │                          ┌─────────────────────┐
       ▼                          │ BlockHashToBlockMap │
┌──────────────────┐              │ deduplicate & share │
│ compute_block_   │ ◀─────────── └─────────────────────┘
│ hash(parent, tok)│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Reconstruct via  │
│ mx.concatenate() │
│ + KVCache.from_  │
│ state()          │
└──────────────────┘
```

**Quick Start - Run Tests:**

```bash
# Run unit tests (37 tests)
python -m pytest tests/test_paged_cache.py -v

# Run real inference test (20 requests, 2 rounds)
python tests/test_paged_cache_real_inference.py
```

**Change Model for Tests:**

```bash
# Edit the model in test file or set environment variable
export VLLM_MLX_TEST_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
python tests/test_paged_cache_real_inference.py

# Or modify directly in test file (line 31):
# model_name = "mlx-community/Qwen3-0.6B-8bit"  # Change this
```

**Production Deployment:**

```bash
# Basic production server with paged cache
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000

# High-concurrency production setup (recommended for 50+ users)
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --max-num-seqs 128 \
  --paged-cache-block-size 64 \
  --max-cache-blocks 2000 \
  --port 8000

# Larger model for production
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --continuous-batching \
  --use-paged-cache \
  --max-num-seqs 64 \
  --port 8000

# With systemd for production (create /etc/systemd/system/vllm-mlx.service)
# [Service]
# ExecStart=/usr/local/bin/vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
#   --continuous-batching --use-paged-cache --port 8000
# Restart=always
```

**CLI Options for Paged Cache:**

| Option | Description | Default |
|--------|-------------|---------|
| `--use-paged-cache` | Enable paged KV cache | `false` |
| `--paged-cache-block-size` | Tokens per cache block | `64` |
| `--max-cache-blocks` | Maximum number of cache blocks | `1000` |
| `--max-num-seqs` | Max concurrent sequences | `256` |
| `--continuous-batching` | Required for paged cache | `false` |

**Real Inference Results - Qwen3-0.6B-8bit (M4 Max, 128GB):**

*Test: 20 real inference requests in 2 rounds (10 per round) with ~286 token shared system prompt*

```
======================================================================
  PAGED KV CACHE - REAL INFERENCE TEST
  (20 requests in 2 rounds - cache reuse on 2nd round)
======================================================================

--------------------------------------------------
Test 1: WITHOUT Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.47s
  Throughput: 681.2 tok/s
  Cache hits: 0
  Tokens saved: 0

--------------------------------------------------
Test 2: WITH Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.31s
  Throughput: 765.8 tok/s

  Paged Cache Stats:
    Blocks allocated: 25
    Shared blocks: 4
    Cache hits: 10
    Tokens saved: 2560

==================================================
SUMMARY
==================================================
  Without paged cache: 681.2 tok/s
  With paged cache:    765.8 tok/s

  Speedup: 1.12x
  Cache hits: 10 (all Round 2 requests)
  Tokens saved: 2,560 (~256 tokens × 10 requests)
==================================================
```

*Sample model outputs (real inference):*
```
Q1: How do I implement a REST API in Python with FastAPI?
A1: I will implement a REST API in Python with FastAPI using a basic example...

Q2: What's the difference between SQL and NoSQL databases?
A2: The main difference between SQL and NoSQL databases is the type of data...

Q3: Explain async/await in JavaScript with an example.
A3: First, let's start with what async/await does. Async/await is a JavaScript feature...
```

**When to Use Paged Cache:**

| Use Case | Recommendation |
|----------|----------------|
| Single user, local development | Standard cache (default) |
| Multiple concurrent users | ✅ **Use Paged Cache** |
| Shared system prompts (chatbots, APIs) | ✅ **Use Paged Cache** |
| Memory-constrained environments | ✅ **Use Paged Cache** |
| High-throughput production servers | ✅ **Use Paged Cache** |

**Example: Chat Application with Shared System Prompt**

```python
# Start server with paged cache enabled
# vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --use-paged-cache

from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# All users share this system prompt (~500 tokens)
SYSTEM_PROMPT = """You are a helpful assistant specialized in..."""

# User 1: Creates 8 blocks for system prompt
response1 = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello!"}
    ]
)

# User 2-N: Share the 8 blocks via ref_count++ (no new allocation!)
response2 = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},  # Cache HIT
        {"role": "user", "content": "Different question"}
    ]
)
# Memory savings: 80%+ for 10+ concurrent users
```

#### VLM KV Cache (Image + Video) Test

This test uses a real VLM model plus the same image and video assets as the benchmark utilities.

Run:
```bash
python tests/test_vlm_cache.py
```

Example output:
```
======================================================================
  VLM KV CACHE TEST
======================================================================

  Model: mlx-community/Qwen3-VL-4B-Instruct-3bit
  Test: Verify KV cache reuse for repeated image/video + prompt combinations
  Expected behavior:
    - Same image + same prompt → cache HIT
    - Same image + different prompt → cache MISS
    - Different image + same prompt → cache MISS
    - Same video + same fps/max_frames → cache HIT
    - Same video + different fps/max_frames → cache MISS

----------------------------------------------------------------------
  SETUP: Loading Model
----------------------------------------------------------------------
    Model loaded in 0.11s
    Model type: qwen3_vl
    KV cache: 36 layers of KVCache

----------------------------------------------------------------------
  SETUP: Downloading Test Images
----------------------------------------------------------------------
    Image 1: 1200x989
    Image 2: 1200x1198
    Resized: 224x224, 336x336, 512x512, 768x768

----------------------------------------------------------------------
  SETUP: Downloading Test Videos
----------------------------------------------------------------------
    Video 1: 640x360, 10.0s @ 30.0fps
    Video 2: 424x240, 596.5s @ 24.0fps

----------------------------------------------------------------------
  TEST 1: Image Cache - Basic Hit/Miss
----------------------------------------------------------------------
    Results:
    Step | Description             | Expected | Actual | Time   | Status
    -----+-------------------------+----------+--------+--------+-------
    1a   | First request (new)     | MISS     | MISS   | 0.39ms | ✓
    1b   | Same image+prompt       | HIT      | HIT    | 0.42ms | ✓
    1c   | Same image, diff prompt | MISS     | MISS   | 0.25ms | ✓

----------------------------------------------------------------------
  TEST 2: Different Images = Different Cache Keys
----------------------------------------------------------------------
    Results:
    Step | Description    | Expected | Actual | Time   | Status
    -----+----------------+----------+--------+--------+-------
    2.2a | Image 2 first  | MISS     | MISS   | 0.20ms | ✓
    2.2b | Image 2 cached | HIT      | HIT    | 0.34ms | ✓

----------------------------------------------------------------------
  TEST 3: Resized Images = Different Cache Keys
----------------------------------------------------------------------
    (Cache uses content hash, so different sizes = different keys)

    Results:
    Step | Description    | Expected | Actual | Time   | Status
    -----+----------------+----------+--------+--------+-------
    3.1a | 224x224 first  | MISS     | MISS   | 0.06ms | ✓
    3.1b | 224x224 cached | HIT      | HIT    | 0.17ms | ✓
    3.2a | 336x336 first  | MISS     | MISS   | 0.06ms | ✓
    3.2b | 336x336 cached | HIT      | HIT    | 0.19ms | ✓
    3.3a | 512x512 first  | MISS     | MISS   | 0.16ms | ✓
    3.3b | 512x512 cached | HIT      | HIT    | 0.20ms | ✓
    3.4a | 768x768 first  | MISS     | MISS   | 0.12ms | ✓
    3.4b | 768x768 cached | HIT      | HIT    | 0.24ms | ✓

----------------------------------------------------------------------
  TEST 4: Video Cache - fps/max_frames in Cache Key
----------------------------------------------------------------------
    Config: fps=2.0, max_frames=16

    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    4a     | Video first request       | MISS     | MISS   | 0.03ms | ✓
    4b     | Same video+params         | HIT      | HIT    | 0.14ms | ✓
    4c     | Different fps (4.0)       | MISS     | MISS   | 0.01ms | ✓
    4d     | Different max_frames (32) | MISS     | MISS   | 0.01ms | ✓
    4.0.5a | fps=0.5 first             | MISS     | MISS   | 0.01ms | ✓
    4.0.5b | fps=0.5 cached            | HIT      | HIT    | 0.14ms | ✓
    4.1.0a | fps=1.0 first             | MISS     | MISS   | 0.01ms | ✓
    4.1.0b | fps=1.0 cached            | HIT      | HIT    | 0.14ms | ✓
    4.2.0a | fps=2.0 first             | MISS     | MISS   | 0.01ms | ✓
    4.2.0b | fps=2.0 cached            | HIT      | HIT    | 0.14ms | ✓
    4.4.0a | fps=4.0 first             | MISS     | MISS   | 0.01ms | ✓
    4.4.0b | fps=4.0 cached            | HIT      | HIT    | 0.14ms | ✓

----------------------------------------------------------------------
  TEST 5: Additional Videos
----------------------------------------------------------------------
    Results:
    Step | Description    | Expected | Actual | Time   | Status
    -----+----------------+----------+--------+--------+-------
    5.2a | Video 2 first  | MISS     | MISS   | 0.01ms | ✓
    5.2b | Video 2 cached | HIT      | HIT    | 0.14ms | ✓

----------------------------------------------------------------------
  TEST 6: LRU Eviction Policy
----------------------------------------------------------------------
    Cache capacity: 2 entries (currently 2/2)
    Touched img1 to make it recently used

    Results:
    Step | Description            | Expected | Actual | Time   | Status
    -----+------------------------+----------+--------+--------+-------
    6a   | img2 (oldest, evicted) | MISS     | MISS   | 0.01ms | ✓
    6b   | img1 (recently used)   | HIT      | HIT    | 0.14ms | ✓
    6c   | img3 (newest)          | HIT      | HIT    | 0.14ms | ✓

    Evictions: 1

======================================================================
  TEST RESULTS SUMMARY
======================================================================

    Final Cache Statistics:
    Metric           | Value
    -----------------+------
    Total Hits       | 12
    Total Misses     | 15
    Hit Rate         | 44.4%
    Tokens Saved     | 6300
    Image/Video Hits | 12
    Evictions        | 0

======================================================================
  ✓ ALL TESTS PASSED - VLM cache working correctly!
======================================================================
```

**Cache Key Strategy:**
- Images: `hash(image_content) + hash(prompt)`
- Videos: `hash(video_path) + hash(fps) + hash(max_frames) + hash(prompt)`

**Metrics:**
| Metric | Description |
|--------|-------------|
| Hits | Cache hits for image/video + prompt combinations |
| Misses | Requests not found in cache (computed from scratch) |
| Hit Rate | Hits / total queries |
| Tokens Saved | Prompt tokens skipped thanks to cache reuse |
| Image/Video Hits | Hits where at least one image/video was present |
| Evictions | LRU entries removed when cache is full |

## Supported Models

**All quantized models from [mlx-community on HuggingFace](https://huggingface.co/mlx-community/models) are compatible!**

Browse thousands of pre-optimized models at: **https://huggingface.co/mlx-community/models**

### Language Models (via mlx-lm)
- Llama 3.x (1B, 3B, 8B, 70B - 4-bit quantized)
- Mistral (7B, Mixtral 8x7B - 4-bit/8-bit quantized)
- Qwen2 (0.5B to 72B - various quantizations)
- Phi-3 (3.8B, 14B - 4-bit quantized)
- Gemma 2 (2B, 9B, 27B - 4-bit quantized)
- DeepSeek (7B, 33B, 67B - 4-bit quantized)
- And thousands more at [mlx-community](https://huggingface.co/mlx-community/models)

### Multimodal Language Models (via mlx-vlm)

| Model Family | Example Models |
|--------------|----------------|
| **Qwen-VL** | `Qwen3-VL-4B-Instruct-3bit`, `Qwen3-VL-30B-A3B-Instruct-6bit`, `Qwen2-VL-2B/7B-Instruct-4bit` |
| **LLaVA** | `llava-1.5-7b-4bit`, `llava-v1.6-mistral-7b-4bit`, `llava-llama-3-8b-v1_1-4bit`, `llava-interleave-qwen-7b-4bit` |
| **Idefics** | `Idefics3-8B-Llama3-4bit`, `idefics2-8b-4bit`, `idefics2-8b-chatty-4bit` |
| **PaliGemma** | `paligemma2-3b-mix-224-4bit`, `paligemma-3b-mix-224-8bit`, `paligemma2-10b-ft-docci-448-6bit` |
| **Pixtral** | `pixtral-12b-4bit`, `pixtral-12b-8bit`, `pixtral-12b-bf16` |
| **Molmo** | `Molmo-7B-D-0924-4bit`, `Molmo-7B-D-0924-8bit` |
| **Phi-3 Vision** | `Phi-3-vision-128k-instruct-4bit`, `Phi-3-vision-128k-instruct-8bit` |
| **DeepSeek-VL** | `deepseek-vl-7b-chat-4bit`, `deepseek-vl2-small-4bit`, `deepseek-vl2-4bit` |

**Find all multimodal models at [mlx-community](https://huggingface.co/mlx-community/models)** - filter by `-VL-`, `llava`, `paligemma`, `pixtral`, `molmo`, `idefics`, `deepseek-vl` patterns.

## Architecture

```
┌─────────────────────────────────────────────┐
│              vLLM API Layer                 │
│     (OpenAI-compatible interface)           │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              MLXPlatform                    │
│   (vLLM platform plugin for Apple Silicon) │
└─────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│     mlx-lm       │   │     mlx-vlm       │
│  (LLM inference) │   │  (MLLM inference) │
└──────────────────┘   └──────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
┌─────────────────────────────────────────────┐
│                   MLX                       │
│    (Apple ML Framework - Metal kernels)    │
└─────────────────────────────────────────────┘
```

## CLI Commands

vllm-mlx provides three CLI commands:

### `vllm-mlx serve` - OpenAI-Compatible Server

Start an OpenAI-compatible API server:

```bash
# Simple mode (default) - Maximum throughput for single user
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --port 8000

# Continuous batching mode - For multiple concurrent users
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --port 8000 --continuous-batching

# With custom max tokens (default: 32768)
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --max-tokens 16384
```

#### When to Use Each Mode

| Mode | Use Case | Performance |
|------|----------|-------------|
| **Simple (default)** | Single user, local development, maximum speed | ~1000 tok/s (Qwen3-0.6B) |
| **Continuous Batching** | Multiple concurrent users, production deployment | ~300-600 tok/s per user (batched) |

**Simple Mode (Default):**
- Best for: Single user, CLI tools, local testing, Open WebUI with one user
- Performance: Maximum tokens/second (~1000 tok/s for Qwen3-0.6B)
- Trade-off: One request at a time (no concurrent handling)

**Continuous Batching Mode (`--continuous-batching`):**
- Best for: Production servers, multiple concurrent users, API services
- Performance: 1.5-3x higher total throughput with multiple users
- Trade-off: Higher latency per request, more overhead

```bash
# Example: Local development (single user)
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit

# Example: Production server (multiple users)
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --max-num-seqs 64

# Example: Production with paged cache (memory efficient, shared system prompts)
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --use-paged-cache
```

| Argument | Description | Default |
|----------|-------------|---------|
| `model` | Model name from HuggingFace or local path | **Required** |
| `--host` | Host address to bind | `0.0.0.0` |
| `--port` | Port number | `8000` |
| `--max-tokens` | Default max tokens for generation | `32768` |
| `--continuous-batching` | Enable batching for multiple users | `false` |
| `--stream-interval` | Tokens to batch before streaming (only with --continuous-batching) | `1` |
| `--max-num-seqs` | Max concurrent sequences (only with --continuous-batching) | `256` |
| `--prefill-batch-size` | Prefill batch size (only with --continuous-batching) | `8` |
| `--completion-batch-size` | Completion batch size (only with --continuous-batching) | `32` |
| `--enable-prefix-cache` | Enable prefix caching (only with --continuous-batching) | `true` |
| `--disable-prefix-cache` | Disable prefix caching | `false` |
| `--prefix-cache-size` | Max entries in prefix cache | `100` |
| `--use-paged-cache` | Enable paged KV cache for memory efficiency | `false` |
| `--paged-cache-block-size` | Tokens per cache block | `64` |
| `--max-cache-blocks` | Maximum number of cache blocks | `1000` |
| `--mcp-config` | Path to MCP configuration file for tool integration | `None` |

#### API Streaming Control

Streaming is controlled per-request using the `stream` parameter in the API:

```bash
# Non-streaming (wait for complete output) - Slightly higher throughput
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'

# Streaming (tokens sent as generated) - Better UX for interactive use
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

**Performance Tip:** Non-streaming (`stream: false`) gives **5-10% higher tokens/second**. Use streaming for interactive applications where users want to see tokens as they're generated.

#### Open WebUI Integration

vllm-mlx is compatible with [Open WebUI](https://github.com/open-webui/open-webui) for a ChatGPT-like interface:

```bash
# 1. Start vllm-mlx server
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --port 8000

# 2. Start Open WebUI with Docker
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -e WEBUI_AUTH=false \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

# 3. Open http://localhost:3000 in your browser
```

**Notes:**
- `host.docker.internal` allows Docker to connect to your Mac's localhost
- Use `--continuous-batching` if multiple users will chat simultaneously
- For VLM models (image/video), Open WebUI supports multimodal content

### `vllm-mlx-chat` - Gradio Chat Interface

Launch a web-based chat interface:

```bash
# Multimodal mode (default) - supports text, images, and video
vllm-mlx-chat --server-url http://localhost:8000 --port 7860

# Text-only mode - faster, no multimodal overhead
vllm-mlx-chat --text-only --port 7860
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--server-url` | URL of the vllm-mlx server | `http://localhost:8000` |
| `--port` | Port for Gradio web interface | `7860` |
| `--share` | Create a public share link | `false` |
| `--text-only` | Use text-only mode (no image/video support) | `false` |
| `--max-tokens` | Maximum tokens to generate | `2048` |
| `--temperature` | Sampling temperature | `0.7` |

### `vllm-mlx-bench` - Performance Benchmark

Run performance benchmarks to measure inference speed for LLM, MLLM images, and video:

#### LLM Benchmark
```bash
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 10 --max-tokens 256
```

#### MLLM Image Benchmark (auto-detected)
```bash
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --quick
```

#### MLLM Video Benchmark
```bash
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video --video-url https://example.com/video.mp4
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name from HuggingFace or local path | **Required** |
| `--prompts` | Number of test prompts to run (LLM only) | `5` |
| `--max-tokens` | Maximum tokens to generate per prompt | `256` |
| `--temperature` | Sampling temperature (0 = deterministic) | `0.7` |
| `--warmup` | Number of warmup runs before measuring | `1` |
| `--output` | Save results to JSON file | `None` |
| `--mllm` | Force MLLM mode (auto-detected by default) | `false` |
| `--video` | Run video benchmark instead of image | `false` |
| `--video-url` | URL of video for benchmark | Big Buck Bunny 10s |
| `--video-path` | Local path to video file | `None` |
| `--quick` | Quick benchmark with fewer configurations | `false` |

**LLM Metrics:**

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token - how fast the model starts responding (ms) |
| **TPOT** | Time Per Output Token - time between each generated token (ms/token) |
| **Generation TPS** | Output tokens per second (tok/s) |
| **Processing TPS** | Input/prompt tokens processed per second (tok/s) |
| **End-to-End Latency** | Total time from request to complete response |
| **Total Throughput** | Overall tokens (input + output) per second |
| **Requests/Second** | Number of requests the system can handle per second |

**MLLM Image Metrics:** Tok/s at different resolutions (224x224 to 1920x1080)

**MLLM Video Metrics:** Tok/s at different frame counts (2 to 32 frames)

**Resource Metrics:**

| Metric | Description |
|--------|-------------|
| **Process Memory** | Peak RAM usage of the Python process (GB) |
| **MLX Peak Memory** | Peak GPU memory used by MLX during inference (GB) |
| **MLX Cache Memory** | Memory used by MLX's computation cache (GB) |
| **System Memory** | Total system RAM usage with percentage |

**Example output:**
```
============================================================
BENCHMARK RESULTS
============================================================

Model          mlx-community/Llama-3.2-1B-Instruct-4bit
Hardware       M4 Max (128 GB)
Total Runs     10
Input Tokens   774
Output Tokens  2,434
Total Time     6.53s

Performance Metrics:
Metric                        Mean          P95/Max
----------------------------  ------------  -----------
TTFT (Time to First Token)    60.5 ms       84.9 ms
TPOT (Time Per Output Token)  2.18 ms       2.21 ms
Generation Speed              459.5 tok/s   462.6 tok/s
Processing Speed              1068.3 tok/s  -
Latency (per request)         0.59s         0.65s

Throughput:
Total Throughput  491.3 tok/s
Requests/Second   1.53 req/s

Resource Usage:
Process Memory (peak)  1.30 GB
MLX Peak Memory        0.91 GB
MLX Cache Memory       0.06 GB
System Memory          25.1 / 128 GB (20%)
```

### GSM8K Evaluation

Run math reasoning evaluation on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) benchmark:

```bash
# Start server
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --port 9000

# Run GSM8K evaluation (10 questions for quick test)
python tests/evals/gsm8k/gsm8k_eval.py --port 9000 --num-questions 10

# Run full GSM8K test set (1319 questions)
python tests/evals/gsm8k/gsm8k_eval.py --port 9000

# Save results to JSON
python tests/evals/gsm8k/gsm8k_eval.py --port 9000 --output results.json
```

## MCP and Tool Calling

vllm-mlx supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for tool integration, allowing LLMs to interact with external tools like filesystems, databases, and APIs.

### How Tool Calling Works

```
┌─────────────┐     1. Request + tools     ┌─────────────┐
│   Client    │ ─────────────────────────► │   Server    │
│  (your app) │                            │  vllm-mlx   │
└─────────────┘                            └─────────────┘
                                                  │
                                           2. LLM generates
                                              tool_calls
                                                  ▼
┌─────────────┐     3. tool_calls          ┌─────────────┐
│   Client    │ ◄───────────────────────── │     LLM     │
│  (your app) │                            │   (Qwen)    │
└─────────────┘                            └─────────────┘
       │
       │ 4. YOUR APP executes
       │    the function via MCP
       ▼
┌─────────────┐     5. Execute tool        ┌─────────────┐
│   Client    │ ─────────────────────────► │ MCP Server  │
│  (your app) │                            │ (filesystem)│
└─────────────┘                            └─────────────┘
       │
       │ 6. Result
       ▼
┌─────────────┐  7. Send tool result       ┌─────────────┐
│   Client    │ ─────────────────────────► │   Server    │
│  (your app) │                            │  vllm-mlx   │
└─────────────┘                            └─────────────┘
       │
       │ 8. LLM generates final response
       ▼
    Response
```

**Important:** The LLM does NOT execute functions directly. It only says "I want to call this function with these arguments". Your application executes the tool and sends the result back.

### Quick Start with MCP

1. **Create MCP config file** (`mcp.json`):

```json
{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "enabled": true,
      "timeout": 30
    }
  },
  "max_tool_calls": 10,
  "default_timeout": 30.0
}
```

2. **Install MCP SDK**:

```bash
pip install mcp
```

3. **Start server with MCP**:

```bash
# Simple mode with MCP
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Continuous batching with MCP
vllm-mlx serve mlx-community/Qwen3-4B-4bit --continuous-batching --mcp-config mcp.json
```

4. **Check MCP status**:

```bash
curl http://localhost:8000/health | python -m json.tool
# Returns: {"mcp": {"enabled": true, "servers_connected": 1, "tools_available": 14}}

curl http://localhost:8000/v1/mcp/tools | python -m json.tool
# Returns list of available tools
```

### Tool Calling Example

```python
from openai import OpenAI
import requests
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 1. Get available MCP tools
tools_response = requests.get("http://localhost:8000/v1/mcp/tools").json()
tools = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
    }
    for tool in tools_response.get("tools", [])
]

# 2. Send request with tools
messages = [{"role": "user", "content": "List the files in /tmp"}]
response = client.chat.completions.create(
    model="default",
    messages=messages,
    tools=tools,
    max_tokens=500,
)

message = response.choices[0].message

# 3. If LLM wants to use tools, execute them
if message.tool_calls:
    for tool_call in message.tool_calls:
        # Execute tool via MCP
        result = requests.post(
            "http://localhost:8000/v1/mcp/execute",
            json={
                "tool_name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
        ).json()

        print(f"Tool: {tool_call.function.name}")
        print(f"Result: {result['content']}")

        # Add tool result to conversation
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result.get("content", "")),
        })

    # 4. Get final response with tool results
    final_response = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=500,
    )
    print(f"Assistant: {final_response.choices[0].message.content}")
```

### MCP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/mcp/tools` | GET | List all available MCP tools |
| `/v1/mcp/execute` | POST | Execute an MCP tool |
| `/v1/mcp/servers` | GET | List MCP server status |

### Example MCP Servers

Install additional MCP servers for different capabilities:

```bash
# Filesystem access
npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir

# GitHub integration
npx -y @modelcontextprotocol/server-github

# PostgreSQL database
npx -y @modelcontextprotocol/server-postgres postgresql://...

# Brave Search
npx -y @anthropic/mcp-server-brave-search
```

Configure in `mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "your-token"}
    }
  }
}
```

### Interactive MCP Chat

Run the interactive chat example:

```bash
python examples/mcp_chat.py
```

This provides a terminal-based chat where the LLM can use filesystem tools to create, read, and modify files.

## Hardware Detection

vllm-mlx can detect your Mac's hardware specifications:

```python
from vllm_mlx.optimizations import detect_hardware

hw = detect_hardware()
print(f"Chip: {hw.chip_name}")           # e.g., "M4 Max"
print(f"Memory: {hw.total_memory_gb} GB")
print(f"Bandwidth: {hw.memory_bandwidth_gbs} GB/s")
print(f"GPU Cores: {hw.gpu_cores}")
```

Supported chips: M1, M1 Pro/Max/Ultra, M2, M2 Pro/Max/Ultra, M3, M3 Pro/Max/Ultra, M4, M4 Pro/Max/Ultra

## Limitations

- **Limited batching**: Optimized for single-request throughput
- **No CUDA graphs**: Not applicable on Metal
- **Memory bound**: Unified memory is shared with system (typically 8-128GB)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black vllm_mlx/
ruff check vllm_mlx/
```

## Contributing

Contributions are welcome! This project is under active development and we appreciate:

- Bug reports and feature requests via [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues)
- Pull requests for bug fixes, optimizations, or new features
- Documentation improvements
- Performance benchmarks on different Apple Silicon chips

Please submit PRs to [https://github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Citation

If you use vLLM-MLX in your research or project, please cite:

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title = {vLLM-MLX: Apple Silicon MLX Backend for vLLM},
  year = {2025},
  url = {https://github.com/waybarrios/vllm-mlx},
  note = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

**Repository**: [https://github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## Acknowledgments

This project builds upon excellent work from:

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving framework
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference on MLX
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models on MLX

**Developed by**: [Wayner Barrios](https://github.com/waybarrios)
**Project**: [vLLM-MLX](https://github.com/waybarrios/vllm-mlx)
