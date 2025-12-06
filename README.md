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

### 🚧 What's Next (Phase 4: Optimizations)

**Performance & Scalability**
- ⏳ Request batching for higher throughput
- ⏳ KV cache optimization / prompt caching
- ⏳ Improved streaming performance
- ⏳ Memory optimization for large models
- ⏳ Multi-request concurrency improvements

**Advanced Features**
- ⏳ Structured output (JSON mode, grammar constraints)
- ⏳ Function calling / tool use
- ⏳ Vision-language reasoning chains
- ⏳ Fine-tuning support

**Current Limitations:**
- Single-request optimization (not yet optimized for batching)
- No prompt caching (each request processes full context)
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
python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
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
python -m vllm_mlx.server --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
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
# Llama 3.2-1B (lightweight model)
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 5 --max-tokens 256
```

**Real Performance (M4 Max, 128GB):**
- **Generation Speed**: 460.1 tok/s
- **TTFT**: 48.6 ms
- **Memory**: 0.69 GB MLX peak

```bash
# Llama 3.2-3B (balanced model)
vllm-mlx-bench --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompts 5 --max-tokens 256
```

**Real Performance (M4 Max, 128GB):**
- **Generation Speed**: 200.8 tok/s
- **TTFT**: 62.7 ms
- **Memory**: 1.79 GB MLX peak

#### Multimodal Image Benchmarks

```bash
# Quick image benchmark (4 resolutions)
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --quick
```

**Real Performance (M4 Max, 128GB):**
- **224x224**: ~80 tok/s
- **512x512**: ~64 tok/s
- **1024x768**: ~37 tok/s

#### Multimodal Video Benchmarks

```bash
# Quick video benchmark (3 frame counts)
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick
```

**Real Performance (M4 Max, 128GB):**
- **4 frames**: 98.2 tok/s (3.5 GB memory)
- **8 frames**: 69.1 tok/s (4.0 GB memory)
- **16 frames**: 41.3 tok/s (4.7 GB memory)

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

### `vllm-mlx` - OpenAI-Compatible Server

Start an OpenAI-compatible API server:

```bash
vllm-mlx --model mlx-community/Llama-3.2-1B-Instruct-4bit --port 8000
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name from HuggingFace or local path | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| `--host` | Host address to bind | `0.0.0.0` |
| `--port` | Port number | `8000` |
| `--mllm` | Force loading as MLLM (multimodal model) | `false` |

**MLLM models are auto-detected** by patterns like `-VL-`, `llava`, `paligemma`, etc. Use `--mllm` flag to force MLLM mode for custom models.

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

## Performance

Performance on Apple Silicon (compared to CPU-only vLLM):

| Model | M1 Pro | M2 Max | M3 Max |
|-------|--------|--------|--------|
| Llama-3.2-3B-4bit | ~30 tok/s | ~45 tok/s | ~60 tok/s |
| Mistral-7B-4bit | ~15 tok/s | ~25 tok/s | ~35 tok/s |
| Qwen2-VL-2B-4bit | ~20 tok/s | ~30 tok/s | ~40 tok/s |

*Note: Actual performance varies based on prompt length, generation settings, and system load.*

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
