# vLLM-MLX Documentation

**Apple Silicon MLX Backend for vLLM** - GPU-accelerated Qwen text inference on Mac

## What is vLLM-MLX?

qmlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization

## Key Features

- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **OpenAI API compatible** - drop-in replacement for OpenAI client
- **MCP Tool Calling** - integrate external tools via Model Context Protocol
- **Paged KV Cache** - memory-efficient caching with prefix sharing
- **Continuous Batching** - high throughput for multiple concurrent users

## Quick Links

### Getting Started
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md) — including `qmlx chat` for an instant REPL

### User Guides
- [OpenAI-Compatible Server](guides/server.md)
- [Python API](guides/python-api.md)
- [Reasoning Models](guides/reasoning.md)
- [Tool Calling](guides/tool-calling.md)
- [MCP & Tool Calling](guides/mcp-tools.md)
- [Continuous Batching](guides/continuous-batching.md)
- [AI Client Compatibility](guides/ai-clients.md)
- [SDK Compatibility Notes](guides/sdk-compat.md)

### Reference
- [CLI Commands](reference/cli.md)
- [Supported Models](reference/models.md)
- [Configuration](reference/configuration.md)

### Benchmarks
- [LLM Benchmarks](benchmarks/llm.md)

### Development
- [Architecture](development/architecture.md)
- [Contributing](development/contributing.md)

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 8GB+ RAM recommended

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.
