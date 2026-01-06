# Configuration Reference

## Server Configuration

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Server host address | `0.0.0.0` |
| `--port` | Server port | `8000` |
| `--max-tokens` | Default max tokens | `32768` |

### Batching Options

| Option | Description | Default |
|--------|-------------|---------|
| `--continuous-batching` | Enable batching | `false` |
| `--stream-interval` | Tokens per stream chunk | `1` |
| `--max-num-seqs` | Max concurrent sequences | `256` |
| `--prefill-batch-size` | Prefill batch size | `8` |
| `--completion-batch-size` | Completion batch size | `32` |

### Cache Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-prefix-cache` | Enable prefix caching | `true` |
| `--disable-prefix-cache` | Disable prefix caching | `false` |
| `--prefix-cache-size` | Max cache entries | `100` |
| `--use-paged-cache` | Enable paged KV cache | `false` |
| `--paged-cache-block-size` | Tokens per block | `64` |
| `--max-cache-blocks` | Maximum blocks | `1000` |

### MCP Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mcp-config` | Path to MCP config file | `None` |

## MCP Configuration

Create `mcp.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-name", "arg1"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### MCP Server Options

| Field | Description | Required |
|-------|-------------|----------|
| `command` | Executable command | Yes |
| `args` | Command arguments | Yes |
| `env` | Environment variables | No |

## API Request Options

### Chat Completions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model name | Required |
| `messages` | Chat messages | Required |
| `max_tokens` | Max tokens to generate | 256 |
| `temperature` | Sampling temperature | 0.7 |
| `top_p` | Nucleus sampling | 0.9 |
| `stream` | Enable streaming | `true` |
| `stop` | Stop sequences | None |
| `tools` | Tool definitions | None |

### Multimodal Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_fps` | Frames per second | 2.0 |
| `video_max_frames` | Max frames | 32 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Default model for tests |
| `HF_TOKEN` | HuggingFace authentication token |
| `OPENAI_API_KEY` | Set to any value for SDK compatibility |

## Example Configurations

### Development (Single User)

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Production (Multiple Users)

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --max-num-seqs 128 \
  --port 8000
```

### With MCP Tools

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --mcp-config mcp.json \
  --continuous-batching
```

### High Throughput

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --stream-interval 5 \
  --max-num-seqs 256
```
