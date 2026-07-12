# Quick Start

## Option 1: Interactive Chat (fastest first taste)

The shortest path to talking to a model — `chat` spawns its own server,
downloads the model on first run (~2.5 GB for the default `qwen3.5-4b-4bit`), and
drops you into a REPL.

```bash
qmlx chat                  # defaults to qwen3.5-4b-4bit
qmlx chat qwen3.5-9b-4bit       # a larger model (5 GB)
qmlx chat --think          # surface chain-of-thought reasoning
```

In-REPL: `/help`, `/reset`, `/save <path>`, `/model <alias>`, `/exit`. Type
`"""` on its own line to start/end a multi-line block. See the
[CLI reference](../reference/cli.md#qmlx-chat) for all flags.

## Option 2: OpenAI-Compatible Server

Start the server:

```bash
qmlx serve qwen3.5-4b-4bit --port 8000
```

Use with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Anthropic SDK (`/v1/messages`)

The same server also speaks the Anthropic Messages API. **Important:** point
`base_url` at the server root, **not** `…/v1` — the Anthropic SDK appends
`/v1/messages` itself, so passing `…/v1` produces `404` on every request
(L-01 in [SDK Compatibility Notes](../guides/sdk-compat.md#l-01--anthropic-sdk-base_url-must-not-include-v1)).

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",   # no trailing /v1
    api_key="not-needed",
)

message = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

## Option 3: Gradio Web UI

A browser-based chat UI ships in the optional `[chat]` extra:

```bash
pip install 'qmlx-serve[chat]'
# Then launch — see `qmlx help` for the UI entry point in your install.
```

## Reasoning Models

Reasoning parsers are auto-detected from the model name. The server splits
chain-of-thought into a separate `reasoning_content` field, leaving `content`
clean.

```bash
qmlx serve qwen3.5-9b-4bit --port 8000   # qwen3 reasoning parser auto-detected
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)
print(response.choices[0].message.content)            # final answer
print(response.choices[0].message.reasoning_content)  # thinking trace
```

## Tool Calling

Tool/function calling is on by default for supported model families (Qwen3.x,
GLM-4.7, GPT-OSS, Llama, Mistral, etc.) — the right parser is auto-detected:

```bash
qmlx serve qwen3.5-9b-4bit --port 8000
```

If you need to pin the parser manually:

```bash
qmlx serve devstral-24b-4bit \
  --enable-auto-tool-choice --tool-call-parser hermes
```

## Next Steps

- [Server Guide](../guides/server.md) - Full server configuration
- [Python API](../guides/python-api.md) - Direct API usage
- [Reasoning Models](../guides/reasoning.md) - Thinking models
- [Tool Calling](../guides/tool-calling.md) - Function calling
- [SDK Compatibility Notes](../guides/sdk-compat.md) - Where qmlx deviates from OpenAI/Anthropic specs
- [Supported Models](../reference/models.md) - Available models
