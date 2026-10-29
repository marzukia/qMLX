# Tool Calling

qmlx supports OpenAI-compatible tool calling (function calling). The parser is auto-detected from the model name; pin it explicitly with `--tool-call-parser`.

## Quick Start

Enable tool calling by adding the `--enable-auto-tool-choice` flag when starting the server:

```bash
qmlx serve devstral-24b-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Then use tools with the standard OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
```

## Supported Parsers

The parser is auto-detected from the model name: the `hermes` parser
for Qwen3.x, Mistral/Devstral and most other chat models (none for
Qwen3.6). The `qwen` parser is explicit-only; pin it with
`--tool-call-parser qwen`:

| Parser | Aliases | Models | Format |
|--------|---------|--------|--------|
| `hermes` | `nous`, `qwen3_coder` | Hermes, NousResearch, Qwen3-Coder, Mistral, Devstral | `<tool_call></tool_call>` JSON |
| `qwen` | `qwen3`, `qwen3_xml` | Qwen, Qwen3 | `<tool_call></tool_call>` JSON in XML |
The non-Qwen tool parsers were removed in the Qwen-only simplification (#19);
models from those families that still ship tool-call templates are
served through the `hermes` parser where the wire shape matches.

## Model Examples

### Qwen

```bash
# Qwen3
qmlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser qwen
```

### Mistral / Devstral

```bash
# Devstral (optimized for coding and tool use)
qmlx serve devstral-24b-4bit \
  --enable-auto-tool-choice --tool-call-parser hermes
```

### Qwen3-Coder

```bash
qmlx serve qwen3-coder-30b-4bit \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

## Auto-Detection

If you don't pin `--tool-call-parser`, the engine auto-detects the
parser from the model name: `hermes` for Qwen3.x, Mistral/Devstral
and most other chat models, none for Qwen3.6. The `qwen` parser is
not auto-assigned; pin it with `--tool-call-parser qwen`. Run
`qmlx info` to see the config a model resolves to.

## Streaming Tool Calls

Tool calls work with streaming. The tool call information is sent when the model finishes generating:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's 25 * 17?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
```

## Handling Tool Results

After receiving a tool call, execute the function and send the result back:

```python
import json

# First request - model decides to call a tool
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# Get the tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Execute the function (your implementation)
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# Send result back to model
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)}
    ],
    tools=[weather_tool]
)

print(response.choices[0].message.content)
# "The weather in Tokyo is sunny with a temperature of 22C."
```

## Think Tag Handling

Models that produce `<think>...</think>` reasoning tags (like DeepSeek-R1, Qwen3, GLM-4.7) are handled automatically. The parser strips thinking content before extracting tool calls, so reasoning tags never interfere with tool call parsing.

This works even when `<think>` was injected in the prompt (implicit think tags with only a closing `</think>`).

## CLI Reference

| Option | Description |
|--------|-------------|
| `--enable-auto-tool-choice` | Enable automatic tool calling |
| `--tool-call-parser` | Select parser (see table above) |

See [CLI Reference](../reference/cli.md) for all options.
