# MCP & Tool Calling

vllm-mlx supports the Model Context Protocol (MCP) for integrating external tools with LLMs.

## How Tool Calling Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Tool Calling Flow                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User Request                                                    │
│     ─────────────────►  "List files in /tmp"                       │
│                                                                     │
│  2. LLM Generates Tool Call                                         │
│     ─────────────────►  tool_calls: [{                             │
│                           name: "list_directory",                   │
│                           arguments: {path: "/tmp"}                 │
│                         }]                                          │
│                                                                     │
│  3. App Executes Tool via MCP                                       │
│     ─────────────────►  MCP Server executes list_directory         │
│                         Returns: ["file1.txt", "file2.txt"]        │
│                                                                     │
│  4. Tool Result Sent Back to LLM                                    │
│     ─────────────────►  role: "tool", content: [...]               │
│                                                                     │
│  5. LLM Generates Final Response                                    │
│     ─────────────────►  "The /tmp directory contains 2 files..."   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create MCP Config

Create `mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### 2. Start Server with MCP

```bash
# Simple mode
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Continuous batching
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json --continuous-batching
```

### 3. Verify MCP Status

```bash
# Check MCP status
curl http://localhost:8000/v1/mcp/status

# List available tools
curl http://localhost:8000/v1/mcp/tools
```

## Tool Calling Example

```python
import json
import httpx

BASE_URL = "http://localhost:8000"

# 1. Get available tools
tools_response = httpx.get(f"{BASE_URL}/v1/mcp/tools")
tools = tools_response.json()["tools"]

# 2. Send request with tools
response = httpx.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "List files in /tmp"}],
        "tools": tools,
        "max_tokens": 1024
    }
)

result = response.json()
message = result["choices"][0]["message"]

# 3. Check for tool calls
if message.get("tool_calls"):
    tool_call = message["tool_calls"][0]

    # 4. Execute tool via MCP
    exec_response = httpx.post(
        f"{BASE_URL}/v1/mcp/execute",
        json={
            "server": "filesystem",
            "tool": tool_call["function"]["name"],
            "arguments": json.loads(tool_call["function"]["arguments"])
        }
    )
    tool_result = exec_response.json()

    # 5. Send result back to LLM
    messages = [
        {"role": "user", "content": "List files in /tmp"},
        message,
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(tool_result["result"])
        }
    ]

    final_response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"model": "default", "messages": messages}
    )
    print(final_response.json()["choices"][0]["message"]["content"])
```

## MCP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/mcp/status` | GET | Check MCP status |
| `/v1/mcp/tools` | GET | List available tools |
| `/v1/mcp/execute` | POST | Execute a tool |

## Example MCP Servers

### Filesystem

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### GitHub

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

### PostgreSQL

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

### Brave Search

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-key"
      }
    }
  }
}
```

## Multiple MCP Servers

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

## Interactive MCP Chat

For testing MCP interactively:

```bash
python examples/mcp_chat.py
```

## Supported Tool Formats

vllm-mlx parses tool calls from both Qwen and Llama formats:

- Qwen: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- Llama: `{"name": "...", "parameters": {...}}`

## Troubleshooting

### MCP server not connecting

Check that the MCP server command is correct:
```bash
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### Tool not executing

Verify tool is available:
```bash
curl http://localhost:8000/v1/mcp/tools | jq '.tools[].name'
```

### Tool call not parsed

Ensure you're using a model that supports function calling (Qwen3, Llama-3.2-Instruct).
