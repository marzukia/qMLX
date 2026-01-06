# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

Supports parsing tool calls from multiple model formats:
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Llama: <function=name>{"arg": "value"}</function>
"""

import json
import re
import uuid
from typing import List, Optional, Tuple

from .models import FunctionCall, ToolCall, ToolDefinition


def parse_tool_calls(text: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse tool calls from model output.

    Supports multiple formats:
    - Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Llama: <function=name>{"arg": "value"}</function>

    Args:
        text: Raw model output text

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    tool_calls = []
    cleaned_text = text

    # Pattern for Qwen-style tool calls: <tool_call>...</tool_call>
    qwen_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    qwen_matches = re.findall(qwen_pattern, text, re.DOTALL)

    for match in qwen_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                )
            ))
        except json.JSONDecodeError:
            continue

    # Remove tool call tags from cleaned text
    if qwen_matches:
        cleaned_text = re.sub(
            r'<tool_call>.*?</tool_call>',
            '',
            text,
            flags=re.DOTALL
        ).strip()

    # Pattern for Llama-style: <function=name>...</function>
    llama_pattern = r'<function=([^>]+)>(\{.*?\})</function>'
    llama_matches = re.findall(llama_pattern, text, re.DOTALL)

    for name, args_str in llama_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name.strip(),
                    arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                )
            ))
        except json.JSONDecodeError:
            continue

    if llama_matches:
        cleaned_text = re.sub(
            r'<function=[^>]+>.*?</function>',
            '',
            cleaned_text,
            flags=re.DOTALL
        ).strip()

    # Remove thinking tags if present (reasoning models)
    cleaned_text = re.sub(
        r'<think>.*?</think>',
        '',
        cleaned_text,
        flags=re.DOTALL
    ).strip()

    return cleaned_text, tool_calls if tool_calls else None


def convert_tools_for_template(
    tools: Optional[List]
) -> Optional[List[dict]]:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            tool_func = tool.get("function")
        else:
            tool_type = getattr(tool, "type", None)
            tool_func = getattr(tool, "function", None)

        if tool_type == "function" and tool_func:
            # Handle function as dict or Pydantic model
            if isinstance(tool_func, dict):
                func_name = tool_func.get("name", "")
                func_desc = tool_func.get("description", "")
                func_params = tool_func.get("parameters", {"type": "object", "properties": {}})
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(tool_func, "parameters", {"type": "object", "properties": {}})

            converted.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_desc,
                    "parameters": func_params
                }
            })

    return converted if converted else None


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
    }
