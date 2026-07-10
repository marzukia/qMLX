# SPDX-License-Identifier: Apache-2.0
"""
Tool call parsers for rapid-mlx.

This module provides tool call parsing functionality for various model formats.
Inspired by vLLM's tool parser architecture but simplified for MLX backend.

Available parsers:
- auto: Auto-detecting parser that tries all formats (default)
- mistral: Mistral models ([TOOL_CALLS] format)
- qwen/qwen3: Qwen models (<tool_call> and [Calling tool:] formats)
- llama/llama3/llama4: Llama models (<function=name> format)
- hermes/nous: Hermes/NousResearch models
- deepseek/deepseek_r1: DeepSeek V2 / R1-distill models (unicode tokens)
- deepseek_v3/deepseek_r1_0528: DeepSeek V3 family — function-typed
  fenced-JSON wire (R1-0528-Qwen3-8B and other V3-chat-template
  checkpoints). R12-5 split from deepseek_v31.
- kimi/kimi_k2/moonshot: Kimi/Moonshot models
- granite/granite3: IBM Granite models
- nemotron/nemotron3: NVIDIA Nemotron models
- xlam: Salesforce xLAM models
- functionary/meetkai: MeetKai Functionary models
- glm47/glm4: GLM-4.7 and GLM-4.7-Flash models
- harmony/gpt-oss: GPT-OSS models (Harmony format with channels)
- seed_oss/seed/gpt_oss: Seed-OSS / GPT-OSS models (XML format)
- deepseek_v31: DeepSeek V3.1 thinking-channel wire shape only
- qwen/qwen3/qwen3_xml: Qwen models (<tool_call>JSON</tool_call> and [Calling tool:] formats)
- qwen3_coder_xml: Qwen3-Coder models (<function=NAME> XML format)

Usage:
    from vllm_mlx.tool_parsers import ToolParserManager

    # Get a parser by name
    parser_cls = ToolParserManager.get_tool_parser("mistral")
    parser = parser_cls(tokenizer)

    # Parse tool calls
    result = parser.extract_tool_calls(model_output)
    if result.tools_called:
        for tc in result.tool_calls:
            print(f"Tool: {tc['name']}, Args: {tc['arguments']}")

    # List available parsers
    print(ToolParserManager.list_registered())
"""

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

# Import parsers to register them
from .auto_tool_parser import AutoToolParser
from .deepseek_tool_parser import DeepSeekToolParser
from .deepseek_v3_tool_parser import DeepSeekV3ToolParser
from .deepseekv31_tool_parser import DeepSeekV31ToolParser
from .harmony_tool_parser import HarmonyToolParser
from .hermes_tool_parser import HermesToolParser
from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .qwen_tool_parser import QwenToolParser
from .ui_tars_tool_parser import UiTarsToolParser

__all__ = [
    # Base classes
    "ToolParser",
    "ToolParserManager",
    "ExtractedToolCallInformation",
    # Specific parsers
    "AutoToolParser",
    "QwenToolParser",
    "HermesToolParser",
    "DeepSeekToolParser",
    "HarmonyToolParser",
    "DeepSeekV3ToolParser",
    "DeepSeekV31ToolParser",
    "Qwen3CoderToolParser",
    "UiTarsToolParser",
]
