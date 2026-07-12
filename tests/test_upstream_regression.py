# SPDX-License-Identifier: Apache-2.0
"""
Upstream regression tests — test cases ported from vLLM (vllm-project/vllm)
to verify our tool parser forks haven't broken correctness.

Sources:
  - tests/tool_parsers/test_glm4_moe_tool_parser.py
  - tests/tool_parsers/test_mistral_tool_parser.py
  - tests/tool_parsers/test_seed_oss_tool_parser.py
  - tests/tool_parsers/test_deepseekv31_tool_parser.py
  - tests/tool_parsers/test_qwen3coder_tool_parser.py
"""

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager

# ─── Fixtures ────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════
# GLM-4.7 (glm47) — ported from vLLM test_glm4_moe_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Mistral — ported from vLLM test_mistral_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


# ─── Fixtures for new parsers ────────────────────────────────────────


@pytest.fixture
def deepseekv31_parser():
    cls = ToolParserManager.get_tool_parser("deepseek_v31")
    return cls(tokenizer=None)


@pytest.fixture
def qwen3coder_parser():
    cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    return cls(tokenizer=None)


@pytest.fixture
def qwen3coder_request():
    """Request with tools for Qwen3-Coder type conversion tests."""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_area",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "shape": {"type": "string"},
                            "dimensions": {"type": "object"},
                            "precision": {"type": "integer"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_types",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "int_param": {"type": "integer"},
                            "float_param": {"type": "float"},
                            "bool_param": {"type": "boolean"},
                            "str_param": {"type": "string"},
                            "obj_param": {"type": "object"},
                        },
                    },
                },
            },
        ]
    }


# ═══════════════════════════════════════════════════════════════════════
# Seed-OSS — ported from vLLM test_seed_oss_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek V3.1 — ported from vLLM test_deepseekv31_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestDeepSeekV31UpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, deepseekv31_parser):
        """Plain text → no tool calls."""
        result = deepseekv31_parser.extract_tool_calls("This is a test", request=None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test"

    def test_single_tool_call(self, deepseekv31_parser):
        """Single tool call in V3.1 format (no code fence, no type prefix)."""
        output = (
            "normal text"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "foo"
        assert result.tool_calls[0]["arguments"] == '{"x":1}'
        assert result.content == "normal text"

    def test_multiple_tool_calls(self, deepseekv31_parser):
        """Multiple tool calls in V3.1 format."""
        output = (
            "some prefix text"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "foo"
        assert result.tool_calls[0]["arguments"] == '{"x":1}'
        assert result.tool_calls[1]["name"] == "bar"
        assert result.tool_calls[1]["arguments"] == '{"y":2}'
        assert result.content == "some prefix text"

    def test_content_preserved(self, deepseekv31_parser):
        """Content before tool calls is preserved."""
        output = (
            "I'll help with that!"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q":"test"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert result.content == "I'll help with that!"

    def test_no_tool_calls_start(self, deepseekv31_parser):
        """Without tool_calls_begin token, treat as content."""
        output = "Just some regular text without any special tokens"
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert not result.tools_called
        assert result.content == output

    def test_complex_json_args(self, deepseekv31_parser):
        """Tool call with nested JSON arguments."""
        output = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>create_event<｜tool▁sep｜>"
            '{"title":"Meeting","details":{"time":"3pm","room":"A1"}}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "create_event"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["title"] == "Meeting"
        assert args["details"]["room"] == "A1"


class TestDeepSeekV31UpstreamStreaming:
    """Streaming tests ported from upstream vLLM."""

    def test_streaming_no_tools(self, deepseekv31_parser):
        """Regular text → content delta."""
        result = deepseekv31_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
        )
        assert result is not None
        assert result["content"] == " world"

    def test_streaming_content_before_tools(self, deepseekv31_parser):
        """Content before tool calls start token."""
        result = deepseekv31_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Some text",
            delta_text="Some text",
        )
        assert result is not None
        assert result["content"] == "Some text"


# ═══════════════════════════════════════════════════════════════════════
# Qwen3-Coder XML — ported from vLLM test_qwen3coder_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestQwen3CoderUpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, qwen3coder_parser):
        """Plain text → no tool calls."""
        result = qwen3coder_parser.extract_tool_calls(
            "This is a test response without any tool calls", request=None
        )
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test response without any tool calls"

    def test_single_tool_call(self, qwen3coder_parser, qwen3coder_request):
        """Single tool call with <tool_call> wrapper."""
        output = (
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "<parameter=unit>\nfahrenheit\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "get_current_weather"
        args = json.loads(tc["arguments"])
        assert args == {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}

    def test_single_tool_with_content(self, qwen3coder_parser, qwen3coder_request):
        """Content before tool call is preserved."""
        output = (
            "Sure! Let me check the weather for you."
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "Sure! Let me check the weather for you."

    def test_parallel_tools(self, qwen3coder_parser, qwen3coder_request):
        """Multiple parallel tool calls."""
        output = (
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nOrlando\n</parameter>\n"
            "<parameter=state>\nFL\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0["city"] == "Dallas"
        assert args1["city"] == "Orlando"

    def test_type_conversion(self, qwen3coder_parser, qwen3coder_request):
        """Parameter type conversion based on tool schema."""
        output = (
            "<tool_call>\n<function=test_types>\n"
            "<parameter=int_param>\n42\n</parameter>\n"
            "<parameter=float_param>\n3.14\n</parameter>\n"
            "<parameter=bool_param>\ntrue\n</parameter>\n"
            "<parameter=str_param>\nhello world\n</parameter>\n"
            '<parameter=obj_param>\n{"key": "value"}\n</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["int_param"] == 42
        assert isinstance(args["int_param"], int)
        assert args["float_param"] == 3.14
        assert isinstance(args["float_param"], float)
        assert args["bool_param"] is True
        assert args["str_param"] == "hello world"
        assert args["obj_param"] == {"key": "value"}

    def test_object_with_single_quotes(self, qwen3coder_parser, qwen3coder_request):
        """Object parameter with single-quote JSON (Python literal)."""
        output = (
            "<tool_call>\n<function=test_types>\n"
            "<parameter=obj_param>\n{'key': 'value'}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["obj_param"] == {"key": "value"}

    def test_array_parameter_double_encoded_json_string(self, qwen3coder_parser):
        """Array parameters may arrive as double-encoded JSON strings."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "todowrite",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "todos": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                }
            ]
        }
        output = (
            "<tool_call>\n<function=todowrite>\n"
            "<parameter=todos>\n"
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"\n'
            "</parameter>\n"
            "</function>\n</tool_call>"
        )

        result = qwen3coder_parser.extract_tool_calls(output, request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"

    def test_array_parameter_nullable_type_list(self, qwen3coder_parser):
        """Schemas may encode nullable arrays as type lists."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "todowrite",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "todos": {
                                    "type": ["array", "null"],
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                }
            ]
        }
        output = (
            "<tool_call>\n<function=todowrite>\n"
            "<parameter=todos>\n"
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"\n'
            "</parameter>\n"
            "</function>\n</tool_call>"
        )

        result = qwen3coder_parser.extract_tool_calls(output, request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"

    def test_fallback_no_tool_call_tags(self, qwen3coder_parser, qwen3coder_request):
        """Bare <function=...> without <tool_call> wrapper also works."""
        output = (
            "<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "</function>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_current_weather"

    def test_missing_closing_parameter_tag(self, qwen3coder_parser, qwen3coder_request):
        """Missing </parameter> tag — graceful handling."""
        output = (
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "<parameter=unit>\nfahrenheit\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert "city" in args
        assert args["state"] == "TX"
        assert args["unit"] == "fahrenheit"

    def test_multiline_object_param(self, qwen3coder_parser, qwen3coder_request):
        """Object parameter spanning multiple lines."""
        output = (
            "<tool_call>\n<function=calculate_area>\n"
            "<parameter=shape>\nrectangle\n</parameter>\n"
            '<parameter=dimensions>\n{"width": 10, \n "height": 20}\n</parameter>\n'
            "<parameter=precision>\n2\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["shape"] == "rectangle"
        assert args["dimensions"] == {"width": 10, "height": 20}
        assert args["precision"] == 2

    def test_tool_with_content_and_typed_params(
        self, qwen3coder_parser, qwen3coder_request
    ):
        """Content before tool call with typed parameters."""
        output = (
            "Let me calculate that area for you."
            "<tool_call>\n<function=calculate_area>\n"
            "<parameter=shape>\ncircle\n</parameter>\n"
            '<parameter=dimensions>\n{"radius": 15.5}\n</parameter>\n'
            "<parameter=precision>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert result.content == "Let me calculate that area for you."
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["shape"] == "circle"
        assert args["dimensions"] == {"radius": 15.5}
        assert args["precision"] == 3


class TestQwen3CoderUpstreamStreaming:
    """Streaming tests ported from upstream vLLM."""

    def test_streaming_no_tools(self, qwen3coder_parser):
        """Regular text → content delta."""
        result = qwen3coder_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
        )
        assert result is not None
        assert result["content"] == " world"

    def test_streaming_content_before_tool(self, qwen3coder_parser):
        """Content before tool call is streamed."""
        result = qwen3coder_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me check",
            delta_text="Let me check",
        )
        assert result is not None
        assert result["content"] == "Let me check"

    def test_streaming_full_tool_call_multistep(
        self, qwen3coder_parser, qwen3coder_request
    ):
        """Multi-step streaming: header → { → param → } across calls."""
        deltas = [
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n",
            "<parameter=city>Dallas</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]
        text = ""
        collected = []
        for d in deltas:
            prev = text
            text += d
            r = qwen3coder_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=text,
                delta_text=d,
                request=qwen3coder_request,
            )
            if r:
                collected.append(r)

        names = [
            c["tool_calls"][0]["function"].get("name")
            for c in collected
            if "tool_calls" in c and "name" in c["tool_calls"][0].get("function", {})
        ]
        assert "get_current_weather" in names

        arg_parts = [
            c["tool_calls"][0]["function"]["arguments"]
            for c in collected
            if "tool_calls" in c
            and "arguments" in c["tool_calls"][0].get("function", {})
        ]
        full_args = "".join(arg_parts)
        assert full_args.startswith("{")
        assert full_args.endswith("}")
        parsed = json.loads(full_args)
        assert parsed["city"] == "Dallas"

    def test_streaming_array_parameter_nullable_type_list(self, qwen3coder_parser):
        """Streaming conversion also handles nullable array schemas."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "todowrite",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "todos": {
                                    "type": ["array", "null"],
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                }
            ]
        }
        deltas = [
            "<tool_call>\n<function=todowrite>\n",
            "<parameter=todos>\n",
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"\n'
            "</parameter>\n",
            "</function>\n</tool_call>",
        ]
        text = ""
        collected = []
        for delta in deltas:
            previous = text
            text += delta
            result = qwen3coder_parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=text,
                delta_text=delta,
                request=request,
            )
            if result:
                collected.append(result)

        arg_parts = [
            chunk["tool_calls"][0]["function"]["arguments"]
            for chunk in collected
            if "tool_calls" in chunk
            and "arguments" in chunk["tool_calls"][0].get("function", {})
        ]
        args = json.loads("".join(arg_parts))
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"

    def test_streaming_coarse_deltas_complete(
        self, qwen3coder_parser, qwen3coder_request
    ):
        """Single coarse delta with complete tool call → full args emitted."""
        deltas = [
            "<tool_call>\n<function=get_current_weather>"
            "\n<parameter=city>Dallas</parameter>\n</function>"
            "\n</tool_call>",
        ]
        text = ""
        collected = []
        for d in deltas:
            prev = text
            text += d
            r = qwen3coder_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=text,
                delta_text=d,
                request=qwen3coder_request,
            )
            if r:
                collected.append(r)

        tc_chunks = [c for c in collected if "tool_calls" in c]
        assert len(tc_chunks) >= 1
        first_tc = tc_chunks[0]["tool_calls"][0]
        assert first_tc["function"]["name"] == "get_current_weather"
        args = first_tc["function"]["arguments"]
        assert args
        parsed = json.loads(args)
        assert parsed["city"] == "Dallas"


# ═══════════════════════════════════════════════════════════════════════
# Registration tests — verify all new parsers are discoverable
# ═══════════════════════════════════════════════════════════════════════


class TestNewParserRegistration:
    """Verify new parsers are registered and discoverable."""

    @pytest.mark.parametrize(
        "name",
        [
            "gpt-oss",
            "deepseek_v31",
            "deepseek_r1_0528",
            "qwen3_coder_xml",
            "qwen3_xml",
        ],
    )
    def test_parser_registered(self, name):
        """Parser name should be in the registry."""
        cls = ToolParserManager.get_tool_parser(name)
        assert cls is not None

    @pytest.mark.parametrize(
        "name",
        ["deepseek_v31", "qwen3_coder_xml"],
    )
    def test_parser_instantiation(self, name):
        """Parser should instantiate without tokenizer."""
        cls = ToolParserManager.get_tool_parser(name)
        parser = cls(tokenizer=None)
        assert parser is not None

    @pytest.mark.parametrize(
        "name",
        ["deepseek_v31", "qwen3_coder_xml"],
    )
    def test_parser_supports_native_format(self, name):
        """All new parsers should support native tool format."""
        cls = ToolParserManager.get_tool_parser(name)
        assert cls.supports_native_format() is True
