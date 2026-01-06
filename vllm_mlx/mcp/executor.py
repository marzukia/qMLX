# SPDX-License-Identifier: Apache-2.0
"""
Tool executor for handling tool calls from model responses.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from .manager import MCPClientManager
from .tools import extract_tool_calls, format_tool_result
from .types import MCPToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Handles execution of tool calls from model responses.

    Provides utilities for:
    - Extracting tool calls from responses
    - Executing multiple tool calls (parallel or sequential)
    - Formatting results for conversation
    """

    def __init__(
        self,
        manager: MCPClientManager,
        max_parallel: int = 5,
        default_timeout: Optional[float] = None,
    ):
        """
        Initialize tool executor.

        Args:
            manager: MCP client manager
            max_parallel: Maximum parallel tool executions
            default_timeout: Default timeout for tool calls
        """
        self.manager = manager
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout or manager.config.default_timeout

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[Tuple[MCPToolResult, str]]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of OpenAI tool call objects
            parallel: Execute in parallel (True) or sequential (False)

        Returns:
            List of (MCPToolResult, tool_call_id) tuples
        """
        if not tool_calls:
            return []

        if parallel:
            return await self._execute_parallel(tool_calls)
        else:
            return await self._execute_sequential(tool_calls)

    async def _execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Tuple[MCPToolResult, str]]:
        """Execute tool calls in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(tool_call: Dict[str, Any]):
            async with semaphore:
                result = await self.manager.execute_tool_call(
                    tool_call,
                    timeout=self.default_timeout,
                )
                call_id = tool_call.get("id", "")
                return (result, call_id)

        tasks = [execute_with_semaphore(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            call_id = tool_calls[i].get("id", "")
            if isinstance(result, Exception):
                processed.append((
                    MCPToolResult(
                        tool_name=tool_calls[i].get("function", {}).get("name", ""),
                        content=None,
                        is_error=True,
                        error_message=str(result),
                    ),
                    call_id,
                ))
            else:
                processed.append(result)

        return processed

    async def _execute_sequential(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Tuple[MCPToolResult, str]]:
        """Execute tool calls sequentially."""
        results = []
        for tool_call in tool_calls:
            try:
                result = await self.manager.execute_tool_call(
                    tool_call,
                    timeout=self.default_timeout,
                )
                call_id = tool_call.get("id", "")
                results.append((result, call_id))
            except Exception as e:
                call_id = tool_call.get("id", "")
                results.append((
                    MCPToolResult(
                        tool_name=tool_call.get("function", {}).get("name", ""),
                        content=None,
                        is_error=True,
                        error_message=str(e),
                    ),
                    call_id,
                ))
        return results

    async def execute_and_format(
        self,
        tool_calls: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls and format results as messages.

        Args:
            tool_calls: List of OpenAI tool call objects
            parallel: Execute in parallel

        Returns:
            List of tool result messages ready for conversation
        """
        results = await self.execute_tool_calls(tool_calls, parallel)
        return [format_tool_result(result, call_id) for result, call_id in results]

    def extract_and_validate(
        self,
        response: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Extract tool calls from response and validate them.

        Args:
            response: Model response in OpenAI format

        Returns:
            Tuple of (tool_calls, all_valid)
        """
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            return [], True

        # Validate each tool call
        all_valid = True
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")

            # Check if tool exists
            if not self._tool_exists(name):
                logger.warning(f"Tool '{name}' not found in any MCP server")
                all_valid = False

        return tool_calls, all_valid

    def _tool_exists(self, full_name: str) -> bool:
        """Check if a tool exists in any connected server."""
        # Check by full name
        for tool in self.manager.get_all_tools():
            if tool.full_name == full_name:
                return True

        # Check by just tool name (without server prefix)
        if "__" not in full_name:
            for tool in self.manager.get_all_tools():
                if tool.name == full_name:
                    return True

        return False


async def execute_single_tool(
    manager: MCPClientManager,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: Optional[float] = None,
) -> MCPToolResult:
    """
    Convenience function to execute a single tool.

    Args:
        manager: MCP client manager
        tool_name: Full tool name (server__tool)
        arguments: Tool arguments
        timeout: Optional timeout

    Returns:
        MCPToolResult
    """
    return await manager.execute_tool(tool_name, arguments, timeout)
