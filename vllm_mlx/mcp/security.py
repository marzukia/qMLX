# SPDX-License-Identifier: Apache-2.0
"""
MCP security module for command validation and sandboxing.

This module provides security controls to prevent command injection
and other attacks via MCP server configurations.
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Whitelist of allowed MCP server commands
# These are well-known, trusted MCP server executables
ALLOWED_COMMANDS: Set[str] = {
    # Node.js package runners (for official MCP servers)
    "npx",
    "npm",
    "node",
    # Python package runners
    "uvx",
    "uv",
    "python",
    "python3",
    "pip",
    "pipx",
    # Official MCP servers (when installed globally)
    "mcp-server-filesystem",
    "mcp-server-sqlite",
    "mcp-server-postgres",
    "mcp-server-github",
    "mcp-server-slack",
    "mcp-server-memory",
    "mcp-server-puppeteer",
    "mcp-server-brave-search",
    "mcp-server-google-maps",
    "mcp-server-fetch",
    # Docker (for containerized MCP servers)
    "docker",
}

# Patterns that indicate dangerous commands
DANGEROUS_PATTERNS: List[re.Pattern] = [
    re.compile(r";\s*"),  # Command chaining with ;
    re.compile(r"\|\s*"),  # Piping
    re.compile(r"&&\s*"),  # Command chaining with &&
    re.compile(r"\|\|\s*"),  # Command chaining with ||
    re.compile(r"`"),  # Backtick command substitution
    re.compile(r"\$\("),  # $() command substitution
    re.compile(r">\s*"),  # Output redirection
    re.compile(r"<\s*"),  # Input redirection
    re.compile(r"\.\./"),  # Path traversal
    re.compile(r"~"),  # Home directory expansion (can be abused)
]

# Dangerous argument patterns
DANGEROUS_ARG_PATTERNS: List[re.Pattern] = [
    re.compile(r";\s*"),
    re.compile(r"\|\s*"),
    re.compile(r"&&\s*"),
    re.compile(r"\|\|\s*"),
    re.compile(r"`"),
    re.compile(r"\$\("),
    re.compile(r"\$\{"),
    re.compile(r">\s*/"),  # Redirect to absolute path
    re.compile(r"<\s*/"),  # Read from absolute path
]


class MCPSecurityError(Exception):
    """Raised when MCP security validation fails."""

    pass


class MCPCommandValidator:
    """
    Validates MCP server commands for security.

    This class provides methods to validate commands and arguments
    before they are executed, preventing command injection attacks.
    """

    def __init__(
        self,
        allowed_commands: Optional[Set[str]] = None,
        allow_unsafe: bool = False,
        custom_whitelist: Optional[Set[str]] = None,
        check_path_exists: bool = True,
    ):
        """
        Initialize the command validator.

        Args:
            allowed_commands: Set of allowed command names. If None, uses default whitelist.
            allow_unsafe: If True, allows any command (for development only).
                         WARNING: This disables security checks!
            custom_whitelist: Additional commands to allow beyond the default whitelist.
            check_path_exists: If True, verify command exists in PATH. Set to False for testing.
        """
        self.allow_unsafe = allow_unsafe
        self.allowed_commands = allowed_commands or ALLOWED_COMMANDS.copy()
        self.check_path_exists = check_path_exists

        if custom_whitelist:
            self.allowed_commands.update(custom_whitelist)

        if allow_unsafe:
            logger.warning(
                "MCP SECURITY WARNING: Unsafe mode enabled. "
                "All commands will be allowed without validation. "
                "This should NEVER be used in production!"
            )

    def validate_command(self, command: str, server_name: str) -> None:
        """
        Validate that a command is safe to execute.

        Args:
            command: The command to validate
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If the command is not allowed
        """
        if self.allow_unsafe:
            logger.warning(
                f"MCP security bypassed for server '{server_name}': "
                f"allowing command '{command}' (unsafe mode)"
            )
            return

        # Check for dangerous patterns in command
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(command):
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Command contains dangerous pattern: "
                    f"'{command}'. Command injection attempt blocked."
                )

        # Extract base command name (without path)
        base_command = Path(command).name

        # Check if command is in whitelist
        if base_command not in self.allowed_commands:
            # Check if it's an absolute path to an allowed command
            if os.path.isabs(command):
                resolved_name = Path(command).name
                if resolved_name in self.allowed_commands:
                    # Verify the path actually exists and is executable
                    if os.path.isfile(command) and os.access(command, os.X_OK):
                        logger.info(
                            f"MCP server '{server_name}': Allowing absolute path "
                            f"to whitelisted command: {command}"
                        )
                        return

            raise MCPSecurityError(
                f"MCP server '{server_name}': Command '{base_command}' is not in the "
                f"allowed commands whitelist. Allowed commands: {sorted(self.allowed_commands)}"
            )

        # Verify command exists in PATH (for non-absolute paths)
        if self.check_path_exists and not os.path.isabs(command):
            resolved_path = shutil.which(command)
            if resolved_path is None:
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Command '{command}' not found in PATH. "
                    f"Ensure the command is installed and accessible."
                )

        logger.debug(
            f"MCP server '{server_name}': Command '{command}' validated successfully"
        )

    def validate_args(self, args: List[str], server_name: str) -> None:
        """
        Validate command arguments for dangerous patterns.

        Args:
            args: List of command arguments
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If any argument contains dangerous patterns
        """
        if self.allow_unsafe:
            return

        for i, arg in enumerate(args):
            for pattern in DANGEROUS_ARG_PATTERNS:
                if pattern.search(arg):
                    raise MCPSecurityError(
                        f"MCP server '{server_name}': Argument {i} contains dangerous "
                        f"pattern: '{arg}'. Potential command injection blocked."
                    )

        logger.debug(
            f"MCP server '{server_name}': {len(args)} arguments validated successfully"
        )

    def validate_env(self, env: Optional[Dict[str, str]], server_name: str) -> None:
        """
        Validate environment variables for dangerous values.

        Args:
            env: Dictionary of environment variables
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If any env var contains dangerous patterns
        """
        if self.allow_unsafe or not env:
            return

        # Dangerous environment variables that could affect execution
        dangerous_env_vars = {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "PATH",  # Modifying PATH could redirect commands
            "PYTHONPATH",
            "NODE_PATH",
        }

        for key, value in env.items():
            # Check for dangerous env var names
            if key.upper() in dangerous_env_vars:
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Setting '{key}' environment variable "
                    f"is not allowed for security reasons."
                )

            # Check for dangerous patterns in values
            for pattern in DANGEROUS_ARG_PATTERNS:
                if pattern.search(value):
                    raise MCPSecurityError(
                        f"MCP server '{server_name}': Environment variable '{key}' "
                        f"contains dangerous pattern. Potential injection blocked."
                    )

        logger.debug(
            f"MCP server '{server_name}': {len(env)} environment variables validated"
        )

    def validate_url(self, url: str, server_name: str) -> None:
        """
        Validate SSE URL for security.

        Args:
            url: The SSE URL to validate
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If the URL is not safe
        """
        if self.allow_unsafe:
            return

        # Must be http or https
        if not url.startswith(("http://", "https://")):
            raise MCPSecurityError(
                f"MCP server '{server_name}': URL must use http:// or https:// scheme. "
                f"Got: {url}"
            )

        # Warn about non-HTTPS URLs
        if url.startswith("http://") and not url.startswith("http://localhost"):
            logger.warning(
                f"MCP server '{server_name}': Using insecure HTTP connection to {url}. "
                f"Consider using HTTPS for production environments."
            )

        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(url):
                raise MCPSecurityError(
                    f"MCP server '{server_name}': URL contains dangerous pattern: {url}"
                )

        logger.debug(f"MCP server '{server_name}': URL '{url}' validated successfully")


# Global validator instance (can be reconfigured)
_validator: Optional[MCPCommandValidator] = None


def get_validator() -> MCPCommandValidator:
    """Get the global command validator instance."""
    global _validator
    if _validator is None:
        _validator = MCPCommandValidator()
    return _validator


def set_validator(validator: MCPCommandValidator) -> None:
    """Set a custom global validator."""
    global _validator
    _validator = validator


def validate_mcp_server_config(
    server_name: str,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    url: Optional[str] = None,
) -> None:
    """
    Validate MCP server configuration for security.

    This is a convenience function that uses the global validator.

    Args:
        server_name: Name of the MCP server
        command: Command to execute (for stdio transport)
        args: Command arguments
        env: Environment variables
        url: SSE URL (for sse transport)

    Raises:
        MCPSecurityError: If validation fails
    """
    validator = get_validator()

    if command:
        validator.validate_command(command, server_name)

    if args:
        validator.validate_args(args, server_name)

    if env:
        validator.validate_env(env, server_name)

    if url:
        validator.validate_url(url, server_name)
