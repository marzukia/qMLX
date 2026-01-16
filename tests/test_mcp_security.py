# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP security module.

These tests verify that the MCP command validation properly prevents
command injection attacks and other security vulnerabilities.
"""

import pytest
from vllm_mlx.mcp.security import (
    MCPCommandValidator,
    MCPSecurityError,
    ALLOWED_COMMANDS,
    validate_mcp_server_config,
)
from vllm_mlx.mcp.types import MCPServerConfig, MCPTransport


class TestMCPCommandValidator:
    """Tests for MCPCommandValidator class."""

    def test_allowed_command_passes(self):
        """Test that allowed commands pass validation."""
        # Use check_path_exists=False for unit tests (we're testing whitelist logic)
        validator = MCPCommandValidator(check_path_exists=False)

        # These should not raise
        validator.validate_command("npx", "test-server")
        validator.validate_command("uvx", "test-server")
        validator.validate_command("python", "test-server")
        validator.validate_command("node", "test-server")

    def test_disallowed_command_fails(self):
        """Test that disallowed commands are rejected."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("bash", "test-server")

        assert "not in the allowed commands whitelist" in str(exc_info.value)

    def test_command_injection_semicolon_blocked(self):
        """Test that command injection via semicolon is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx; rm -rf /", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_pipe_blocked(self):
        """Test that command injection via pipe is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx | cat /etc/passwd", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_and_blocked(self):
        """Test that command injection via && is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx && rm -rf /", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_backtick_blocked(self):
        """Test that command injection via backticks is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx `whoami`", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_dollar_paren_blocked(self):
        """Test that command injection via $() is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx $(whoami)", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_path_traversal_blocked(self):
        """Test that path traversal is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("../../../bin/bash", "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestArgumentValidation:
    """Tests for argument validation."""

    def test_safe_args_pass(self):
        """Test that safe arguments pass validation."""
        validator = MCPCommandValidator()

        # These should not raise
        validator.validate_args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "test")
        validator.validate_args(["--db-path", "data.db"], "test")
        validator.validate_args(["--port", "8080"], "test")

    def test_injection_in_args_blocked(self):
        """Test that command injection in arguments is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_args(["-y", "; rm -rf /"], "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_backtick_in_args_blocked(self):
        """Test that backticks in arguments are blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_args(["--path", "`cat /etc/passwd`"], "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_dollar_expansion_in_args_blocked(self):
        """Test that $() in arguments is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_args(["--cmd", "$(whoami)"], "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestEnvironmentValidation:
    """Tests for environment variable validation."""

    def test_safe_env_passes(self):
        """Test that safe environment variables pass."""
        validator = MCPCommandValidator()

        # These should not raise
        validator.validate_env({"API_KEY": "secret123", "DEBUG": "true"}, "test")

    def test_ld_preload_blocked(self):
        """Test that LD_PRELOAD is blocked (library injection)."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"LD_PRELOAD": "/tmp/malicious.so"}, "test-server")

        assert "not allowed for security reasons" in str(exc_info.value)

    def test_path_modification_blocked(self):
        """Test that PATH modification is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"PATH": "/tmp/fake:/usr/bin"}, "test-server")

        assert "not allowed for security reasons" in str(exc_info.value)

    def test_pythonpath_blocked(self):
        """Test that PYTHONPATH is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"PYTHONPATH": "/tmp/malicious"}, "test-server")

        assert "not allowed for security reasons" in str(exc_info.value)

    def test_injection_in_env_value_blocked(self):
        """Test that injection in env values is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"SAFE_VAR": "value; rm -rf /"}, "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestURLValidation:
    """Tests for SSE URL validation."""

    def test_https_url_passes(self):
        """Test that HTTPS URLs pass validation."""
        validator = MCPCommandValidator()

        # These should not raise
        validator.validate_url("https://example.com/sse", "test")
        validator.validate_url("https://api.service.com:8443/mcp", "test")

    def test_localhost_http_passes(self):
        """Test that localhost HTTP URLs pass (for development)."""
        validator = MCPCommandValidator()

        # These should not raise (but may warn)
        validator.validate_url("http://localhost:3000/sse", "test")
        validator.validate_url("http://localhost/mcp", "test")

    def test_non_http_scheme_blocked(self):
        """Test that non-HTTP schemes are blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_url("file:///etc/passwd", "test-server")

        assert "must use http:// or https://" in str(exc_info.value)

    def test_ftp_scheme_blocked(self):
        """Test that FTP scheme is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_url("ftp://malicious.com/file", "test-server")

        assert "must use http:// or https://" in str(exc_info.value)

    def test_injection_in_url_blocked(self):
        """Test that injection in URL is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_url("https://example.com/sse; rm -rf /", "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestUnsafeMode:
    """Tests for unsafe mode (development only)."""

    def test_unsafe_mode_allows_any_command(self):
        """Test that unsafe mode allows any command (with warning)."""
        validator = MCPCommandValidator(allow_unsafe=True)

        # These should not raise even though they're dangerous
        validator.validate_command("bash", "test")
        validator.validate_command("/bin/sh -c 'dangerous'", "test")

    def test_unsafe_mode_allows_any_args(self):
        """Test that unsafe mode allows any arguments."""
        validator = MCPCommandValidator(allow_unsafe=True)

        # These should not raise
        validator.validate_args(["; rm -rf /"], "test")


class TestCustomWhitelist:
    """Tests for custom command whitelist."""

    def test_custom_whitelist_extends_default(self):
        """Test that custom whitelist extends the default."""
        validator = MCPCommandValidator(
            custom_whitelist={"my-custom-mcp-server"},
            check_path_exists=False,
        )

        # Default commands should still work
        validator.validate_command("npx", "test")

        # Custom command should now work
        validator.validate_command("my-custom-mcp-server", "test")

    def test_custom_whitelist_only(self):
        """Test using only custom whitelist."""
        validator = MCPCommandValidator(
            allowed_commands={"custom-only"},
            check_path_exists=False,
        )

        # Custom command should work
        validator.validate_command("custom-only", "test")

        # Default commands should now fail
        with pytest.raises(MCPSecurityError):
            validator.validate_command("npx", "test")


class TestMCPServerConfigSecurity:
    """Tests for security validation in MCPServerConfig."""

    def test_valid_stdio_config(self):
        """Test that valid stdio config passes security validation."""
        # This should not raise
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        assert config.command == "npx"

    def test_invalid_command_rejected(self):
        """Test that invalid commands are rejected."""
        with pytest.raises(ValueError) as exc_info:
            MCPServerConfig(
                name="malicious-server",
                transport=MCPTransport.STDIO,
                command="bash",
                args=["-c", "rm -rf /"],
            )

        assert "not in the allowed commands whitelist" in str(exc_info.value)

    def test_command_injection_in_config_rejected(self):
        """Test that command injection in config is rejected."""
        with pytest.raises(ValueError) as exc_info:
            MCPServerConfig(
                name="injection-server",
                transport=MCPTransport.STDIO,
                command="npx; rm -rf /",
            )

        assert "dangerous pattern" in str(exc_info.value)

    def test_valid_sse_config(self):
        """Test that valid SSE config passes validation."""
        config = MCPServerConfig(
            name="sse-server",
            transport=MCPTransport.SSE,
            url="https://api.example.com/mcp",
        )
        assert config.url == "https://api.example.com/mcp"

    def test_skip_security_validation(self):
        """Test that skip_security_validation allows any command (with warning)."""
        # This should not raise even with dangerous command
        config = MCPServerConfig(
            name="unsafe-server",
            transport=MCPTransport.STDIO,
            command="bash",
            args=["-c", "echo hello"],
            skip_security_validation=True,
        )
        assert config.command == "bash"


class TestDefaultWhitelist:
    """Tests for the default command whitelist."""

    def test_default_whitelist_contains_expected_commands(self):
        """Test that the default whitelist contains expected safe commands."""
        assert "npx" in ALLOWED_COMMANDS
        assert "uvx" in ALLOWED_COMMANDS
        assert "python" in ALLOWED_COMMANDS
        assert "python3" in ALLOWED_COMMANDS
        assert "node" in ALLOWED_COMMANDS
        assert "docker" in ALLOWED_COMMANDS

    def test_default_whitelist_excludes_dangerous_commands(self):
        """Test that dangerous commands are not in default whitelist."""
        assert "bash" not in ALLOWED_COMMANDS
        assert "sh" not in ALLOWED_COMMANDS
        assert "zsh" not in ALLOWED_COMMANDS
        assert "rm" not in ALLOWED_COMMANDS
        assert "curl" not in ALLOWED_COMMANDS
        assert "wget" not in ALLOWED_COMMANDS
