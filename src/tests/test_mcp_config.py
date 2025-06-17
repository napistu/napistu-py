"""
Tests for MCP configuration validation, presets, and Pydantic models.
"""

import pytest
from pydantic import ValidationError
import click

from napistu.mcp.config import (
    MCPServerConfig,
    MCPClientConfig,
    local_server_config,
    production_server_config,
    local_client_config,
    production_client_config,
    validate_server_config_flags,
    validate_client_config_flags,
)
from napistu.mcp.constants import MCP_DEFAULTS


def test_configuration_validation():
    """Test validation logic for preset conflicts and manual configuration requirements."""

    # Test conflicting presets
    with pytest.raises(
        click.BadParameter, match="Cannot use both --local and --production"
    ):
        validate_server_config_flags(
            local=True, production=True, host=None, port=None, server_name=None
        )

    with pytest.raises(
        click.BadParameter, match="Cannot use both --local and --production"
    ):
        validate_client_config_flags(
            local=True, production=True, host=None, port=None, use_https=None
        )

    # Test preset + manual config conflicts
    with pytest.raises(click.BadParameter, match="Cannot use --local with manual"):
        validate_server_config_flags(
            local=True,
            production=False,
            host="192.168.1.1",
            port=None,
            server_name=None,
        )

    with pytest.raises(click.BadParameter, match="Cannot use --production with manual"):
        validate_client_config_flags(
            local=False, production=True, host="localhost", port=None, use_https=None
        )

    # Test incomplete manual configuration
    with pytest.raises(
        click.BadParameter,
        match="Manual configuration requires --host, --port, and --server-name",
    ):
        validate_server_config_flags(
            local=False, production=False, host="localhost", port=None, server_name=None
        )

    with pytest.raises(
        click.BadParameter,
        match="Manual configuration requires --host, --port, and --https",
    ):
        validate_client_config_flags(
            local=False, production=False, host="localhost", port=8080, use_https=None
        )

    # Test valid configurations don't raise errors
    config = validate_server_config_flags(
        local=True, production=False, host=None, port=None, server_name=None
    )
    assert isinstance(config, MCPServerConfig)

    config = validate_client_config_flags(
        local=False, production=True, host=None, port=None, use_https=None
    )
    assert isinstance(config, MCPClientConfig)

    # Test valid manual configuration
    config = validate_server_config_flags(
        local=False,
        production=False,
        host="localhost",
        port=9000,
        server_name="test-server",
    )
    assert config.host == "localhost"
    assert config.port == 9000
    assert config.server_name == "test-server"


def test_preset_configurations():
    """Test that preset functions return correct and consistent configurations."""

    # Test local server preset
    local_server = local_server_config()
    assert local_server.host == MCP_DEFAULTS.LOCAL_HOST
    assert local_server.port == MCP_DEFAULTS.LOCAL_PORT
    assert local_server.server_name == MCP_DEFAULTS.LOCAL_SERVER_NAME
    assert (
        local_server.bind_address
        == f"{MCP_DEFAULTS.LOCAL_HOST}:{MCP_DEFAULTS.LOCAL_PORT}"
    )

    # Test production server preset
    production_server = production_server_config()
    assert production_server.host == MCP_DEFAULTS.PRODUCTION_HOST
    assert production_server.port == MCP_DEFAULTS.PRODUCTION_PORT
    assert production_server.server_name == MCP_DEFAULTS.PRODUCTION_SERVER_NAME

    # Test local client preset
    local_client = local_client_config()
    assert local_client.host == MCP_DEFAULTS.LOCAL_HOST
    assert local_client.port == MCP_DEFAULTS.LOCAL_PORT
    assert not local_client.use_https
    assert (
        local_client.base_url
        == f"http://{MCP_DEFAULTS.LOCAL_HOST}:{MCP_DEFAULTS.LOCAL_PORT}"
    )
    assert (
        local_client.mcp_url
        == f"http://{MCP_DEFAULTS.LOCAL_HOST}:{MCP_DEFAULTS.LOCAL_PORT}{MCP_DEFAULTS.MCP_PATH}"
    )

    # Test production client preset
    production_client = production_client_config()
    assert production_client.use_https
    assert production_client.port == MCP_DEFAULTS.HTTPS_PORT
    assert (
        "napistu-mcp-server" in production_client.host
    )  # Extracted from production URL
    assert production_client.base_url.startswith("https://")
    assert production_client.mcp_url.endswith(MCP_DEFAULTS.MCP_PATH)

    # Test that presets are deterministic (same result on multiple calls)
    assert local_server_config().host == local_server_config().host
    assert production_client_config().port == production_client_config().port


def test_pydantic_model_validation():
    """Test Pydantic model validation for ports, hosts, and computed properties."""

    # Test valid configurations
    valid_server = MCPServerConfig(host="localhost", port=8080, server_name="test")
    assert valid_server.host == "localhost"
    assert valid_server.port == 8080
    assert valid_server.server_name == "test"
    assert valid_server.bind_address == "localhost:8080"

    valid_client = MCPClientConfig(host="example.com", port=443, use_https=True)
    assert valid_client.host == "example.com"
    assert valid_client.base_url == "https://example.com:443"
    assert valid_client.mcp_url == f"https://example.com:443{MCP_DEFAULTS.MCP_PATH}"

    # Test invalid port validation
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 1"
    ):
        MCPServerConfig(host="localhost", port=0, server_name="test")

    with pytest.raises(
        ValidationError, match="Input should be less than or equal to 65535"
    ):
        MCPServerConfig(host="localhost", port=70000, server_name="test")

    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 1"
    ):
        MCPClientConfig(host="localhost", port=-1, use_https=False)

    # Test empty/whitespace host validation
    with pytest.raises(ValidationError, match="Host cannot be empty"):
        MCPServerConfig(host="", port=8080, server_name="test")

    with pytest.raises(ValidationError, match="Host cannot be empty"):
        MCPServerConfig(host="   ", port=8080, server_name="test")

    # Test host whitespace trimming
    server_with_spaces = MCPServerConfig(
        host="  localhost  ", port=8080, server_name="test"
    )
    assert server_with_spaces.host == "localhost"  # Should be trimmed

    # Test URL generation with different protocols
    http_client = MCPClientConfig(host="localhost", port=8080, use_https=False)
    assert http_client.base_url == "http://localhost:8080"

    https_client = MCPClientConfig(host="secure.com", port=443, use_https=True)
    assert https_client.base_url == "https://secure.com:443"
