from pydantic import BaseModel, Field, computed_field, validator
from typing import Optional
import click
from napistu.mcp.constants import MCP_DEFAULTS, MCP_PRODUCTION_URL


class MCPServerConfig(BaseModel):
    """Server-side MCP configuration with validation."""

    host: str = Field(description="Host to bind server to")
    port: int = Field(ge=1, le=65535, description="Port to bind server to")
    server_name: str = Field(description="Server name")

    @validator("host")
    def validate_host(cls, v):
        """Basic host validation."""
        if not v or v.isspace():
            raise ValueError("Host cannot be empty")
        return v.strip()

    @computed_field
    @property
    def bind_address(self) -> str:
        """Full address for server binding."""
        return f"{self.host}:{self.port}"


class MCPClientConfig(BaseModel):
    """Client-side MCP configuration with validation."""

    host: str = Field(description="Target server host")
    port: int = Field(ge=1, le=65535, description="Target server port")
    use_https: bool = Field(description="Use HTTPS instead of HTTP")

    @computed_field
    @property
    def base_url(self) -> str:
        """Base URL for HTTP requests."""
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @computed_field
    @property
    def mcp_url(self) -> str:
        """Full MCP endpoint URL."""
        return f"{self.base_url}{MCP_DEFAULTS.MCP_PATH}"


# Preset configurations - no overrides needed
def local_server_config() -> MCPServerConfig:
    """Local development server configuration."""
    return MCPServerConfig(
        host=MCP_DEFAULTS.LOCAL_HOST,
        port=MCP_DEFAULTS.LOCAL_PORT,
        server_name=MCP_DEFAULTS.LOCAL_SERVER_NAME,
    )


def production_server_config() -> MCPServerConfig:
    """Production server configuration."""
    return MCPServerConfig(
        host=MCP_DEFAULTS.PRODUCTION_HOST,
        port=MCP_DEFAULTS.PRODUCTION_PORT,
        server_name=MCP_DEFAULTS.PRODUCTION_SERVER_NAME,
    )


def local_client_config() -> MCPClientConfig:
    """Local development client configuration."""
    return MCPClientConfig(
        host=MCP_DEFAULTS.LOCAL_HOST, port=MCP_DEFAULTS.LOCAL_PORT, use_https=False
    )


def production_client_config() -> MCPClientConfig:
    """Production client configuration."""
    # Parse production URL to extract host
    production_host = MCP_PRODUCTION_URL.replace("https://", "").split("/")[0]
    return MCPClientConfig(
        host=production_host, port=MCP_DEFAULTS.HTTPS_PORT, use_https=True
    )


# Utility functions for CLI validation
def validate_server_config_flags(
    local: bool,
    production: bool,
    host: Optional[str],
    port: Optional[int],
    server_name: Optional[str],
) -> MCPServerConfig:
    """
    Validate server configuration flags and return appropriate config.

    Raises click.BadParameter if incompatible flags are used.
    """
    # Check for mutually exclusive presets
    if local and production:
        raise click.BadParameter("Cannot use both --local and --production flags")

    # Check for preset + manual config conflicts
    preset_used = local or production
    manual_config_used = any(
        [host is not None, port is not None, server_name is not None]
    )

    if preset_used and manual_config_used:
        preset_name = "local" if local else "production"
        raise click.BadParameter(
            f"Cannot use --{preset_name} with manual host/port/server-name configuration. "
            f"Use either preset flags OR manual configuration, not both."
        )

    # Return appropriate config
    if local:
        return local_server_config()
    elif production:
        return production_server_config()
    else:
        # Manual configuration - all required
        if not all([host, port, server_name]):
            raise click.BadParameter(
                "Manual configuration requires --host, --port, and --server-name"
            )
        return MCPServerConfig(host=host, port=port, server_name=server_name)


def validate_client_config_flags(
    local: bool,
    production: bool,
    host: Optional[str],
    port: Optional[int],
    use_https: Optional[bool],
) -> MCPClientConfig:
    """
    Validate client configuration flags and return appropriate config.

    Raises click.BadParameter if incompatible flags are used.
    """
    # Check for mutually exclusive presets
    if local and production:
        raise click.BadParameter("Cannot use both --local and --production flags")

    # Check for preset + manual config conflicts
    preset_used = local or production
    manual_config_used = any(
        [host is not None, port is not None, use_https is not None]
    )

    if preset_used and manual_config_used:
        preset_name = "local" if local else "production"
        raise click.BadParameter(
            f"Cannot use --{preset_name} with manual host/port/https configuration. "
            f"Use either preset flags OR manual configuration, not both."
        )

    # Return appropriate config
    if local:
        return local_client_config()
    elif production:
        return production_client_config()
    else:
        # Manual configuration - all required
        if not all([host is not None, port is not None, use_https is not None]):
            raise click.BadParameter(
                "Manual configuration requires --host, --port, and --https"
            )
        return MCPClientConfig(host=host, port=port, use_https=use_https)


# Click option decorators for reuse
def server_config_options(f):
    """Decorator to add server configuration options to click commands."""
    f = click.option(
        "--production", is_flag=True, help="Use production server configuration"
    )(f)
    f = click.option(
        "--local", is_flag=True, help="Use local development server configuration"
    )(f)
    f = click.option(
        "--host",
        type=str,
        help="Host to bind server to (requires --port and --server-name)",
    )(f)
    f = click.option(
        "--port",
        type=int,
        help="Port to bind server to (requires --host and --server-name)",
    )(f)
    f = click.option(
        "--server-name", type=str, help="Server name (requires --host and --port)"
    )(f)
    return f


def client_config_options(f):
    """Decorator to add client configuration options to click commands."""
    f = click.option("--production", is_flag=True, help="Connect to production server")(
        f
    )
    f = click.option(
        "--local", is_flag=True, help="Connect to local development server"
    )(f)
    f = click.option(
        "--host",
        type=str,
        default=None,
        help="Server host (requires --port and --https)",
    )(f)
    f = click.option(
        "--port",
        type=int,
        default=None,
        help="Server port (requires --host and --https)",
    )(f)
    f = click.option(
        "--https",
        is_flag=True,
        default=None,
        flag_value=True,
        help="Use HTTPS (requires --host and --port)",
    )(f)
    return f
