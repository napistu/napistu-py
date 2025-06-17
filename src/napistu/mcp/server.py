# src/napistu/mcp/server.py
"""
Core MCP server implementation for Napistu.
"""

import asyncio
import logging
import os

from mcp.server import FastMCP

from napistu.mcp import codebase
from napistu.mcp import documentation
from napistu.mcp import execution
from napistu.mcp import tutorials
from napistu.mcp import health

from napistu.mcp.profiles import ServerProfile, get_profile

logger = logging.getLogger(__name__)


def _register_component(
    name: str, module, config_key: str, config: dict, mcp: FastMCP, **kwargs
) -> None:
    """
    Register a single component with the MCP server.

    Parameters
    ----------
    name : str
        Component name for logging
    module : module
        Component module with get_component() function or create_component() for execution
    config_key : str
        Configuration key to check if component is enabled
    config : dict
        Server configuration
    mcp : FastMCP
        FastMCP server instance
    **kwargs : dict
        Additional arguments for component creation (used by execution component)
    """
    if not config.get(config_key, False):
        return  # Skip disabled components

    logger.info(f"Registering {name} components")

    if name == "execution":
        # Special handling for execution component which needs session context
        component = module.create_component(
            session_context=kwargs.get("session_context"),
            object_registry=kwargs.get("object_registry"),
        )
    else:
        component = module.get_component()

    component.register(mcp)


def create_server(profile: ServerProfile, **kwargs) -> FastMCP:
    """
    Create an MCP server based on a profile configuration.

    Parameters
    ----------
    profile : ServerProfile
        Server profile to use. All configuration must be set in the profile.
    **kwargs
        Additional arguments to pass to the FastMCP constructor such as host and port.

    Returns
    -------
    FastMCP
        Configured FastMCP server instance.
    """

    config = profile.get_config()

    # Create the server with FastMCP-specific parameters
    mcp = FastMCP(config["server_name"], **kwargs)

    # Define component configurations
    component_configs = [
        ("documentation", documentation, "enable_documentation"),
        ("codebase", codebase, "enable_codebase"),
        ("tutorials", tutorials, "enable_tutorials"),
        ("execution", execution, "enable_execution"),
    ]

    # Register all components
    for name, module, config_key in component_configs:
        _register_component(
            name,
            module,
            config_key,
            config,
            mcp,
            session_context=config.get("session_context"),
            object_registry=config.get("object_registry"),
        )

    # Always register health components
    health.register_components(mcp)
    logger.info("Registered health components")

    return mcp


async def _initialize_component(
    name: str, module, config_key: str, config: dict
) -> bool:
    """
    Initialize a single component with error handling.

    Parameters
    ----------
    name : str
        Component name for logging
    module : module
        Component module with get_component() function
    config_key : str
        Configuration key to check if component is enabled
    config : dict
        Server configuration

    Returns
    -------
    bool
        True if initialization successful
    """
    if not config.get(config_key, False):
        return True  # Skip disabled components

    logger.info(f"Initializing {name} components")
    try:
        component = module.get_component()
        result = await component.safe_initialize()
        return result
    except Exception as e:
        logger.error(f"❌ {name.title()} components failed to initialize: {e}")
        return False


async def initialize_components(profile: ServerProfile) -> None:
    """
    Asynchronously initialize all enabled components for the MCP server.

    Parameters
    ----------
    profile : ServerProfile
        The profile whose configuration determines which components to initialize.

    Returns
    -------
    None
    """
    config = profile.get_config()

    # Define component configurations
    component_configs = [
        ("documentation", documentation, "enable_documentation"),
        ("codebase", codebase, "enable_codebase"),
        ("tutorials", tutorials, "enable_tutorials"),
        ("execution", execution, "enable_execution"),
    ]

    # Initialize all components
    initialization_results = {}

    for name, module, config_key in component_configs:
        result = await _initialize_component(name, module, config_key, config)
        initialization_results[name] = result

    # Initialize health components last since they monitor the other components
    logger.info("Initializing health components")
    try:
        result = await health.initialize_components()
        initialization_results["health"] = result
        if result:
            logger.info("✅ Health components initialized successfully")
        else:
            logger.warning("⚠️ Health components initialized with issues")
    except Exception as e:
        logger.error(f"❌ Health components failed to initialize: {e}")
        initialization_results["health"] = False

    # Summary of initialization
    successful = sum(1 for success in initialization_results.values() if success)
    total = len(initialization_results)
    logger.info(
        f"Component initialization complete: {successful}/{total} components successful"
    )

    if successful == 0:
        logger.error(
            "❌ All components failed to initialize - server may not function correctly"
        )
    elif successful < total:
        logger.warning(
            "⚠️ Some components failed to initialize - server running in degraded mode"
        )


def start_mcp_server(
    profile_name: str = "remote",
    host: str = "0.0.0.0",
    port: int = 8080,
    server_name: str | None = None,
) -> None:
    """
    Start MCP server - main entry point for server startup.

    The server will be started with HTTP transport on the specified host and port.
    Environment variables can override the default configuration:
    - MCP_PROFILE: Server profile to use
    - HOST: Host to bind to
    - PORT: Port to bind to
    - MCP_SERVER_NAME: Name of the MCP server

    Parameters
    ----------
    profile_name : str, optional
        Server profile to use ('local', 'remote', 'full'). Defaults to 'remote'.
    host : str, optional
        Host address to bind the server to. Defaults to '0.0.0.0'.
    port : int, optional
        Port number to listen on. Defaults to 8080.
    server_name : str | None, optional
        Custom name for the MCP server. If None, will be generated from profile name.
        Defaults to None.

    Returns
    -------
    None
        This function runs indefinitely until interrupted.

    Notes
    -----
    The server uses HTTP transport (streamable-http) for all connections.
    Components are initialized asynchronously before the server starts.
    Health components are always registered and initialized last.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("napistu")

    # Get configuration from environment variables (for Cloud Run)
    env_profile = os.getenv("MCP_PROFILE", profile_name)
    env_host = os.getenv("HOST", host)
    env_port = int(os.getenv("PORT", port))
    env_server_name = os.getenv(
        "MCP_SERVER_NAME", server_name or f"napistu-{env_profile}"
    )

    logger.info("Starting Napistu MCP Server")
    logger.info(f"  Profile: {env_profile}")
    logger.info(f"  Host: {env_host}")
    logger.info(f"  Port: {env_port}")
    logger.info(f"  Server Name: {env_server_name}")
    logger.info("  Transport: streamable-http")

    # Create session context for execution components
    session_context = {}
    object_registry = {}

    # Get profile with configuration
    profile: ServerProfile = get_profile(
        env_profile,
        session_context=session_context,
        object_registry=object_registry,
        server_name=env_server_name,
    )

    # Create server with Cloud Run proxy settings
    mcp = create_server(profile, host=env_host, port=env_port)

    # Initialize components first (separate async call)
    async def init_components():
        logger.info("Initializing MCP components...")
        await initialize_components(profile)
        logger.info("✅ Component initialization complete")

    # Run initialization
    asyncio.run(init_components())

    # Debug info
    logger.info(f"Server settings: {mcp.settings}")

    logger.info("🚀 Starting MCP server...")
    logger.info(f"Using HTTP transport on http://{env_host}:{env_port}")

    mcp.run(transport="streamable-http")
