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
    # Pass all kwargs directly to the FastMCP constructor
    mcp = FastMCP(config["server_name"], **kwargs)

    # Always register health endpoint for deployment monitoring
    health.register_health_endpoint(mcp)
    logger.info("Registered health endpoint")

    if config["enable_documentation"]:
        logger.info("Registering documentation components")
        documentation.register_components(mcp)
    if config["enable_codebase"]:
        logger.info("Registering codebase components")
        codebase.register_components(mcp)
    if config["enable_execution"]:
        logger.info("Registering execution components")
        execution.register_components(
            mcp,
            session_context=config["session_context"],
            object_registry=config["object_registry"],
        )
    if config["enable_tutorials"]:
        logger.info("Registering tutorials components")
        tutorials.register_components(mcp)
    return mcp


async def initialize_components(profile: ServerProfile) -> None:
    """
    Asynchronously initialize all enabled components for the MCP server, using the provided ServerProfile.

    Parameters
    ----------
    profile : ServerProfile
        The profile whose configuration determines which components to initialize.

    Returns
    -------
    None
    """
    config = profile.get_config()
    if config["enable_documentation"]:
        logger.info("Initializing documentation components")
        await documentation.initialize_components()
    if config["enable_codebase"]:
        logger.info("Initializing codebase components")
        await codebase.initialize_components()
    if config["enable_tutorials"]:
        logger.info("Initializing tutorials components")
        await tutorials.initialize_components()
    if config["enable_execution"]:
        logger.info("Initializing execution components")
        await execution.initialize_components()


def start_mcp_server(profile_name="remote", host="0.0.0.0", port=8080, server_name=None, transport=None):
    """
    Start MCP server - main entry point for server startup.
    
    Args:
        profile_name: Server profile ('local', 'remote', 'full')
        host: Host to bind to
        port: Port to bind to  
        server_name: Name of the MCP server
        transport: Transport type ('stdio', 'http')
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("napistu")

    # Get configuration from environment variables (for Cloud Run)
    env_profile = os.getenv("MCP_PROFILE", profile_name)
    env_host = os.getenv("HOST", host)
    env_port = int(os.getenv("PORT", port))
    env_server_name = os.getenv("MCP_SERVER_NAME", server_name or f"napistu-{env_profile}")
    env_transport = transport or ("http" if os.getenv("PORT") else "stdio")

    logger.info(f"Starting Napistu MCP Server")
    logger.info(f"  Profile: {env_profile}")
    logger.info(f"  Host: {env_host}")
    logger.info(f"  Port: {env_port}")
    logger.info(f"  Server Name: {env_server_name}")
    logger.info(f"  Transport: {env_transport}")

    # Create session context for execution components
    session_context = {}
    object_registry = {}

    # Get profile with configuration
    profile: ServerProfile = get_profile(env_profile, 
                                        session_context=session_context,
                                        object_registry=object_registry,
                                        server_name=env_server_name)

    # Create server
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
    
    # Start server synchronously (no nested async)
    if env_transport == "http":
        logger.info(f"Using HTTP transport on http://{env_host}:{env_port}")
        mcp.run(transport="streamable-http")
    else:
        logger.info("Using stdio transport")
        mcp.run(transport="stdio")