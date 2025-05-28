"""
Core MCP server implementation for Napistu.
"""

import logging

from mcp.server import FastMCP

from napistu.mcp import codebase
from napistu.mcp import documentation
from napistu.mcp import execution
from napistu.mcp import tutorials

from napistu.mcp.profiles import ServerProfile

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
