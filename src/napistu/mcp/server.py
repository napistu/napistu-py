"""
Core MCP server implementation for Napistu.
"""

from typing import Dict, List, Any, Optional
from mcp.server import FastMCP
import asyncio

from napistu.mcp import codebase
from napistu.mcp import documentation
from napistu.mcp import execution

from napistu.mcp.profiles import ServerProfile

def create_server(profile: Optional[ServerProfile] = None, **kwargs) -> FastMCP:
    """
    Create an MCP server based on a profile or configuration.
    
    Args:
        profile: Server profile to use
        **kwargs: Configuration overrides
    
    Returns:
        Configured FastMCP server
    """
    # Start with an empty profile if none provided
    config = (profile or ServerProfile()).get_config()
    
    # Override with any kwargs
    config.update(kwargs)
    
    # Create the server
    mcp = FastMCP(config["server_name"])
    
    # Import components only when needed to avoid circular imports
    from . import tutorials
    
    # Add components based on configuration
    if config["enable_documentation"]:
        documentation.register_components(mcp)
    
    if config["enable_codebase"]:
        codebase.register_components(
            mcp, 
            codebase_path=config["codebase_path"]
        )
    
    if config["enable_execution"]:
        execution.register_components(
            mcp, 
            session_context=config["session_context"],
            object_registry=config["object_registry"]
        )
    
    if config["enable_tutorials"]:
        tutorials.register_components(
            mcp, 
            tutorials_path=config["tutorials_path"]
        )
    
    return mcp

async def start_server(server):
    """
    Start the MCP server with proper initialization.
    
    Args:
        server: FastMCP server instance
    """
    # Get the configuration (adjust based on how FastMCP stores config)
    config = getattr(server, "settings", {})
    
    # Import components only when needed to avoid circular imports
    from . import tutorials
    
    # Initialize components
    if getattr(config, "enable_documentation", False):
        await documentation.initialize_components(server, config)
    
    if getattr(config, "enable_codebase", False):
        await codebase.initialize_components(server, config)
    
    if getattr(config, "enable_tutorials", False):
        await tutorials.initialize_components(server, config)
    
    if getattr(config, "enable_execution", False):
        await execution.initialize_components(server, config)
    
    # Start the server
    await server.run()  # Using run() method based on available methods