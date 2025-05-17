"""
Core MCP server implementation for Napistu.
"""

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP support not installed. Install with 'pip install napistu[mcp]'"
    )

from typing import Dict, Any, Optional
import asyncio

from .profiles import ServerProfile
from .components import documentation, codebase, execution, tutorials

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
    
    # Add components based on configuration
    if config["enable_documentation"]:
        documentation.register_components(
            mcp, 
            docs_paths=config["docs_paths"]
        )
    
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
    config = server.config
    
    # Initialize components
    if config.get("enable_documentation"):
        await documentation.initialize_components(server, config)
    
    if config.get("enable_codebase"):
        await codebase.initialize_components(server, config)
    
    if config.get("enable_tutorials"):
        await tutorials.initialize_components(server, config)
    
    if config.get("enable_execution"):
        await execution.initialize_components(server, config)
    
    # Start the server
    await server.start()