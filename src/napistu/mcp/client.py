# src/napistu/mcp/client.py
"""
MCP client for testing and interacting with Napistu MCP servers.
"""

import json
import logging
import sys
from typing import Optional, Dict, Any, Mapping

from fastmcp import Client

logger = logging.getLogger(__name__)

async def check_server_health(transport: str = "stdio", server_url: str = "http://127.0.0.1:8765") -> Optional[Dict[str, Any]]:
    """
    Simple health check of an MCP server.
    
    Parameters
    ----------
    transport : str, optional
        Transport type ('stdio' or 'http'). Defaults to 'stdio'.
    server_url : str, optional
        Server URL for HTTP transport. Defaults to 'http://127.0.0.1:8765'.
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing health status information if successful, None if failed.
        The dictionary contains:
            - status : str
                Overall server status ('healthy', 'degraded', or 'unhealthy')
            - timestamp : str
                ISO format timestamp of the health check
            - version : str
                Version of the Napistu package
            - components : Dict[str, Dict[str, str]]
                Status of each component ('healthy', 'inactive', or 'unavailable')
    """
    try:
        # Create client based on transport type
        if transport == "stdio":
            # For stdio, we need to provide the command to run the server
            config = {
                "mcpServers": {
                    "local": {
                        "command": sys.executable,  # Current Python interpreter
                        "args": ["-m", "napistu.mcp", "server", "local"]
                    }
                }
            }
            client = Client(config)
        else:
            # For HTTP, connect to the specified URL
            config = {
                "mcpServers": {
                    "remote": {
                        "url": server_url
                    }
                }
            }
            client = Client(config)
            
        async with client:
            # Read the health resource
            result = await client.read_resource("napistu://health")
            if result and hasattr(result[0], 'text'):
                try:
                    return json.loads(result[0].text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse health check response: {e}")
                    return None
            return None
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            logger.error("Traceback:\n" + "".join(traceback.format_tb(e.__traceback__)))
        return None

def print_health_status(health: Optional[Mapping[str, Any]]) -> None:
    """
    Pretty print health status information.
    
    Parameters
    ----------
    health : Optional[Mapping[str, Any]]
        Health status dictionary from check_server_health, or None if health check failed.
        Expected to contain:
            - status : str
                Overall server status
            - components : Dict[str, Dict[str, str]]
                Status of each component
                
    Returns
    -------
    None
        Prints health status information to stdout.
    """
    if not health:
        print("❌ Could not get health status")
        print("Check the logs above for detailed error information")
        return
    
    status = health.get('status', 'unknown')
    print(f"\nServer Status: {status}")
    
    components = health.get('components', {})
    if components:
        print("\nComponents:")
        for name, comp_status in components.items():
            icon = "✅" if comp_status.get('status') == 'healthy' else "❌"
            print(f"  {icon} {name}: {comp_status.get('status', 'unknown')}")
