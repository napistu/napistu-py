"""
MCP client for testing and interacting with Napistu MCP servers.
"""

import json
import logging
from typing import Optional, Dict, Any, Mapping

from fastmcp import Client
from napistu.mcp.config import MCPClientConfig

logger = logging.getLogger(__name__)


async def check_server_health(config: MCPClientConfig) -> Optional[Dict[str, Any]]:
    """
    Health check using FastMCP client.

    Parameters
    ----------
    config : MCPClientConfig
        Client configuration object with validated settings.

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
        logger.info(f"Connecting to MCP server at: {config.mcp_url}")

        client = Client(config.mcp_url)

        async with client:
            logger.info("✅ FastMCP client connected")

            # List all available resources
            resources = await client.list_resources()
            logger.info(f"Found {len(resources)} resources")

            # Find health resource
            health_resource = None
            for resource in resources:
                uri_str = str(resource.uri).lower()
                if "health" in uri_str:
                    health_resource = resource
                    logger.info(f"Found health resource: {resource.uri}")
                    break

            if not health_resource:
                logger.error("No health resource found")
                logger.info(f"Available resources: {[str(r.uri) for r in resources]}")
                return None

            # Read the health resource
            logger.info(f"Reading health resource: {health_resource.uri}")
            result = await client.read_resource(str(health_resource.uri))

            if result and len(result) > 0 and hasattr(result[0], "text"):
                try:
                    health_data = json.loads(result[0].text)
                    logger.info("✅ Health check successful")
                    return health_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse health JSON: {e}")
                    logger.error(f"Raw response: {result[0].text}")
                    return None
            else:
                logger.error(f"No valid response from health resource: {result}")
                return None

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        if hasattr(e, "__traceback__"):
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
            - timestamp : str, optional
                ISO format timestamp
            - version : str, optional
                Package version

    Returns
    -------
    None
        Prints health status information to stdout.
    """
    if not health:
        print("❌ Could not get health status")
        print("Check the logs above for detailed error information")
        return

    status = health.get("status", "unknown")
    print(f"\nServer Status: {status}")

    components = health.get("components", {})
    if components:
        print("\nComponents:")
        for name, comp_status in components.items():
            icon = "✅" if comp_status.get("status") == "healthy" else "❌"
            print(f"  {icon} {name}: {comp_status.get('status', 'unknown')}")

    # Show additional info if available
    if "timestamp" in health:
        print(f"\nTimestamp: {health['timestamp']}")
    if "version" in health:
        print(f"Version: {health['version']}")


async def list_server_resources(config: MCPClientConfig) -> Optional[list]:
    """
    List all available resources on the MCP server.

    Parameters
    ----------
    config : MCPClientConfig
        Client configuration object with validated settings.

    Returns
    -------
    Optional[list]
        List of available resources, or None if failed.
    """
    try:
        logger.info(f"Listing resources from: {config.mcp_url}")

        client = Client(config.mcp_url)

        async with client:
            resources = await client.list_resources()
            logger.info(f"Found {len(resources)} resources")
            return resources

    except Exception as e:
        logger.error(f"Failed to list resources: {str(e)}")
        return None


async def read_server_resource(
    resource_uri: str, config: MCPClientConfig
) -> Optional[str]:
    """
    Read a specific resource from the MCP server.

    Parameters
    ----------
    resource_uri : str
        URI of the resource to read (e.g., 'napistu://health')
    config : MCPClientConfig
        Client configuration object with validated settings.

    Returns
    -------
    Optional[str]
        Resource content as text, or None if failed.
    """
    try:
        logger.info(f"Reading resource {resource_uri} from: {config.mcp_url}")

        client = Client(config.mcp_url)

        async with client:
            result = await client.read_resource(resource_uri)

            if result and len(result) > 0 and hasattr(result[0], "text"):
                return result[0].text
            else:
                logger.error(f"No content found for resource: {resource_uri}")
                return None

    except Exception as e:
        logger.error(f"Failed to read resource {resource_uri}: {str(e)}")
        return None
