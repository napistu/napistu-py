"""
MCP client for testing and interacting with Napistu MCP servers.
"""

import json
import logging
from typing import Any, Dict, Mapping, Optional

from fastmcp import Client

from napistu.mcp.config import MCPClientConfig

logger = logging.getLogger(__name__)


async def call_server_tool(
    tool_name: str, arguments: Dict[str, Any], config: MCPClientConfig
) -> Optional[Dict[str, Any]]:
    """
    Call a tool on the MCP server.

    Parameters
    ----------
    tool_name : str
        Name of the tool to call (e.g., 'search_documentation', 'search_codebase')
    arguments : Dict[str, Any]
        Arguments to pass to the tool
    config : MCPClientConfig
        Client configuration object with validated settings.

    Returns
    -------
    Optional[Dict[str, Any]]
        Tool result as dictionary, or None if failed.

    Examples
    --------
    Search for SBML_dfs usage in documentation:

    >>> config = local_client_config()
    >>> result = await call_server_tool(
    ...     "search_documentation",
    ...     {"query": "how do i use sbml_dfs"},
    ...     config
    ... )
    >>> print(f"Found {len(result['results'])} results")

    Search for an exact string in the codebase's class, method, and function definitions:

    >>> result = await call_server_tool(
    ...     "search_codebase",
    ...     {"query": "process_napistu_graph", "search_type": "exact"},
    ...     config
    ... )

    Search tutorials for network creation:

    >>> result = await call_server_tool(
    ...     "search_tutorials",
    ...     {"query": "create consensus networks", "search_type": "semantic"},
    ...     config
    ... )
    """
    try:
        logger.info(f"Calling tool {tool_name} on: {config.mcp_url}")

        client = Client(config.mcp_url)

        async with client:
            # Call the tool
            result = await client.call_tool(tool_name, arguments)

            # Parse the result
            parsed_result = _parse_tool_result(result)
            if parsed_result is None:
                logger.error(f"No result from tool: {tool_name}")
                return None

            return parsed_result

    except Exception as e:
        logger.error(f"Failed to call tool {tool_name}: {str(e)}")
        return None


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


async def search_all(
    query: str,
    search_type: str = "semantic",
    n_results: int = 10,
    config: MCPClientConfig = None,
) -> Optional[Dict[str, Any]]:
    """
    Search across all Napistu components using unified search.

    Parameters
    ----------
    query : str
        Search query or natural language question
    search_type : str, optional
        Search strategy: 'semantic' (default) or 'exact'
    n_results : int, optional
        Maximum number of results to return overall (not per component).
        Results are ranked by similarity score across all components.
        Default is 10.
    config : MCPClientConfig, optional
        Client configuration object with validated settings

    Returns
    -------
    Optional[Dict[str, Any]]
        Search results dictionary containing:
        - query : str
            Original search query
        - search_type : str
            Actual search type used
        - results : List[Dict]
            Search results with component labels
        - tip : str
            Helpful guidance for improving results

    Examples
    --------
    Search across all components:

    >>> config = local_client_config()
    >>> result = await search_all("how to create consensus networks", config=config)
    >>> print(f"Found {len(result['results'])} results")
    >>> for r in result['results']:
    ...     print(f"[{r['component']}] {r['source']}")
    """
    # Validate search type
    valid_search_types = {"semantic", "exact"}
    if search_type not in valid_search_types:
        raise ValueError(
            f"Invalid search_type '{search_type}'. Must be one of: {', '.join(sorted(valid_search_types))}"
        )

    # Call the unified search tool
    return await call_server_tool(
        "search_all",
        {"query": query, "search_type": search_type, "n_results": n_results},
        config,
    )


async def search_component(
    component: str,
    query: str,
    search_type: str = "semantic",
    n_results: int = 5,
    config: MCPClientConfig = None,
) -> Optional[Dict[str, Any]]:
    """
    Search a specific Napistu component using semantic or exact search.

    Parameters
    ----------
    component : str
        Component to search. Must be one of: 'documentation', 'tutorials', 'codebase'
    query : str
        Search query or natural language question
    search_type : str, optional
        Search strategy: 'semantic' (default) or 'exact'
    n_results : int, optional
        Maximum number of results to return. Results are ranked by similarity score.
        Default is 5.
    config : MCPClientConfig, optional
        Client configuration object with validated settings

    Returns
    -------
    Optional[Dict[str, Any]]
        Search results dictionary containing:
        - query : str
            Original search query
        - search_type : str
            Actual search type used
        - results : List[Dict] or Dict[str, List]
            Search results (format depends on search type)
        - tip : str
            Helpful guidance for improving results

    Raises
    ------
    ValueError
        If component is not one of the valid options

    Examples
    --------
    Search documentation for installation info:

    >>> config = local_client_config()
    >>> result = await search_component("documentation", "how to install", config=config)
    >>> print(f"Found {len(result['results'])} results")

    Search codebase for SBML_dfs usage:

    >>> result = await search_component("codebase", "how do i use sbml_dfs", config=config)
    >>> for r in result['results']:
    ...     print(f"- {r['source']}: score {r['similarity_score']:.3f}")

    Search tutorials with exact matching:

    >>> result = await search_component("tutorials", "consensus", "exact", config)

    Search all components by calling multiple times:

    >>> for comp in ['documentation', 'tutorials', 'codebase']:
    ...     result = await search_component(comp, "SBML processing", config=config)
    ...     print(f"{comp}: {len(result.get('results', []))} results")
    """
    # Validate component
    valid_components = {"documentation", "tutorials", "codebase"}
    if component not in valid_components:
        raise ValueError(
            f"Invalid component '{component}'. Must be one of: {', '.join(sorted(valid_components))}"
        )

    # Validate search type
    valid_search_types = {"semantic", "exact"}
    if search_type not in valid_search_types:
        raise ValueError(
            f"Invalid search_type '{search_type}'. Must be one of: {', '.join(sorted(valid_search_types))}"
        )

    # Map component to tool name
    tool_name = f"search_{component}"

    # Call the appropriate tool
    return await call_server_tool(
        tool_name,
        {"query": query, "search_type": search_type, "n_results": n_results},
        config,
    )


# private utils


def _parse_tool_result(result: Any) -> Optional[Dict[str, Any]]:
    """
    Parse a CallToolResult object from FastMCP into a dictionary.

    Handles different result formats:
    - FastMCP 2.12+ CallToolResult with content attribute
    - Legacy list-based format
    - Direct text attributes

    Parameters
    ----------
    result : Any
        The result object from client.call_tool()

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed result as dictionary, or None if result is empty/invalid
    """
    if not result:
        return None

    # Extract text from result based on format
    text = None

    # FastMCP 2.12+ format: CallToolResult with content attribute
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, (list, tuple)) and len(content) > 0:
            # Content is a list of content items
            first_item = content[0]
            if hasattr(first_item, "text"):
                text = first_item.text
            elif isinstance(first_item, str):
                text = first_item
            else:
                text = str(first_item)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

    # Direct text attribute
    elif hasattr(result, "text"):
        text = result.text

    # Legacy format: list/tuple
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        if hasattr(result[0], "text"):
            text = result[0].text
        else:
            text = str(result[0])

    # Fallback: try to convert to string
    else:
        if hasattr(result, "__dict__"):
            return result.__dict__
        text = str(result)

    # Parse JSON if possible, otherwise return as content dict
    if text is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"content": text}

    return None
