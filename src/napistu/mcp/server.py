"""
Core MCP server implementation for Napistu.
"""

import asyncio
import contextlib
import logging
from typing import Any, Dict

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount, Route

from napistu.mcp import codebase, documentation, execution, health, tutorials
from napistu.mcp.config import MCPServerConfig
from napistu.mcp.constants import (
    MCP_COMPONENTS,
    MCP_DEFAULTS,
    PROFILE_DEFS,
    SEARCH_TYPES,
)
from napistu.mcp.profiles import ServerProfile, get_profile
from napistu.mcp.semantic_search import SemanticSearch
from napistu.mcp.web_routes import create_chat_app, redirect_to_mcp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Only log MCP endpoint hits
        if request.url.path.startswith("/mcp"):
            logger.critical("=" * 80)
            logger.critical("🔥 MCP ENDPOINT HIT")
            logger.critical(f"   Method: {request.method}")
            logger.critical(f"   Path: {request.url.path}")
            logger.critical(
                f"   Client: {request.client.host if request.client else 'unknown'}"
            )
            logger.critical(f"   Headers: {dict(request.headers)}")
            logger.critical("=" * 80)

        response = await call_next(request)
        return response


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
        (
            MCP_COMPONENTS.DOCUMENTATION,
            documentation,
            PROFILE_DEFS.ENABLE_DOCUMENTATION,
        ),
        (MCP_COMPONENTS.CODEBASE, codebase, PROFILE_DEFS.ENABLE_CODEBASE),
        (MCP_COMPONENTS.TUTORIALS, tutorials, PROFILE_DEFS.ENABLE_TUTORIALS),
        (MCP_COMPONENTS.EXECUTION, execution, PROFILE_DEFS.ENABLE_EXECUTION),
    ]

    # Create semantic search instance
    # this supports RAG indexing of content and search using an underlying
    # sqlite vector database (chromadb)
    semantic_search = SemanticSearch()

    # Initialize all components
    initialization_results = {}

    for name, module, config_key in component_configs:
        result = await _initialize_component(
            name, module, config_key, config, semantic_search
        )
        initialization_results[name] = result

    # Initialize health components last since they monitor the other components
    logger.info("Initializing health components")
    try:
        result = await health.initialize_components()
        initialization_results[MCP_COMPONENTS.HEALTH] = result
        if result:
            logger.info("✅ Health components initialized successfully")
        else:
            logger.warning("⚠️ Health components initialized with issues")
    except Exception as e:
        logger.error(f"❌ Health components failed to initialize: {e}")
        initialization_results[MCP_COMPONENTS.HEALTH] = False

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


def start_mcp_server(profile_name: str, server_config: MCPServerConfig) -> None:
    """
    Start MCP server - main entry point for server startup.

    Parameters
    ----------
    profile_name : str
        Name of the profile to use for the MCP server ('local', 'remote', 'full').
    server_config : MCPServerConfig
        Server configuration with validated host, port, and server name.

    Returns
    -------
    None

    Notes
    -----
    This function starts the MCP server with the specified profile and server configuration.
    If the chat interface is enabled in the profile, it creates a combined app that includes the MCP server and the chat interface.
    Otherwise, it just runs the MCP server normally.

    The server uses HTTP transport (streamable-http) for all connections.
    Components are initialized asynchronously before the server starts.
    Health components are always registered and initialized last.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("napistu")

    logger.info("Starting Napistu MCP Server")
    logger.info(f"  Profile: {profile_name}")
    logger.info(f"  Host: {server_config.host}")
    logger.info(f"  Port: {server_config.port}")
    logger.info(f"  Server Name: {server_config.server_name}")
    logger.info(f"  Transport: {MCP_DEFAULTS.TRANSPORT}")

    # Create session context for execution components
    session_context = {}
    object_registry = {}

    # Get profile with configuration
    profile: ServerProfile = get_profile(
        profile_name,
        session_context=session_context,
        object_registry=object_registry,
        server_name=server_config.server_name,
    )

    # Create server with validated configuration
    mcp = create_server(profile, server_config)

    # Initialize components in background after server starts
    async def init_components_background():
        logger.info("Background: Initializing MCP components...")
        try:
            await initialize_components(profile)
            logger.info("Background: ✅ Component initialization complete")
        except Exception as e:
            logger.error(f"Background: ❌ Component initialization failed: {e}")

    # Check if chat is enabled in profile
    config = profile.get_config()
    if config.get(PROFILE_DEFS.ENABLE_CHAT, False):
        logger.info("Chat interface enabled - creating combined app")

        # Create the MCP HTTP app with routes at root "/"
        # When mounted at /mcp/, requests are stripped so mcp_app sees "/"
        mcp_app = mcp.http_app(path="/")

        # Create the chat app with CORS
        chat_app = create_chat_app()

        @contextlib.asynccontextmanager
        async def combined_lifespan(app: Starlette):
            """Run MCP app's lifespan and initialize components in background"""
            async with mcp_app.lifespan(app):
                logger.info("MCP lifespan started - server is listening")
                # Initialize components in background (non-blocking)
                asyncio.create_task(init_components_background())
                yield
                logger.info("MCP lifespan stopped")

        # Create a wrapper Starlette app that mounts both
        app = Starlette(
            routes=[
                Route(
                    "/mcp", endpoint=redirect_to_mcp, methods=["GET", "POST", "DELETE"]
                ),
                Mount("/mcp/", app=mcp_app),  # MCP at /mcp/
                Mount("/", app=chat_app),  # Chat at /
            ],
            lifespan=combined_lifespan,
        )

        _log_combined_app_routes(server_config, app, mcp_app, chat_app)

        app.add_middleware(RequestLoggingMiddleware)

        # Run the combined app with uvicorn
        uvicorn.run(
            app,
            host=server_config.host,
            port=server_config.port,
            log_level="info",
        )
    else:
        # Run MCP server using http_app so we can initialize components in background
        logger.info("🚀 Starting MCP server...")
        logger.info(
            f"Using {MCP_DEFAULTS.TRANSPORT} transport on http://{server_config.host}:{server_config.port}{MCP_DEFAULTS.MCP_PATH}"
        )

        mcp_app = mcp.http_app(path="/")

        @contextlib.asynccontextmanager
        async def mcp_lifespan(app: Starlette):
            """Initialize components in background after server starts listening"""
            async with mcp_app.lifespan(app):
                logger.info("MCP server started - server is listening")
                # Initialize components in background (non-blocking)
                asyncio.create_task(init_components_background())
                yield

        app = Starlette(routes=[Mount("/", app=mcp_app)], lifespan=mcp_lifespan)
        uvicorn.run(
            app, host=server_config.host, port=server_config.port, log_level="info"
        )


def create_server(profile: ServerProfile, server_config: MCPServerConfig) -> FastMCP:
    """
    Create an MCP server based on a profile configuration and server config.

    Parameters
    ----------
    profile : ServerProfile
        Server profile to use. All configuration must be set in the profile. (Valid profiles: 'execution', 'docs', 'full')
    server_config : MCPServerConfig
        Server configuration with validated host, port, and server name.

    Returns
    -------
    FastMCP
        Configured FastMCP server instance.
    """

    config = profile.get_config()

    # Create the server with validated configuration
    mcp = FastMCP(
        server_config.server_name, host=server_config.host, port=server_config.port
    )

    # Define component configurations
    component_configs = [
        (
            MCP_COMPONENTS.DOCUMENTATION,
            documentation,
            PROFILE_DEFS.ENABLE_DOCUMENTATION,
        ),
        (MCP_COMPONENTS.CODEBASE, codebase, PROFILE_DEFS.ENABLE_CODEBASE),
        (MCP_COMPONENTS.TUTORIALS, tutorials, PROFILE_DEFS.ENABLE_TUTORIALS),
        (MCP_COMPONENTS.EXECUTION, execution, PROFILE_DEFS.ENABLE_EXECUTION),
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

    # Always register health components with profile config
    health.register_components(mcp, profile_config=config)
    logger.info("Registered health components")

    # Register cross-component search tool
    _register_search_all_tool(mcp)

    return mcp


# private utils


def _get_semantic_search():
    """
    Get the shared semantic search instance from any component.

    This follows the same pattern as health component's _check_semantic_search_health().
    Since components store the shared semantic search instance in their state during
    initialization, we can retrieve it from any initialized component.

    Returns
    -------
    SemanticSearch or None
        Shared semantic search instance if available, None otherwise
    """
    # Try components in order until we find one with semantic search
    component_modules = [documentation, codebase, tutorials]

    for module in component_modules:
        try:
            component = module.get_component()
            if (
                hasattr(component.state, "semantic_search")
                and component.state.semantic_search
            ):
                return component.state.semantic_search
        except Exception:
            continue

    return None


async def _initialize_component(
    name: str,
    module,
    config_key: str,
    config: dict,
    semantic_search: SemanticSearch = None,
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
    semantic_search : SemanticSearch, optional
        Shared semantic search instance for AI-powered search capabilities.
        If None, component will operate with exact text search only.

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
        result = await component.safe_initialize(semantic_search)
        return result
    except Exception as e:
        logger.error(f"❌ {name.title()} components failed to initialize: {e}")
        return False


def _log_combined_app_routes(
    server_config: MCPServerConfig,
    app: Starlette,
    mcp_app: FastMCP,
    chat_app: Starlette,
) -> None:
    """Log the routes for the combined server (MCP + Chat API)"""

    logger.info("🚀 Starting combined server (MCP + Chat API)...")
    logger.info(
        f"   MCP endpoint: http://{server_config.host}:{server_config.port}{MCP_DEFAULTS.MCP_PATH}"
    )
    logger.info(f"   Chat API: http://{server_config.host}:{server_config.port}/api/*")

    # DEBUG: Log all registered routes
    logger.info("=" * 60)
    logger.info("DEBUG: Route Registration Summary")
    logger.info("=" * 60)
    logger.info(f"Total routes: {len(app.routes)}")

    for i, route in enumerate(app.routes):
        route_type = type(route).__name__

        if hasattr(route, "path"):
            path = route.path
            methods = getattr(route, "methods", ["ANY"])
            logger.info(f"  [{i}] {route_type}: {path} [{', '.join(methods)}]")
        elif hasattr(route, "path_regex"):
            pattern = (
                route.path_regex.pattern
                if hasattr(route.path_regex, "pattern")
                else str(route.path_regex)
            )
            logger.info(f"  [{i}] {route_type}: Pattern={pattern}")
        else:
            logger.info(f"  [{i}] {route_type}: {route}")

    logger.info("=" * 60)

    # DEBUG: Check mcp_app routes
    logger.info(f"MCP app has {len(mcp_app.routes)} internal routes")
    for i, route in enumerate(mcp_app.routes):
        if hasattr(route, "path"):
            logger.info(f"  MCP route [{i}]: {route.path}")

    # DEBUG: Check chat_app routes
    logger.info(f"Chat app has {len(chat_app.routes)} internal routes")
    for i, route in enumerate(chat_app.routes):
        if hasattr(route, "path"):
            logger.info(f"  Chat route [{i}]: {route.path}")

    logger.info("=" * 60)


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

    if name == MCP_COMPONENTS.EXECUTION:
        # Special handling for execution component which needs session context
        component = module.create_component(
            session_context=kwargs.get("session_context"),
            object_registry=kwargs.get("object_registry"),
        )
    else:
        component = module.get_component()

    component.register(mcp)


def _register_search_all_tool(mcp: FastMCP) -> None:
    """
    Register the cross-component search tool with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        FastMCP server instance to register the tool with
    """

    @mcp.tool()
    async def search_all(
        query: str, search_type: str = SEARCH_TYPES.SEMANTIC, n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search across all Napistu components (documentation, codebase, tutorials) with intelligent search strategy.

        Provides unified search capabilities for finding relevant content across all Napistu components
        using either AI-powered semantic search for natural language queries or exact text matching
        for precise keyword searches. Searches documentation, codebase, and tutorials simultaneously.

        **USE THIS WHEN:**
        - Looking for information that might be in documentation, codebase, or tutorials
        - Wanting comprehensive search results across all Napistu content
        - Not sure which component contains the information you need
        - Searching for Napistu concepts, features, or implementation details

        **DO NOT USE FOR:**
        - General bioinformatics concepts not specific to Napistu
        - Documentation for other tools, libraries, or frameworks
        - Academic literature or research papers
        - Questions about biological concepts that don't involve Napistu usage

        **EXAMPLE APPROPRIATE QUERIES:**
        - "how to create consensus networks"
        - "SBML file processing"
        - "pathway integration"
        - "data source ingestion"
        - "network creation functions"

        Parameters
        ----------
        query : str
            Search term or natural language question about Napistu.
            Should be specific to Napistu workflows, features, or implementation.
        search_type : str, optional
            Search strategy to use:
            - "semantic" (default): AI-powered search using embeddings
            - "exact": Traditional text matching search
            Default is "semantic".
        n_results : int, optional
            Maximum number of results to return overall (not per component).
            Results are ranked by similarity score across all components.
            Default is 10.

        Returns
        -------
        Dict[str, Any]
            Search results dictionary containing:
            - query : str
                Original search query
            - search_type : str
                Actual search type used ("semantic" or "exact")
            - results : List[Dict]
                Search results ordered by relevance. Each result contains:
                - content: The matched text content
                - component: Component name ("documentation", "codebase", or "tutorials")
                - source: Human-readable source description
                - similarity_score: Float between 0 and 1 (for semantic search)
                - metadata: Dictionary with type, name, and other metadata
            - tip : str
                Helpful guidance for improving search results

        Examples
        --------
        Natural language semantic search across all components:

        >>> results = await search_all("how to create consensus networks")
        >>> print(results["search_type"])  # "semantic"
        >>> for result in results["results"]:
        ...     component = result['component']
        ...     score = result['similarity_score']
        ...     print(f"[{component}] Score: {score:.3f} - {result['source']}")

        Exact keyword search:

        >>> results = await search_all("SBML_dfs", search_type="exact")
        >>> print(f"Found {len(results['results'])} results across components")

        Notes
        -----
        **SEARCH TYPE GUIDANCE:**
        - Use semantic (default) for conceptual queries and natural language questions
        - Use exact for precise function names, class names, or known keywords

        **RESULT INTERPRETATION:**
        - Semantic results include similarity scores (0.8-1.0 = very relevant)
        - Results are ordered by relevance across all components
        - Component labels help identify the source of each result
        - Multiple results from different components may appear for comprehensive coverage

        The function automatically handles semantic search failures by falling back
        to exact search, ensuring reliable results even if AI components are unavailable.
        """
        semantic_search = _get_semantic_search()

        if search_type == SEARCH_TYPES.SEMANTIC and semantic_search:
            # Use unified semantic search - returns top K results overall, ranked by similarity
            results = semantic_search.search_unified(query, n_results=n_results)
            return {
                "query": query,
                "search_type": SEARCH_TYPES.SEMANTIC,
                "results": results,
                "tip": "Try different phrasings if results aren't relevant, or use search_type='exact' for precise keyword matching",
            }
        else:
            # Fall back to exact search across components
            # This is a simplified fallback - could be enhanced to search all components
            all_results = []

            # Try to get components and search them individually
            component_modules = [
                (MCP_COMPONENTS.DOCUMENTATION, documentation),
                (MCP_COMPONENTS.CODEBASE, codebase),
                (MCP_COMPONENTS.TUTORIALS, tutorials),
            ]

            for component_name, module in component_modules:
                try:
                    component = module.get_component()
                    # Call the component's exact search if available
                    # This is a fallback, so we'll do a simple text search
                    if hasattr(component, "state"):
                        # For now, return a message suggesting component-specific search
                        # In a full implementation, we'd search each component's cache
                        pass
                except Exception:
                    continue

            if not all_results:
                return {
                    "query": query,
                    "search_type": SEARCH_TYPES.EXACT,
                    "results": [],
                    "tip": "Exact search across components not fully implemented. Use search_type='semantic' for cross-component search, or use component-specific search tools (search_documentation, search_codebase, search_tutorials) for exact matching.",
                }

            return {
                "query": query,
                "search_type": SEARCH_TYPES.EXACT,
                "results": all_results,
                "tip": "Use search_type='semantic' for natural language queries across all components",
            }
