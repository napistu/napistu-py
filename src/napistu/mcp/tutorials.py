"""
Tutorial components for the Napistu MCP server.
"""

import logging
from typing import Any, Dict

from fastmcp import FastMCP

from napistu.mcp import tutorials_utils
from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp.constants import (
    HEALTH_SUMMARIES,
    MCP_COMPONENTS,
    SEARCH_RESULT_DEFS,
    SEARCH_TYPES,
    TUTORIAL_URLS,
)
from napistu.mcp.semantic_search import SemanticSearch

logger = logging.getLogger(__name__)


class TutorialsState(ComponentState):
    """
    State management for tutorials component with semantic search capabilities.

    Manages cached tutorial content and tracks semantic search availability.
    Extends ComponentState to provide standardized health monitoring and status reporting.

    Attributes
    ----------
    tutorials : Dict[str, str]
        Dictionary mapping tutorial IDs to their markdown content
    semantic_search : SemanticSearch or None
        Reference to shared semantic search instance for AI-powered tutorial search,
        None if not initialized

    Examples
    --------
    >>> state = TutorialsState()
    >>> state.tutorials["consensus-networks"] = "# Creating Networks..."
    >>> print(state.is_healthy())  # True if any tutorials loaded
    >>> health = state.get_health_details()
    >>> print(health["tutorial_count"])
    """

    def __init__(self):
        super().__init__()
        self.tutorials: Dict[str, str] = {}
        self.semantic_search = None

    def is_healthy(self) -> bool:
        """
        Check if component has successfully loaded tutorial content.

        Returns
        -------
        bool
            True if any tutorials are loaded, False otherwise

        Notes
        -----
        This method checks for the presence of any tutorial content.
        Semantic search availability is not required for health.
        """
        return bool(self.tutorials)

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information including tutorial counts.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - tutorial_count : int
                Number of tutorials loaded
            - tutorial_ids : List[str]
                List of loaded tutorial IDs

        Examples
        --------
        >>> state = TutorialsState()
        >>> # ... load content ...
        >>> details = state.get_health_details()
        >>> print(f"Total tutorials: {details['tutorial_count']}")
        """
        return {
            HEALTH_SUMMARIES.TUTORIAL_COUNT: len(self.tutorials),
            HEALTH_SUMMARIES.TUTORIAL_IDS: list(self.tutorials.keys()),
        }


class TutorialsComponent(MCPComponent):
    """
    MCP component for tutorial management and search with semantic capabilities.

    Provides access to Napistu project tutorials with both exact text matching and
    AI-powered semantic search for natural language queries. Tutorials cover practical
    workflows, code examples, and step-by-step guides for using Napistu functionality.

    The component loads tutorial content from configured URLs and uses a shared
    semantic search instance for natural language tutorial discovery.

    Public Methods
    --------------
    initialize(semantic_search)
        Initialize tutorials component with content loading and semantic indexing.
    register(mcp)
        Register tutorial resources and tools with the MCP server.

    Private Methods
    ---------------
    _check_initialized()
        Check if component is initialized, raise clear error if not.
    _create_state()
        Create tutorials-specific state instance.
    _initialize_semantic_search()
        Index tutorial content into the shared semantic search instance.

    Examples
    --------
    Basic component usage:

    >>> component = TutorialsComponent()
    >>> semantic_search = SemanticSearch()  # Shared instance
    >>> success = await component.safe_initialize(semantic_search)
    >>> if success:
    ...     state = component.get_state()
    ...     print(f"Loaded {state.get_health_details()['tutorial_count']} tutorials")

    Notes
    -----
    The component gracefully handles failures in individual tutorial loading and
    semantic search initialization. If semantic search is not provided, the component
    continues to function with exact text search only.

    **CONTENT SCOPE:**
    Tutorials specifically cover Napistu workflows, not general bioinformatics concepts.
    Use this component for Napistu-specific implementation guidance, not broad domain knowledge.
    """

    async def initialize(self, semantic_search: SemanticSearch = None) -> bool:
        """
        Initialize tutorials component with content loading and semantic indexing.

        Performs the following operations:
        1. Loads tutorial content from configured URLs
        2. Stores reference to shared semantic search instance
        3. Indexes loaded tutorials if semantic search is available

        Parameters
        ----------
        semantic_search : SemanticSearch, optional
            Shared semantic search instance for AI-powered search capabilities.
            If None, component will operate with exact text search only.

        Returns
        -------
        bool
            True if at least one tutorial was loaded successfully, False if
            all loading operations failed

        Notes
        -----
        Individual tutorial failures are logged as warnings but don't fail the entire
        initialization. Semantic search indexing failure is logged but doesn't
        affect the return value - the component can function without semantic search.
        """
        tutorials_loaded = 0

        logger.info("Loading tutorials...")
        for tutorial_id, url in TUTORIAL_URLS.items():
            try:
                content = await tutorials_utils.get_tutorial_markdown(tutorial_id)
                self.state.tutorials[tutorial_id] = content
                tutorials_loaded += 1
                logger.debug(f"Loaded tutorial: {tutorial_id}")
            except Exception as e:
                logger.warning(f"Failed to load tutorial {tutorial_id}: {e}")
                # Continue loading other tutorials even if one fails

        logger.info(f"Loaded {tutorials_loaded}/{len(TUTORIAL_URLS)} tutorials")

        # Store reference to shared semantic search instance
        content_loaded = tutorials_loaded > 0
        if semantic_search and content_loaded:
            self.state.semantic_search = semantic_search
            semantic_success = await self._initialize_semantic_search()
            logger.info(
                f"Semantic search initialization: {'✅ Success' if semantic_success else '⚠️ Failed'}"
            )

        return content_loaded

    def register(self, mcp: FastMCP) -> None:
        """
        Register tutorial resources and tools with the MCP server.

        Registers the following MCP endpoints:
        - Resources for accessing tutorial summaries and specific content
        - Tools for searching tutorials with semantic and exact modes

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance to register endpoints with

        Notes
        -----
        The search tool automatically selects semantic search when available,
        falling back to exact search if semantic search is not initialized.
        """

        # Register resources
        @mcp.resource("napistu://tutorials/index")
        async def get_tutorials_index() -> Dict[str, Any]:
            # Tutorials index doesn't require initialization (uses static URLs)
            pass
            """
            Get the index of all available tutorials.

            **USE THIS WHEN:**
            - Getting an overview of available Napistu tutorials
            - Understanding what tutorial content is available before searching
            - Checking tutorial availability and counts

            **DO NOT USE FOR:**
            - General bioinformatics tutorials (only covers Napistu-specific content)
            - Questions not related to Napistu workflows or implementation
            - Academic research that doesn't involve Napistu usage

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - tutorials : List[Dict[str, str]]
                    List of tutorial metadata with IDs and URLs
                - total_count : int
                    Total number of available tutorials

            Examples
            --------
            Use this to understand what Napistu tutorials are available before
            searching for specific implementation guidance.
            """
            tutorial_list = [
                {"id": tutorial_id, "url": url}
                for tutorial_id, url in TUTORIAL_URLS.items()
            ]

            return {
                "tutorials": tutorial_list,
                "total_count": len(tutorial_list),
            }

        @mcp.resource("napistu://tutorials/content/{tutorial_id}")
        async def get_tutorial_content_resource(tutorial_id: str) -> Dict[str, Any]:
            self._check_initialized()
            """
            Get the full content of a specific Napistu tutorial as markdown.

            **USE THIS WHEN:**
            - Reading a specific Napistu tutorial you've identified via search
            - Getting detailed implementation steps for Napistu workflows
            - Following along with Napistu code examples and exercises

            **DO NOT USE FOR:**
            - General bioinformatics concepts (only covers Napistu implementation)
            - Non-Napistu tools or frameworks
            - Academic theory not related to practical Napistu usage

            Parameters
            ----------
            tutorial_id : str
                ID of the tutorial (from tutorials index)

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - content : str
                    Full markdown content of the tutorial
                - format : str
                    Content format ("markdown")
                - tutorial_id : str
                    ID of the loaded tutorial

            Raises
            ------
            Exception
                If the tutorial cannot be loaded or doesn't exist

            Examples
            --------
            After finding a relevant tutorial via search, use this to get the
            complete tutorial content for step-by-step implementation guidance.
            """
            # Check local state first
            content = self.state.tutorials.get(tutorial_id)

            if content is None:
                # Fallback: try to load on-demand
                try:
                    logger.info(f"Loading tutorial {tutorial_id} on-demand")
                    content = await tutorials_utils.get_tutorial_markdown(tutorial_id)
                    self.state.tutorials[tutorial_id] = content  # Cache for future use
                except Exception as e:
                    logger.error(f"Tutorial {tutorial_id} could not be loaded: {e}")
                    raise

            return {
                "content": content,
                "format": "markdown",
                "tutorial_id": tutorial_id,
            }

        @mcp.tool()
        async def search_tutorials(
            query: str,
            search_type: str = SEARCH_TYPES.SEMANTIC,
            n_results: int = 5,
            max_exact_results: int = 20,
        ) -> Dict[str, Any]:
            self._check_initialized()
            """
            Search Napistu tutorials with intelligent search strategy.

            Provides flexible search capabilities for finding relevant Napistu tutorial content
            using either AI-powered semantic search for natural language queries or exact text
            matching for precise keyword searches. Uses shared semantic search instance when available.

            **USE THIS WHEN:**
            - Looking for Napistu implementation tutorials and workflows
            - Finding step-by-step guides for specific Napistu features
            - Searching for code examples and practical usage patterns
            - Learning how to use Napistu for consensus networks, SBML processing, etc.

            **DO NOT USE FOR:**
            - General bioinformatics concepts not related to Napistu
            - Questions about other tools, libraries, or frameworks
            - Academic research that doesn't involve Napistu implementation
            - Basic programming concepts unrelated to Napistu workflows

            **EXAMPLE APPROPRIATE QUERIES:**
            - "how to create consensus networks"
            - "SBML file processing tutorial"
            - "pathway integration with Napistu"
            - "data source ingestion examples"

            **EXAMPLE INAPPROPRIATE QUERIES:**
            - "what is systems biology" (too general)
            - "how to use pandas" (not Napistu-specific)
            - "gene ontology enrichment" (unless specifically about Napistu implementation)

            Parameters
            ----------
            query : str
                Search term or natural language question about Napistu usage.
                Should be specific to Napistu workflows, features, or implementation.
            search_type : str, optional
                Search strategy to use:
                - "semantic" (default): AI-powered search using embeddings
                - "exact": Traditional text matching search
                Default is "semantic".
            n_results : int, optional
                Maximum number of results to return. Results are ranked by similarity score.
                Default is 5.
            max_exact_results : int, optional
                Only applicable when search_type is "exact". If more than max_exact_results are found,
                an error will be returned rather than returning all results.

            Returns
            -------
            Dict[str, Any]
                Search results dictionary containing:
                - query : str
                    Original search query
                - search_type : str
                    Actual search type used ("semantic" or "exact")
                - results : List[Dict] or List[Dict[str, Any]]
                    Search results. Format depends on search type:
                    - Semantic: List of result dictionaries with content, metadata, source, similarity_score
                    - Exact: List of tutorial matches with id and snippet
                - tip : str
                    Helpful guidance for improving search results

            Examples
            --------
            Natural language semantic search for Napistu workflows:

            >>> results = await search_tutorials("how to create consensus networks")
            >>> print(results["search_type"])  # "semantic"
            >>> for result in results["results"]:
            ...     score = result['similarity_score']
            ...     print(f"Score: {score:.3f} - {result['source']}")

            Exact keyword search for specific Napistu terms:

            >>> results = await search_tutorials("SBML", search_type="exact")
            >>> for result in results["results"]:
            ...     print(f"Tutorial: {result['id']}")

            Notes
            -----
            **SEARCH TYPE GUIDANCE:**
            - Use semantic (default) for conceptual queries and natural language questions
            - Use exact for precise Napistu function names, API calls, or known keywords

            **RESULT INTERPRETATION:**
            - Semantic results include similarity scores (0.8-1.0 = very relevant)
            - Multiple tutorial sections may appear for comprehensive coverage
            - Follow up with get_tutorial_content_resource() for full tutorial text

            The function automatically handles semantic search failures by falling back
            to exact search, ensuring reliable results even if AI components are unavailable.
            """
            if search_type == SEARCH_TYPES.SEMANTIC and self.state.semantic_search:
                # Use shared semantic search instance
                results = self.state.semantic_search.search(
                    query, MCP_COMPONENTS.TUTORIALS, n_results=n_results
                )
                return {
                    SEARCH_RESULT_DEFS.QUERY: query,
                    SEARCH_RESULT_DEFS.SEARCH_TYPE: SEARCH_TYPES.SEMANTIC,
                    SEARCH_RESULT_DEFS.RESULTS: results,
                    SEARCH_RESULT_DEFS.TIP: "For Napistu-specific tutorials only. Try different phrasings if results aren't relevant, or use search_type='exact' for precise keyword matching",
                }
            else:
                # Fall back to exact search
                return tutorials_utils._exact_search_tutorials(
                    query, self.state.tutorials, max_exact_results
                )

    def _check_initialized(self) -> None:
        """
        Check if component is initialized, raise clear error if not.

        Distinguishes between:
        - Still initializing (initialized=False)
        - Failed initialization (initialized=True, initialization_error set)
        """
        if not self.state.initialized:
            raise RuntimeError(
                "TutorialsComponent is still initializing. "
                "This component loads tutorials from GitHub, which may take a few minutes. "
                "Please wait a moment and try again. "
                "You can check initialization status using the health check endpoint."
            )
        elif self.state.initialization_error:
            raise RuntimeError(
                f"TutorialsComponent failed to initialize: {self.state.initialization_error}. "
                "This component requires GitHub tutorial content to function. "
                "Please check the server logs for details or try again later."
            )

    def _create_state(self) -> TutorialsState:
        """
        Create tutorials-specific state instance.

        Returns
        -------
        TutorialsState
            New state instance for managing tutorial content and semantic search
        """
        return TutorialsState()

    async def _initialize_semantic_search(self) -> bool:
        """
        Index tutorial content into the shared semantic search instance.

        Uses the shared semantic search instance (stored in self.state.semantic_search)
        to index this component's tutorial content into the "tutorials" collection.

        Returns
        -------
        bool
            True if content was successfully indexed, False if indexing failed

        Notes
        -----
        Assumes self.state.semantic_search has already been set to a valid
        SemanticSearch instance during initialize().

        Failure to index content is not considered a critical error.
        The component continues to function with exact text search if semantic
        search indexing fails.
        """
        try:
            if not self.state.semantic_search:
                logger.warning("No semantic search instance available")
                return False

            logger.info("Indexing tutorial content for semantic search...")

            # Prepare content for indexing in the format expected by SemanticSearch
            content_dict = {MCP_COMPONENTS.TUTORIALS: self.state.tutorials}

            # Index content into the shared semantic search instance
            self.state.semantic_search.index_content(
                MCP_COMPONENTS.TUTORIALS, content_dict
            )

            logger.info("✅ Tutorial content indexed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to index tutorial content: {e}")
            return False


# Module-level component instance
_component = TutorialsComponent()


def get_component() -> TutorialsComponent:
    """
    Get the tutorials component instance.

    Returns
    -------
    TutorialsComponent
        Singleton tutorials component instance for use across the MCP server.
        The same instance is returned on every call to ensure consistent state.

    Notes
    -----
    This function provides the standard interface for accessing the tutorials
    component. The component must be initialized via safe_initialize() before use.
    """
    return _component
