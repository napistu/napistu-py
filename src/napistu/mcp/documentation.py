"""
Documentation components for the Napistu MCP server.
"""

import logging
from typing import Any, Dict

from fastmcp import FastMCP

from napistu.mcp import documentation_utils
from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp.constants import (
    DOCUMENTATION,
    DOCUMENTATION_SUMMARY_DEFS,
    GITHUB_DEFS,
    HEALTH_SUMMARIES,
    MCP_COMPONENTS,
    READMES,
    REPOS_WITH_ISSUES,
    SEARCH_RESULT_DEFS,
    SEARCH_TYPES,
    WIKI_PAGES,
)
from napistu.mcp.semantic_search import SemanticSearch

logger = logging.getLogger(__name__)


class DocumentationState(ComponentState):
    """
    State management for documentation component with semantic search capabilities.

    Manages cached documentation content from multiple sources and tracks semantic
    search initialization status. Extends ComponentState to provide standardized
    health monitoring and status reporting.

    Attributes
    ----------
    docs_cache : Dict[str, Dict[str, Any]]
        Nested dictionary containing cached documentation content organized by type:
        - readme: README files from repositories
        - wiki: Wiki pages from project documentation
        - issues: GitHub issues from project repositories
        - prs: Pull requests from project repositories
        - packagedown: Package documentation sections (if any)
    semantic_search : SemanticSearch or None
        Semantic search instance for AI-powered content search, None if not initialized

    Examples
    --------
    >>> state = DocumentationState()
    >>> state.docs_cache["readme"]["install"] = "Installation guide..."
    >>> print(state.is_healthy())  # True if any content loaded
    >>> health = state.get_health_details()
    """

    def __init__(self):
        """Initialize documentation state with empty cache and no semantic search."""
        super().__init__()
        self.docs_cache: Dict[str, Dict[str, Any]] = {
            DOCUMENTATION.README: {},
            DOCUMENTATION.WIKI: {},
            DOCUMENTATION.ISSUES: {},
            DOCUMENTATION.PRS: {},
            DOCUMENTATION.PACKAGEDOWN: {},
        }
        self.semantic_search = None

    def is_healthy(self) -> bool:
        """
        Check if component has successfully loaded documentation content.

        Returns
        -------
        bool
            True if any documentation section contains content, False otherwise

        Notes
        -----
        This method checks for the presence of any content in any documentation
        category. Semantic search availability is not required for health.
        """
        return any(bool(section) for section in self.docs_cache.values())

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information including content counts and semantic search status.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - readme_count : int
                Number of README files loaded
            - wiki_pages : int
                Number of wiki pages loaded
            - issues_repos : int
                Number of repositories with issues loaded
            - prs_repos : int
                Number of repositories with pull requests loaded
            - total_sections : int
                Total number of content items across all categories

        Examples
        --------
        >>> state = DocumentationState()
        >>> # ... load content ...
        >>> details = state.get_health_details()
        >>> print(f"Total content items: {details['total_sections']}")
        """
        base_details = {
            HEALTH_SUMMARIES.README_COUNT: len(self.docs_cache[DOCUMENTATION.README]),
            HEALTH_SUMMARIES.WIKI_PAGES: len(self.docs_cache[DOCUMENTATION.WIKI]),
            HEALTH_SUMMARIES.ISSUES_REPOS: len(self.docs_cache[DOCUMENTATION.ISSUES]),
            HEALTH_SUMMARIES.PRS_REPOS: len(self.docs_cache[DOCUMENTATION.PRS]),
            HEALTH_SUMMARIES.TOTAL_SECTIONS: sum(
                len(section) for section in self.docs_cache.values()
            ),
        }

        return base_details


class DocumentationComponent(MCPComponent):
    """
    MCP component for documentation management and search with semantic capabilities.

    Provides access to Napistu project documentation including README files, wiki pages,
    GitHub issues, and pull requests. Supports both exact text matching and AI-powered
    semantic search for flexible content discovery.

    The component loads documentation from multiple sources:
    - README files from GitHub repositories (raw URLs)
    - Wiki pages from project wikis
    - GitHub issues and pull requests via GitHub API
    - Optional package documentation sections

    After loading content, the component initializes semantic search capabilities
    using ChromaDB and sentence transformers for natural language queries.

    Public Methods
    --------------
    initialize(semantic_search)
        Initialize documentation component with content loading and semantic indexing.
    register(mcp)
        Register documentation resources and tools with the MCP server.

    Private Methods
    ---------------
    _check_initialized()
        Check if component is initialized, raise clear error if not.
    _create_state()
        Create documentation-specific state instance.
    _initialize_semantic_search()
        Index documentation content into the shared semantic search instance.

    Examples
    --------
    Basic component usage:

    >>> component = DocumentationComponent()
    >>> success = await component.safe_initialize()
    >>> if success:
    ...     state = component.get_state()
    ...     print(f"Loaded {state.get_health_details()['total_sections']} items")

    Notes
    -----
    The component gracefully handles failures in individual documentation sources
    and semantic search initialization. If semantic search fails, the component
    continues to function with exact text search only.
    """

    async def initialize(self, semantic_search: SemanticSearch = None) -> bool:
        """
        Initialize documentation component with content loading and semantic indexing.

        Performs the following operations:
        1. Loads README files from configured repository URLs
        2. Fetches wiki pages from project wikis
        3. Retrieves GitHub issues and pull requests via API
        4. Initializes semantic search and indexes loaded content

        Returns
        -------
        bool
            True if at least some documentation was loaded successfully, False if
            all loading operations failed
        semantic_search : SemanticSearch
            Semantic search instance for AI-powered content search, None if not initialized

        Notes
        -----
        Individual source failures are logged as warnings but don't fail the entire
        initialization. Semantic search initialization failure is logged but doesn't
        affect the return value - the component can function without semantic search.

        The method tracks success/failure rates and provides detailed logging for
        debugging content loading issues.
        """
        success_count = 0
        total_operations = 0

        # Load README files
        logger.info("Loading README files...")
        for name, url in READMES.items():
            total_operations += 1
            try:
                content = await documentation_utils.load_readme_content(url)
                self.state.docs_cache[DOCUMENTATION.README][name] = content
                success_count += 1
                logger.debug(f"Loaded README: {name}")
            except Exception as e:
                logger.warning(f"Failed to load README {name}: {e}")

        # Load wiki pages
        logger.info("Loading wiki pages...")
        for page in WIKI_PAGES:
            total_operations += 1
            try:
                content = await documentation_utils.fetch_wiki_page(page)
                self.state.docs_cache[DOCUMENTATION.WIKI][page] = content
                success_count += 1
                logger.debug(f"Loaded wiki page: {page}")
            except Exception as e:
                logger.warning(f"Failed to load wiki page {page}: {e}")

        # Load issues and PRs
        logger.info("Loading issues and pull requests...")
        for repo in REPOS_WITH_ISSUES:
            total_operations += 2  # Issues and PRs
            try:
                issues = await documentation_utils.list_issues(repo)
                self.state.docs_cache[DOCUMENTATION.ISSUES][repo] = issues
                success_count += 1
                logger.debug(f"Loaded issues for repo: {repo}")
            except Exception as e:
                logger.warning(f"Failed to load issues for {repo}: {e}")

            try:
                prs = await documentation_utils.list_pull_requests(repo)
                self.state.docs_cache[DOCUMENTATION.PRS][repo] = prs
                success_count += 1
                logger.debug(f"Loaded PRs for repo: {repo}")
            except Exception as e:
                logger.warning(f"Failed to load PRs for {repo}: {e}")

        logger.info(
            f"Documentation loading complete: {success_count}/{total_operations} operations successful"
        )

        # Initialize semantic search if content was loaded
        content_loaded = success_count > 0
        if semantic_search and content_loaded:
            self.state.semantic_search = semantic_search  # Reference, not creation
            semantic_success = await self._initialize_semantic_search()
            logger.info(
                f"Semantic search initialization: {'✅ Success' if semantic_success else '⚠️ Failed'}"
            )

        return content_loaded

    def register(self, mcp: FastMCP) -> None:
        """
        Register documentation resources and tools with the MCP server.

        Registers the following MCP endpoints:
        - Resources for accessing documentation summaries and specific content
        - Tools for searching documentation with semantic and exact modes

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance to register endpoints with

        Notes
        -----
        The search tool automatically selects semantic search when available,
        falling back to exact search if semantic search is not initialized.
        """

        # Register existing resources (unchanged)
        @mcp.resource("napistu://documentation/summary")
        async def get_documentation_summary():
            """
            Get a comprehensive summary of all available Napistu project documentation.

            **USE THIS WHEN:**
            - Getting an overview of available Napistu documentation before searching
            - Understanding what types of Napistu content are available (READMEs, wikis, issues)
            - Checking if semantic search is available for documentation

            **DO NOT USE FOR:**
            - General bioinformatics documentation (only covers Napistu project)
            - Questions not related to Napistu implementation or usage
            - Academic literature or external tool documentation

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - readme_files : List[str]
                    Names of loaded README files from Napistu repositories
                - issues : List[str]
                    Repository names with loaded GitHub issues
                - prs : List[str]
                    Repository names with loaded pull requests
                - wiki_pages : List[str]
                    Names of loaded Napistu wiki pages
                - packagedown_sections : List[str]
                    Names of package documentation sections
                - semantic_search : Dict[str, bool]
                    Status of semantic search availability and indexing

            Examples
            --------
            Use this to understand what Napistu documentation is available before
            searching for specific implementation details or troubleshooting information.
            """

            summary = {
                DOCUMENTATION_SUMMARY_DEFS.README_FILES: list(
                    self.state.docs_cache[DOCUMENTATION.README].keys()
                ),
                DOCUMENTATION_SUMMARY_DEFS.ISSUES: list(
                    self.state.docs_cache[DOCUMENTATION.ISSUES].keys()
                ),
                DOCUMENTATION_SUMMARY_DEFS.PRS: list(
                    self.state.docs_cache[DOCUMENTATION.PRS].keys()
                ),
                DOCUMENTATION_SUMMARY_DEFS.WIKI_PAGES: list(
                    self.state.docs_cache[DOCUMENTATION.WIKI].keys()
                ),
                DOCUMENTATION_SUMMARY_DEFS.PACKAGEDOWN_SECTIONS: list(
                    self.state.docs_cache[DOCUMENTATION.PACKAGEDOWN].keys()
                ),
            }

            return summary

        @mcp.resource("napistu://documentation/readme/{file_name}")
        async def get_readme_content(file_name: str):
            """Get the content of a specific README file."""
            if file_name not in self.state.docs_cache[DOCUMENTATION.README]:
                return {"error": f"README file {file_name} not found"}

            return {
                "content": self.state.docs_cache[DOCUMENTATION.README][file_name],
                "format": "markdown",
            }

        @mcp.resource("napistu://documentation/issues/{repo}")
        async def get_issues(repo: str):
            """Get the list of issues for a given repository."""
            return self.state.docs_cache[DOCUMENTATION.ISSUES].get(repo, [])

        @mcp.resource("napistu://documentation/prs/{repo}")
        async def get_prs(repo: str):
            """Get the list of pull requests for a given repository."""
            return self.state.docs_cache[DOCUMENTATION.PRS].get(repo, [])

        @mcp.resource("napistu://documentation/issue/{repo}/{number}")
        async def get_issue_resource(repo: str, number: int):
            """Get a single issue by number for a given repository."""
            # Try cache first
            cached = next(
                (
                    i
                    for i in self.state.docs_cache[DOCUMENTATION.ISSUES].get(repo, [])
                    if i[GITHUB_DEFS.NUMBER] == number
                ),
                None,
            )
            if cached:
                return cached
            # Fallback to live fetch
            return await documentation_utils.get_issue(repo, number)

        @mcp.resource("napistu://documentation/pr/{repo}/{number}")
        async def get_pr_resource(repo: str, number: int):
            """Get a single pull request by number for a given repository."""
            # Try cache first
            cached = next(
                (
                    pr
                    for pr in self.state.docs_cache[DOCUMENTATION.PRS].get(repo, [])
                    if pr[GITHUB_DEFS.NUMBER] == number
                ),
                None,
            )
            if cached:
                return cached
            # Fallback to live fetch
            return await documentation_utils.get_issue(repo, number)

        # Register tools
        @mcp.tool()
        async def search_documentation(
            query: str,
            search_type: str = SEARCH_TYPES.SEMANTIC,
            n_results: int = 5,
            max_exact_results: int = 20,
        ):
            self._check_initialized()
            """
            Search all Napistu project documentation with intelligent search strategy.

            Provides flexible search capabilities for finding relevant Napistu documentation
            using either AI-powered semantic search for natural language queries or exact text
            matching for precise keyword searches. Covers README files, wiki pages, GitHub
            issues, and pull requests from the Napistu project.

            **USE THIS WHEN:**
            - Looking for Napistu project information, setup instructions, or usage documentation
            - Finding README content, wiki pages, GitHub issues, or pull requests
            - Researching Napistu-specific concepts, workflows, troubleshooting, or features
            - Understanding Napistu installation, configuration, or implementation details

            **DO NOT USE FOR:**
            - General bioinformatics concepts not specific to Napistu
            - Documentation for other tools, libraries, or frameworks
            - Academic literature or research papers
            - Programming concepts unrelated to Napistu implementation
            - Questions about biological concepts that don't involve Napistu usage

            **EXAMPLE APPROPRIATE QUERIES:**
            - "how to install Napistu"
            - "SBML file processing with Napistu"
            - "consensus network creation"
            - "troubleshooting pathway integration"
            - "GitHub issues about data ingestion"

            **EXAMPLE INAPPROPRIATE QUERIES:**
            - "what is systems biology" (too general, not Napistu-specific)
            - "how to use pandas" (not related to Napistu)
            - "latest research in pathway analysis" (academic, not implementation)
            - "machine learning algorithms" (unless specifically about Napistu ML features)

            Parameters
            ----------
            query : str
                Search term or natural language question about Napistu. Should be
                specific to Napistu project documentation, features, or implementation.
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
                - results : List[Dict] or Dict[str, List]
                    Search results. Format depends on search type:
                    - Semantic: List of result dictionaries with content, metadata, source, similarity_score
                    - Exact: Dictionary organized by content type (readme, wiki, issues, prs)
                - tip : str
                    Helpful guidance for improving search results

            Examples
            --------
            Natural language semantic search for Napistu concepts:

            >>> results = await search_documentation("how to install Napistu")
            >>> print(results["search_type"])  # "semantic"
            >>> for result in results["results"]:
            ...     score = result['similarity_score']
            ...     print(f"Score: {score:.3f} - Found in: {result['source']}")

            Exact keyword search for specific Napistu terms:

            >>> results = await search_documentation("installation", search_type="exact")
            >>> print(len(results["results"]["readme"]))  # Number of matching READMEs

            Notes
            -----
            **CONTENT SCOPE:**
            This tool searches only Napistu project documentation including:
            - Official README files from Napistu repositories
            - Napistu project wiki pages with implementation details
            - GitHub issues and pull requests for troubleshooting and feature discussions
            - Package documentation specific to Napistu functionality

            **SEARCH TYPE GUIDANCE:**
            - Use semantic (default) for conceptual queries and natural language questions
            - Use exact for precise Napistu function names, error messages, or known keywords

            **RESULT INTERPRETATION:**
            - Semantic results include similarity scores (0.8-1.0 = very relevant)
            - Results may include chunked sections from long documents for precision
            - Multiple related sections may appear for comprehensive coverage

            The function automatically handles semantic search failures by falling back
            to exact search, ensuring reliable results even if AI components are unavailable.
            """
            if search_type == SEARCH_TYPES.SEMANTIC and self.state.semantic_search:
                # Use semantic search
                results = self.state.semantic_search.search(
                    query, MCP_COMPONENTS.DOCUMENTATION, n_results=n_results
                )
                return {
                    SEARCH_RESULT_DEFS.QUERY: query,
                    SEARCH_RESULT_DEFS.SEARCH_TYPE: SEARCH_TYPES.SEMANTIC,
                    SEARCH_RESULT_DEFS.RESULTS: results,
                    SEARCH_RESULT_DEFS.TIP: "Try different phrasings if results aren't relevant, or use search_type='exact' for precise keyword matching",
                }
            else:
                # Fall back to exact search
                return documentation_utils._exact_search_documentation(
                    query, self.state.docs_cache, max_exact_results
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
                "DocumentationComponent is still initializing. "
                "This component loads documentation from GitHub and other sources, which may take several minutes. "
                "Please wait a moment and try again. "
                "You can check initialization status using the health check endpoint."
            )
        elif self.state.initialization_error:
            raise RuntimeError(
                f"DocumentationComponent failed to initialize: {self.state.initialization_error}. "
                "This component requires documentation content to function. "
                "Please check the server logs for details or try again later."
            )

    def _create_state(self) -> DocumentationState:
        """
        Create documentation-specific state instance.

        Returns
        -------
        DocumentationState
            New state instance for managing documentation content and semantic search
        """
        return DocumentationState()

    async def _initialize_semantic_search(self) -> bool:
        """
        Index documentation content into the shared semantic search instance.

        Uses the shared semantic search instance (stored in self.state.semantic_search)
        to index this component's content into the appropriate collection.

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

            logger.info("Indexing documentation content for semantic search...")

            # Index content using the shared instance stored in component state
            self.state.semantic_search.index_content(
                MCP_COMPONENTS.DOCUMENTATION, self.state.docs_cache
            )

            logger.info("✅ Documentation content indexed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to index documentation content: {e}")
            return False


# Module-level component instance
_component = DocumentationComponent()


def get_component() -> DocumentationComponent:
    """
    Get the documentation component instance.

    Returns
    -------
    DocumentationComponent
        Singleton documentation component instance for use across the MCP server.
        The same instance is returned on every call to ensure consistent state.

    Notes
    -----
    This function provides the standard interface for accessing the documentation
    component. The component must be initialized via safe_initialize() before use.
    """
    return _component
