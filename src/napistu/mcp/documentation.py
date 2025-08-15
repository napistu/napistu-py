"""
Documentation components for the Napistu MCP server.
"""

from typing import Dict, Any
import logging

from fastmcp import FastMCP

from napistu.mcp import documentation_utils
from napistu.mcp import utils as mcp_utils
from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp.semantic_search import SemanticSearch
from napistu.mcp.constants import DOCUMENTATION, READMES, REPOS_WITH_ISSUES, WIKI_PAGES

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
    semantic_indexed : bool
        Whether documentation content has been indexed for semantic search

    Examples
    --------
    >>> state = DocumentationState()
    >>> state.docs_cache["readme"]["install"] = "Installation guide..."
    >>> print(state.is_healthy())  # True if any content loaded
    >>> health = state.get_health_details()
    >>> print(health["semantic_search_available"])  # False initially
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
        self.semantic_indexed = False

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
            - semantic_search_available : bool
                Whether semantic search has been initialized
            - semantic_indexed : bool
                Whether content has been indexed for semantic search

        Examples
        --------
        >>> state = DocumentationState()
        >>> # ... load content ...
        >>> details = state.get_health_details()
        >>> print(f"Total content items: {details['total_sections']}")
        >>> print(f"Semantic search ready: {details['semantic_indexed']}")
        """
        base_details = {
            "readme_count": len(self.docs_cache[DOCUMENTATION.README]),
            "wiki_pages": len(self.docs_cache[DOCUMENTATION.WIKI]),
            "issues_repos": len(self.docs_cache[DOCUMENTATION.ISSUES]),
            "prs_repos": len(self.docs_cache[DOCUMENTATION.PRS]),
            "total_sections": sum(len(section) for section in self.docs_cache.values()),
        }

        # Add semantic search status
        base_details.update(
            {
                "semantic_search_available": self.semantic_search is not None,
                "semantic_indexed": self.semantic_indexed,
            }
        )

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

    def _create_state(self) -> DocumentationState:
        """
        Create documentation-specific state instance.

        Returns
        -------
        DocumentationState
            New state instance for managing documentation content and semantic search
        """
        return DocumentationState()

    async def initialize(self) -> bool:
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
        if content_loaded:
            semantic_success = await self._initialize_semantic_search()
            logger.info(
                f"Semantic search initialization: {'✅ Success' if semantic_success else '⚠️ Failed'}"
            )

        return content_loaded

    async def _initialize_semantic_search(self) -> bool:
        """
        Initialize semantic search engine and index documentation content.

        Creates a SemanticSearch instance, sets up ChromaDB collections, and indexes
        all loaded documentation content for AI-powered search capabilities.

        Returns
        -------
        bool
            True if semantic search was successfully initialized and content indexed,
            False if initialization failed

        Notes
        -----
        Failure to initialize semantic search is not considered a critical error.
        The component continues to function with exact text search if semantic
        search initialization fails.

        The method updates state.semantic_search and state.semantic_indexed to
        track semantic search availability for health monitoring.

        Raises
        ------
        Exception
            Any exception during semantic search setup is caught and logged,
            ensuring component initialization continues
        """
        try:
            logger.info("Initializing semantic search...")

            # Create semantic search instance
            self.state.semantic_search = SemanticSearch()

            # Index the documentation content
            logger.info("Indexing documentation content for semantic search...")
            self.state.semantic_search.index_content(
                "documentation", self.state.docs_cache
            )

            self.state.semantic_indexed = True
            logger.info("✅ Semantic search initialized and content indexed")
            return True

        except Exception as e:
            logger.error(f"❌ Semantic search initialization failed: {e}")
            # Don't fail the entire component if semantic search fails
            self.state.semantic_search = None
            self.state.semantic_indexed = False
            return False

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
            Get a comprehensive summary of all available documentation.

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - readme_files : List[str]
                    Names of loaded README files
                - issues : List[str]
                    Repository names with loaded issues
                - prs : List[str]
                    Repository names with loaded pull requests
                - wiki_pages : List[str]
                    Names of loaded wiki pages
                - packagedown_sections : List[str]
                    Names of package documentation sections
                - semantic_search : Dict[str, bool]
                    Status of semantic search availability and indexing

            Examples
            --------
            Resource provides overview of all documentation content and capabilities
            for clients to understand what information is available.
            """
            summary = {
                "readme_files": list(
                    self.state.docs_cache[DOCUMENTATION.README].keys()
                ),
                "issues": list(self.state.docs_cache[DOCUMENTATION.ISSUES].keys()),
                "prs": list(self.state.docs_cache[DOCUMENTATION.PRS].keys()),
                "wiki_pages": list(self.state.docs_cache[DOCUMENTATION.WIKI].keys()),
                "packagedown_sections": list(
                    self.state.docs_cache[DOCUMENTATION.PACKAGEDOWN].keys()
                ),
            }

            # Add semantic search status
            summary["semantic_search"] = {
                "available": self.state.semantic_search is not None,
                "indexed": self.state.semantic_indexed,
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
                    if i["number"] == number
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
                    if pr["number"] == number
                ),
                None,
            )
            if cached:
                return cached
            # Fallback to live fetch
            return await documentation_utils.get_issue(repo, number)

        # Register tools
        @mcp.tool()
        async def search_documentation(query: str, search_type: str = "semantic"):
            """
            Search all documentation with intelligent search strategy.

            Provides flexible search capabilities using either AI-powered semantic search
            for natural language queries or exact text matching for precise keyword searches.
            Automatically falls back to exact search if semantic search is unavailable.

            Parameters
            ----------
            query : str
                Search term or natural language question. For semantic search, can be
                full questions like "how to install napistu". For exact search, use
                specific keywords like "installation" or "consensus".
            search_type : str, optional
                Search strategy to use:
                - "semantic" (default): AI-powered search using embeddings
                - "exact": Traditional text matching search
                Default is "semantic".

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
                    - Semantic: List of result dictionaries with content, metadata, source
                    - Exact: Dictionary organized by content type (readme, wiki, issues, prs)
                - tip : str
                    Helpful guidance for improving search results

            Examples
            --------
            Natural language semantic search:

            >>> results = await search_documentation("how to install napistu")
            >>> print(results["search_type"])  # "semantic"
            >>> for result in results["results"]:
            ...     print(f"Found in: {result['source']}")

            Exact keyword search:

            >>> results = await search_documentation("installation", search_type="exact")
            >>> print(len(results["results"]["readme"]))  # Number of matching READMEs

            Notes
            -----
            **WHEN TO USE:**
            - Finding information about Napistu project features, setup, or usage
            - Looking for README content, wiki pages, GitHub issues, or pull requests
            - Researching Napistu-specific concepts, workflows, or troubleshooting

            **SEARCH TYPE GUIDANCE:**
            - Use semantic (default) for conceptual queries and natural language
            - Use exact for precise terms, function names, or known keywords

            **EXAMPLE QUERIES:**
            - Semantic: "consensus network creation", "SBML file processing", "pathway integration"
            - Exact: "installation", "consensus", "SBML", "create_graph"

            The function automatically handles semantic search failures by falling back
            to exact search, ensuring reliable results even if AI components are unavailable.
            """
            if search_type == "semantic" and self.state.semantic_search:
                # Use semantic search
                results = self.state.semantic_search.search(
                    query, "documentation", n_results=5
                )
                return {
                    "query": query,
                    "search_type": "semantic",
                    "results": results,
                    "tip": "Try different phrasings if results aren't relevant, or use search_type='exact' for precise keyword matching",
                }
            else:
                # Fall back to exact search (existing logic)
                results = {
                    DOCUMENTATION.README: [],
                    DOCUMENTATION.WIKI: [],
                    DOCUMENTATION.ISSUES: [],
                    DOCUMENTATION.PRS: [],
                }

                # Search README files
                for readme_name, content in self.state.docs_cache[
                    DOCUMENTATION.README
                ].items():
                    if query.lower() in content.lower():
                        results[DOCUMENTATION.README].append(
                            {
                                "name": readme_name,
                                "snippet": mcp_utils.get_snippet(content, query),
                            }
                        )

                # Search wiki pages
                for page_name, content in self.state.docs_cache[
                    DOCUMENTATION.WIKI
                ].items():
                    if query.lower() in content.lower():
                        results[DOCUMENTATION.WIKI].append(
                            {
                                "name": page_name,
                                "snippet": mcp_utils.get_snippet(content, query),
                            }
                        )

                # Search issues
                for repo, issues in self.state.docs_cache[DOCUMENTATION.ISSUES].items():
                    for issue in issues:
                        issue_text = f"{issue.get('title', '')} {issue.get('body', '')}"
                        if query.lower() in issue_text.lower():
                            results[DOCUMENTATION.ISSUES].append(
                                {
                                    "name": f"{repo}#{issue.get('number')}",
                                    "title": issue.get("title"),
                                    "url": issue.get("url"),
                                    "snippet": mcp_utils.get_snippet(issue_text, query),
                                }
                            )

                # Search PRs
                for repo, prs in self.state.docs_cache[DOCUMENTATION.PRS].items():
                    for pr in prs:
                        pr_text = f"{pr.get('title', '')} {pr.get('body', '')}"
                        if query.lower() in pr_text.lower():
                            results[DOCUMENTATION.PRS].append(
                                {
                                    "name": f"{repo}#{pr.get('number')}",
                                    "title": pr.get("title"),
                                    "url": pr.get("url"),
                                    "snippet": mcp_utils.get_snippet(pr_text, query),
                                }
                            )

                return {
                    "query": query,
                    "search_type": "exact",
                    "results": results,
                    "tip": "Use search_type='semantic' for natural language queries",
                }


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
