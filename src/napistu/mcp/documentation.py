"""
Documentation components for the Napistu MCP server.
"""

from typing import Dict, Any
import logging

from fastmcp import FastMCP

from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp import documentation_utils
from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import DOCUMENTATION, READMES, REPOS_WITH_ISSUES, WIKI_PAGES

logger = logging.getLogger(__name__)


class DocumentationState(ComponentState):
    """State management for documentation component."""

    def __init__(self):
        super().__init__()
        self.docs_cache: Dict[str, Dict[str, Any]] = {
            DOCUMENTATION.README: {},
            DOCUMENTATION.WIKI: {},
            DOCUMENTATION.ISSUES: {},
            DOCUMENTATION.PRS: {},
            DOCUMENTATION.PACKAGEDOWN: {},
        }

    def is_healthy(self) -> bool:
        """Component is healthy if it has loaded any documentation."""
        return any(bool(section) for section in self.docs_cache.values())

    def get_health_details(self) -> Dict[str, Any]:
        """Provide documentation-specific health details."""
        return {
            "readme_count": len(self.docs_cache[DOCUMENTATION.README]),
            "wiki_pages": len(self.docs_cache[DOCUMENTATION.WIKI]),
            "issues_repos": len(self.docs_cache[DOCUMENTATION.ISSUES]),
            "prs_repos": len(self.docs_cache[DOCUMENTATION.PRS]),
            "total_sections": sum(len(section) for section in self.docs_cache.values()),
        }


class DocumentationComponent(MCPComponent):
    """MCP component for documentation management and search."""

    def _create_state(self) -> DocumentationState:
        """Create documentation-specific state."""
        return DocumentationState()

    async def initialize(self) -> bool:
        """
        Initialize documentation component by loading all documentation sources.

        Returns
        -------
        bool
            True if at least some documentation was loaded successfully
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

        # Consider successful if at least some documentation loaded
        return success_count > 0

    def register(self, mcp: FastMCP) -> None:
        """
        Register documentation resources and tools with the MCP server.

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance
        """

        # Register resources
        @mcp.resource("napistu://documentation/summary")
        async def get_documentation_summary():
            """Get a summary of all available documentation."""
            return {
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
        async def search_documentation(query: str):
            """
            Search all documentation for a specific query.

            Args:
                query: Search term

            Returns:
                Dictionary with search results organized by documentation type
            """
            results = {
                DOCUMENTATION.README: [],
                DOCUMENTATION.WIKI: [],
                DOCUMENTATION.ISSUES: [],
                DOCUMENTATION.PRS: [],
                DOCUMENTATION.PACKAGEDOWN: [],
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
            for page_name, content in self.state.docs_cache[DOCUMENTATION.WIKI].items():
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

            return results


# Module-level component instance
_component = DocumentationComponent()


def get_component() -> DocumentationComponent:
    """
    Get the documentation component instance.

    Returns
    -------
    DocumentationComponent
        The documentation component instance
    """
    return _component
