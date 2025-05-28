"""
Documentation components for the Napistu MCP server.
"""

from fastmcp import FastMCP

from napistu.mcp import documentation_utils
from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import DOCUMENTATION, READMES, REPOS_WITH_ISSUES

# Global cache for documentation content
_docs_cache = {
    DOCUMENTATION.README: {},
    DOCUMENTATION.WIKI: {},
    DOCUMENTATION.ISSUES: {},
    DOCUMENTATION.PRS: {},
    DOCUMENTATION.PACKAGEDOWN: {},
}


async def initialize_components() -> bool:
    """
    Initialize documentation components.
    Loads documentation content and should be called before the server starts handling requests.

    Returns
    -------
    bool
        True if initialization is successful.
    """
    global _docs_cache
    # Load documentation from the READMES dict
    for name, url in READMES.items():
        _docs_cache[DOCUMENTATION.README][name] = (
            await documentation_utils.load_readme_content(url)
        )
    # Load issue and PR summaries with the GitHub API
    for repo in REPOS_WITH_ISSUES:
        _docs_cache[DOCUMENTATION.ISSUES][repo] = await documentation_utils.list_issues(
            repo
        )
        _docs_cache[DOCUMENTATION.PRS][repo] = (
            await documentation_utils.list_pull_requests(repo)
        )
    return True


def register_components(mcp: FastMCP):
    """
    Register documentation components with the MCP server.

    Args:
        mcp: FastMCP server instance
    """

    # Register resources
    @mcp.resource("napistu://documentation/summary")
    async def get_documentation_summary():
        """
        Get a summary of all available documentation.
        """
        return {
            "readme_files": list(_docs_cache[DOCUMENTATION.README].keys()),
            "issues": list(_docs_cache[DOCUMENTATION.ISSUES].keys()),
            "prs": list(_docs_cache[DOCUMENTATION.PRS].keys()),
            "wiki_pages": list(_docs_cache[DOCUMENTATION.WIKI].keys()),
            "packagedown_sections": list(_docs_cache[DOCUMENTATION.PACKAGEDOWN].keys()),
        }

    @mcp.resource("napistu://documentation/readme/{file_name}")
    async def get_readme_content(file_name: str):
        """
        Get the content of a specific README file.

        Args:
            file_name: Name of the README file
        """
        if file_name not in _docs_cache[DOCUMENTATION.README]:
            return {"error": f"README file {file_name} not found"}

        return {
            "content": _docs_cache[DOCUMENTATION.README][file_name],
            "format": "markdown",
        }

    @mcp.resource("napistu://documentation/issues/{repo}")
    async def get_issues(repo: str):
        """
        Get the list of issues for a given repository.
        """
        return _docs_cache[DOCUMENTATION.ISSUES].get(repo, [])

    @mcp.resource("napistu://documentation/prs/{repo}")
    async def get_prs(repo: str):
        """
        Get the list of pull requests for a given repository.
        """
        return _docs_cache[DOCUMENTATION.PRS].get(repo, [])

    @mcp.resource("napistu://documentation/issue/{repo}/{number}")
    async def get_issue_resource(repo: str, number: int):
        """
        Get a single issue by number for a given repository.
        """
        # Try cache first
        cached = next(
            (
                i
                for i in _docs_cache[DOCUMENTATION.ISSUES].get(repo, [])
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
        """
        Get a single pull request by number for a given repository.
        """
        # Try cache first
        cached = next(
            (
                pr
                for pr in _docs_cache[DOCUMENTATION.PRS].get(repo, [])
                if pr["number"] == number
            ),
            None,
        )
        if cached:
            return cached
        # Fallback to live fetch
        return await documentation_utils.get_issue(repo, number)

    # Register tools
    @mcp.tool("search_documentation")
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
        # Simple text search
        for readme_name, content in _docs_cache[DOCUMENTATION.README].items():
            if query.lower() in content.lower():
                results[DOCUMENTATION.README].append(
                    {
                        "name": readme_name,
                        "snippet": mcp_utils.get_snippet(content, query),
                    }
                )
        return results
