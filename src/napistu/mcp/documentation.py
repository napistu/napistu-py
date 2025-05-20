"""
Documentation components for the Napistu MCP server.
"""

from typing import Dict, List, Any, Optional
import os
import json

from fastmcp import FastMCP

from napistu.mcp import documentation_utils
from napistu.mcp import github

from napistu.mcp.constants import READMES
from napistu.mcp.constants import REPOS_WITH_ISSUES

# Global cache for documentation content
_docs_cache = {
    "readme": {},
    "wiki": {},
    "issues": {},
    "prs": {},
    "packagedown": {},
}

async def initialize_components(mcp: FastMCP, config: dict) -> bool:
    """
    Initialize documentation components.
    
    This function loads documentation content and should be called before
    the server starts handling requests.
    
    Args:
        mcp: FastMCP server instance
        config: Server configuration dictionary
    """
    global _docs_cache
    
    # Load documentation from the READMES dict
    for name, url in READMES.items():
        _docs_cache["readme"][name] = await documentation_utils.load_readme_content(url)
    
    # Load issue and PR summaries with the GitHub API
    for repo in REPOS_WITH_ISSUES:
        _docs_cache["issues"][repo] = await github.list_issues(repo)
        _docs_cache["prs"][repo] = await github.list_pull_requests(repo)

    # Return True to indicate successful initialization
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
            "readme_files": list(_docs_cache["readme"].keys()),
            "readthedocs_sections": list(_docs_cache["readthedocs"].keys()),
            "wiki_pages": list(_docs_cache["wiki"].keys()),
            "packagedown_sections": list(_docs_cache["packagedown"].keys()),
        }
    
    @mcp.resource("napistu://documentation/readme/{file_name}")
    async def get_readme_content(file_name: str):
        """
        Get the content of a specific README file.
        
        Args:
            file_name: Name of the README file
        """
        if file_name not in _docs_cache["readme"]:
            return {"error": f"README file {file_name} not found"}
        
        return {
            "content": _docs_cache["readme"][file_name],
            "format": "markdown",
        }

    @mcp.resource("napistu://documentation/readthedocs/{module_name}")
    async def get_readthedocs_module(module_name: str):
        """
        Get the parsed ReadTheDocs documentation for a specific module.
        Args:
            module_name: Name of the module (as parsed from the docs tree)
        """
        docs = _docs_cache["readthedocs"]
        if module_name not in docs:
            return {"error": f"Module {module_name} not found in ReadTheDocs documentation."}
        return docs[module_name]

    @mcp.resource("napistu://documentation/readthedocs")
    async def get_readthedocs_tree():
        """
        Get the full parsed ReadTheDocs documentation tree.
        """
        return _docs_cache["readthedocs"]

    @mcp.resource("napistu://documentation/issues/{repo}")
    async def get_issues(repo: str):
        """
        Get the list of issues for a given repository.
        """
        return _docs_cache["issues"].get(repo, [])

    @mcp.resource("napistu://documentation/prs/{repo}")
    async def get_prs(repo: str):
        """
        Get the list of pull requests for a given repository.
        """
        return _docs_cache["prs"].get(repo, [])

    @mcp.resource("napistu://documentation/issue/{repo}/{number}")
    async def get_issue_resource(repo: str, number: int):
        """
        Get a single issue by number for a given repository.
        """
        # Try cache first
        cached = next((i for i in _docs_cache["issues"].get(repo, []) if i["number"] == number), None)
        if cached:
            return cached
        # Fallback to live fetch
        return await github.get_issue(repo, number)

    @mcp.resource("napistu://documentation/pr/{repo}/{number}")
    async def get_pr_resource(repo: str, number: int):
        """
        Get a single pull request by number for a given repository.
        """
        # Try cache first
        cached = next((pr for pr in _docs_cache["prs"].get(repo, []) if pr["number"] == number), None)
        if cached:
            return cached
        # Fallback to live fetch
        return await github.get_issue(repo, number)

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
            "readme": [],
            "readthedocs": [],
            "wiki": [],
            "packagedown": [],
        }
        # Simple text search
        for readme_name, content in _docs_cache["readme"].items():
            if query.lower() in content.lower():
                results["readme"].append({
                    "name": readme_name,
                    "snippet": documentation_utils.get_snippet(content, query),
                })
        return results