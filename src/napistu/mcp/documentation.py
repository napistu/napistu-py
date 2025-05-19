"""
Documentation components for the Napistu MCP server.
"""

from typing import Dict, List, Any, Optional
import os
import json

from napistu.mcp import documentation_utils
from napistu.mcp.constants import READMES

# Global cache for documentation content
_docs_cache = {
    "readme": {},
    "readthedocs": {},
    "wiki": {},
    "packagedown": {},
}

async def initialize_components(mcp, config):
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
    
    # Return True to indicate successful initialization
    return True

def register_components(mcp, docs_paths=None):
    """
    Register documentation components with the MCP server.
    
    Args:
        mcp: FastMCP server instance
        docs_paths: List of paths to documentation files
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