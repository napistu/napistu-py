"""
Tutorial components for the Napistu MCP server.
"""

from typing import Dict, List, Any
import logging

from fastmcp import FastMCP

from napistu.mcp import tutorials_utils
from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import TUTORIALS
from napistu.mcp.constants import TUTORIAL_URLS
from napistu.mcp.constants import TOOL_VARS

# Global cache for tutorial content
_tutorial_cache: Dict[str, Dict[str, str]] = {
    TUTORIALS.TUTORIALS: {},
}

logger = logging.getLogger(__name__)


async def initialize_components() -> bool:
    """
    Initialize tutorial components by preloading all tutorials into the cache.

    Returns
    -------
    bool
        True if initialization is successful.
    """
    global _tutorial_cache
    for k, v in TUTORIAL_URLS.items():
        _tutorial_cache[TUTORIALS.TUTORIALS][k] = (
            await tutorials_utils.get_tutorial_markdown(k)
        )
    return True


def register_components(mcp: FastMCP) -> None:
    """
    Register tutorial components with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        FastMCP server instance.
    """

    # Register resources
    @mcp.resource("napistu://tutorials/index")
    async def get_tutorial_index() -> List[Dict[str, Any]]:
        """
        Get the index of all available tutorials.

        Returns
        -------
        List[dict]
            List of dictionaries with tutorial IDs and URLs.
        """
        return [
            {"id": tutorial_id, "url": url}
            for tutorial_id, url in TUTORIAL_URLS.items()
        ]

    @mcp.resource("napistu://tutorials/content/{tutorial_id}")
    async def get_tutorial_content_resource(tutorial_id: str) -> Dict[str, Any]:
        """
        Get the content of a specific tutorial as markdown.

        Parameters
        ----------
        tutorial_id : str
            ID of the tutorial.

        Returns
        -------
        dict
            Dictionary with markdown content and format.

        Raises
        ------
        Exception
            If the tutorial cannot be loaded.
        """
        content = _tutorial_cache[TUTORIALS.TUTORIALS].get(tutorial_id)
        if content is None:
            try:
                content = await tutorials_utils.get_tutorial_markdown(tutorial_id)
                _tutorial_cache[TUTORIALS.TUTORIALS][tutorial_id] = content
            except Exception as e:
                logger.error(f"Tutorial {tutorial_id} could not be loaded: {e}")
                raise
        return {
            "content": content,
            "format": "markdown",
        }

    @mcp.tool()
    async def search_tutorials(query: str) -> List[Dict[str, Any]]:
        """
        Search tutorials for a specific query.

        Parameters
        ----------
        query : str
            Search term.

        Returns
        -------
        List[dict]
            List of matching tutorials with metadata and snippet.
        """
        results: List[Dict[str, Any]] = []
        for tutorial_id, content in _tutorial_cache[TUTORIALS.TUTORIALS].items():
            if query.lower() in content.lower():
                results.append(
                    {
                        TOOL_VARS.ID: tutorial_id,
                        TOOL_VARS.SNIPPET: mcp_utils.get_snippet(content, query),
                    }
                )
        return results
