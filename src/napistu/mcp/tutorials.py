"""
Tutorial components for the Napistu MCP server.
"""

from typing import Dict, List, Any
import logging

from fastmcp import FastMCP

from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp import tutorials_utils
from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import TUTORIAL_URLS

logger = logging.getLogger(__name__)


class TutorialsState(ComponentState):
    """State management for tutorials component."""

    def __init__(self):
        super().__init__()
        self.tutorials: Dict[str, str] = {}

    def is_healthy(self) -> bool:
        """Component is healthy if it has loaded tutorials."""
        return bool(self.tutorials)

    def get_health_details(self) -> Dict[str, Any]:
        """Provide tutorials-specific health details."""
        return {
            "tutorial_count": len(self.tutorials),
            "tutorial_ids": list(self.tutorials.keys()),
        }


class TutorialsComponent(MCPComponent):
    """MCP component for tutorial management and search."""

    def _create_state(self) -> TutorialsState:
        """Create tutorials-specific state."""
        return TutorialsState()

    async def initialize(self) -> bool:
        """
        Initialize tutorials component by loading all tutorials.

        Returns
        -------
        bool
            True if at least one tutorial was loaded successfully
        """
        tutorials_loaded = 0

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

        # Consider successful if at least one tutorial loaded
        return tutorials_loaded > 0

    def register(self, mcp: FastMCP) -> None:
        """
        Register tutorial resources and tools with the MCP server.

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance
        """

        # Register resources
        @mcp.resource("napistu://tutorials/index")
        async def get_tutorials_index() -> List[Dict[str, Any]]:
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

            for tutorial_id, content in self.state.tutorials.items():
                if query.lower() in content.lower():
                    results.append(
                        {
                            "id": tutorial_id,
                            "snippet": mcp_utils.get_snippet(content, query),
                        }
                    )

            return results


# Module-level component instance
_component = TutorialsComponent()


def get_component() -> TutorialsComponent:
    """
    Get the tutorials component instance.

    Returns
    -------
    TutorialsComponent
        The tutorials component instance
    """
    return _component
