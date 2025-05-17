"""
Tutorial components for the Napistu MCP server.
"""

from typing import Dict, List, Any, Optional
from mcp import tool, resource

from napistu.mcp.utils import tutorials as tutorials_utils

# Global cache for tutorial content
_tutorial_cache = {
    "index": [],
    "content": {},
}

def register_components(mcp, tutorials_path=None):
    """
    Register tutorial components with the MCP server.
    
    Args:
        mcp: FastMCP server instance
        tutorials_path: Path to the tutorials directory
    """
    global _tutorial_cache
    
    # Load tutorials if path provided
    if tutorials_path:
        async def _load_tutorials():
            nonlocal _tutorial_cache
            tutorial_index = await tutorials_utils.load_tutorial_index(tutorials_path)
            _tutorial_cache["index"] = tutorial_index
            
            # Preload tutorial content
            for tutorial in tutorial_index:
                tutorial_id = tutorial["id"]
                content = await tutorials_utils.get_tutorial_content(tutorials_path, tutorial_id)
                _tutorial_cache["content"][tutorial_id] = content
        
        # Schedule tutorial loading
        import asyncio
        asyncio.create_task(_load_tutorials())
    
    # Register resources
    @mcp.resource("napistu://tutorials/index")
    async def get_tutorial_index() -> List[Dict[str, Any]]:
        """
        Get the index of all available tutorials.
        """
        return _tutorial_cache["index"]
    
    @mcp.resource("napistu://tutorials/content/{tutorial_id}")
    async def get_tutorial_content_resource(tutorial_id: str) -> Dict[str, Any]:
        """
        Get the content of a specific tutorial.
        
        Args:
            tutorial_id: ID of the tutorial
        """
        if tutorial_id not in _tutorial_cache["content"]:
            return {"error": f"Tutorial {tutorial_id} not found"}
        
        return {
            "content": _tutorial_cache["content"][tutorial_id],
            "format": "jupyter_notebook",
        }
    
    # Register tools
    @mcp.tool()
    async def search_tutorials(query: str) -> List[Dict[str, Any]]:
        """
        Search tutorials for a specific query.
        
        Args:
            query: Search term
        
        Returns:
            List of matching tutorials with metadata
        """
        results = []
        
        # Search tutorial index
        for tutorial in _tutorial_cache["index"]:
            tutorial_id = tutorial["id"]
            tutorial_content = _tutorial_cache["content"].get(tutorial_id, "")
            
            # Check if query matches title, description, or content
            if (query.lower() in tutorial["title"].lower() or
                query.lower() in tutorial.get("description", "").lower() or
                query.lower() in tutorial_content.lower()):
                results.append({
                    "id": tutorial_id,
                    "title": tutorial["title"],
                    "description": tutorial.get("description", ""),
                    "snippet": tutorials_utils.get_snippet(tutorial_content, query) if tutorial_content else "",
                })
        
        return results