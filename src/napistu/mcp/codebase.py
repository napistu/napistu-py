"""
Codebase exploration components for the Napistu MCP server.
"""

from napistu.mcp.constants import NAPISTU_PY_READTHEDOCS_API

from fastmcp import FastMCP

from typing import Dict, Any
import json

from napistu.mcp import codebase_utils
from napistu.mcp import utils as mcp_utils

# Global cache for codebase information
_codebase_cache = {
    "modules": {},
    "classes": {},
    "functions": {},
}


async def initialize_components() -> bool:
    """
    Initialize codebase components.

    Returns
    -------
    bool
        True if initialization is successful.
    """
    global _codebase_cache
    # Load documentation from the ReadTheDocs API
    _codebase_cache["modules"] = await codebase_utils.read_read_the_docs(
        NAPISTU_PY_READTHEDOCS_API
    )
    # Extract functions and classes from the modules
    _codebase_cache["functions"], _codebase_cache["classes"] = (
        codebase_utils.extract_functions_and_classes_from_modules(
            _codebase_cache["modules"]
        )
    )
    return True


def register_components(mcp: FastMCP):
    """
    Register codebase components with the MCP server.

    Args:
        mcp: FastMCP server instance
    """
    global _codebase_cache

    # Register resources
    @mcp.resource("napistu://codebase/summary")
    async def get_codebase_summary():
        """
        Get a summary of all available codebase information.
        """
        return {
            "modules": list(_codebase_cache["modules"].keys()),
            "classes": list(_codebase_cache["classes"].keys()),
            "functions": list(_codebase_cache["functions"].keys()),
        }

    @mcp.resource("napistu://codebase/modules/{module_name}")
    async def get_module_details(module_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific module.

        Args:
            module_name: Name of the module
        """
        if module_name not in _codebase_cache["modules"]:
            return {"error": f"Module {module_name} not found"}

        return _codebase_cache["modules"][module_name]

    # Register tools
    @mcp.tool()
    async def search_codebase(query: str) -> Dict[str, Any]:
        """
        Search the codebase for a specific query.

        Args:
            query: Search term

        Returns:
            Dictionary with search results organized by code element type, including snippets for context.
        """
        results = {
            "modules": [],
            "classes": [],
            "functions": [],
        }

        # Search modules
        for module_name, info in _codebase_cache["modules"].items():
            # Use docstring or description for snippet
            doc = info.get("doc") or info.get("description") or ""
            module_text = json.dumps(info)
            if query.lower() in module_text.lower():
                snippet = mcp_utils.get_snippet(doc, query)
                results["modules"].append(
                    {
                        "name": module_name,
                        "description": doc,
                        "snippet": snippet,
                    }
                )

        # Search classes
        for class_name, info in _codebase_cache["classes"].items():
            doc = info.get("doc") or info.get("description") or ""
            class_text = json.dumps(info)
            if query.lower() in class_text.lower():
                snippet = mcp_utils.get_snippet(doc, query)
                results["classes"].append(
                    {
                        "name": class_name,
                        "description": doc,
                        "snippet": snippet,
                    }
                )

        # Search functions
        for func_name, info in _codebase_cache["functions"].items():
            doc = info.get("doc") or info.get("description") or ""
            func_text = json.dumps(info)
            if query.lower() in func_text.lower():
                snippet = mcp_utils.get_snippet(doc, query)
                results["functions"].append(
                    {
                        "name": func_name,
                        "description": doc,
                        "signature": info.get("signature", ""),
                        "snippet": snippet,
                    }
                )

        return results

    @mcp.tool()
    async def get_function_documentation(function_name: str) -> Dict[str, Any]:
        """
        Get detailed documentation for a specific function.

        Args:
            function_name: Name of the function

        Returns:
            Dictionary with function documentation
        """
        if function_name not in _codebase_cache["functions"]:
            return {"error": f"Function {function_name} not found"}

        return _codebase_cache["functions"][function_name]

    @mcp.tool()
    async def get_class_documentation(class_name: str) -> Dict[str, Any]:
        """
        Get detailed documentation for a specific class.

        Args:
            class_name: Name of the class

        Returns:
            Dictionary with class documentation
        """
        if class_name not in _codebase_cache["classes"]:
            return {"error": f"Class {class_name} not found"}

        return _codebase_cache["classes"][class_name]
