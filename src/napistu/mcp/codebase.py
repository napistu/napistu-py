"""
Codebase exploration components for the Napistu MCP server.
"""

from napistu.mcp.constants import NAPISTU_PY_READTHEDOCS_API

from fastmcp import FastMCP

from typing import Dict, List, Any, Optional
import json

from napistu.mcp import codebase_utils

# Global cache for codebase information
_codebase_cache = {
    "modules": {},
    "classes": {},
    "functions": {},
}


async def initialize_components(mcp: FastMCP) -> bool:
    """
    Initialize codebase components.
    """
    global _codebase_cache
    
    # Load documentation from the ReadTheDocs API
    _codebase_cache["modules"] = await codebase_utils.read_read_the_docs(NAPISTU_PY_READTHEDOCS_API)

    # Extract functions and classes from the modules
    _codebase_cache["functions"], _codebase_cache["classes"] = codebase_utils.extract_functions_and_classes_from_modules(_codebase_cache["modules"])

    return True

def register_components(mcp: FastMCP):
    """
    Register codebase exploration components with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    global _codebase_cache
    # Register resources
    @mcp.resource("napistu://codebase/summary")
    async def get_codebase_summary() -> Dict[str, Any]:
        """
        Get a summary of the Napistu codebase structure.
        """
        return {
            "modules": list(_codebase_cache["modules"].keys()),
            "top_level_classes": [
                class_name for class_name, info in _codebase_cache["classes"].items()
                if "." not in class_name  # Only include top-level classes
            ],
            "top_level_functions": [
                func_name for func_name, info in _codebase_cache["functions"].items()
                if "." not in func_name  # Only include top-level functions
            ],
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
            Dictionary with search results organized by code element type
        """
        results = {
            "modules": [],
            "classes": [],
            "functions": [],
        }
        
        # Search modules
        for module_name, info in _codebase_cache["modules"].items():
            module_text = json.dumps(info)
            if query.lower() in module_text.lower():
                results["modules"].append({
                    "name": module_name,
                    "description": info.get("description", ""),
                })
        
        # Search classes
        for class_name, info in _codebase_cache["classes"].items():
            class_text = json.dumps(info)
            if query.lower() in class_text.lower():
                results["classes"].append({
                    "name": class_name,
                    "description": info.get("description", ""),
                })
        
        # Search functions
        for func_name, info in _codebase_cache["functions"].items():
            func_text = json.dumps(info)
            if query.lower() in func_text.lower():
                results["functions"].append({
                    "name": func_name,
                    "description": info.get("description", ""),
                    "signature": info.get("signature", ""),
                })
        
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

