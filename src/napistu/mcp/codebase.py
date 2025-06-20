"""
Codebase exploration components for the Napistu MCP server.
"""

from typing import Dict, Any
import json
import logging

from fastmcp import FastMCP

from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp.constants import NAPISTU_PY_READTHEDOCS_API
from napistu.mcp import codebase_utils
from napistu.mcp import utils as mcp_utils

logger = logging.getLogger(__name__)


class CodebaseState(ComponentState):
    """State management for codebase component."""

    def __init__(self):
        super().__init__()
        self.codebase_cache: Dict[str, Dict[str, Any]] = {
            "modules": {},
            "classes": {},
            "functions": {},
        }

    def is_healthy(self) -> bool:
        """Component is healthy if it has loaded any codebase information."""
        return any(bool(section) for section in self.codebase_cache.values())

    def get_health_details(self) -> Dict[str, Any]:
        """Provide codebase-specific health details."""
        return {
            "modules_count": len(self.codebase_cache["modules"]),
            "classes_count": len(self.codebase_cache["classes"]),
            "functions_count": len(self.codebase_cache["functions"]),
            "total_items": sum(
                len(section) for section in self.codebase_cache.values()
            ),
        }


class CodebaseComponent(MCPComponent):
    """MCP component for codebase exploration and search."""

    def _create_state(self) -> CodebaseState:
        """Create codebase-specific state."""
        return CodebaseState()

    async def initialize(self) -> bool:
        """
        Initialize codebase component by loading documentation from ReadTheDocs.

        Returns
        -------
        bool
            True if codebase information was loaded successfully
        """
        try:
            logger.info("Loading codebase documentation from ReadTheDocs...")

            # Load documentation from the ReadTheDocs API
            modules = await codebase_utils.read_read_the_docs(
                NAPISTU_PY_READTHEDOCS_API
            )
            self.state.codebase_cache["modules"] = modules

            # Extract functions and classes from the modules
            functions, classes = (
                codebase_utils.extract_functions_and_classes_from_modules(modules)
            )
            self.state.codebase_cache["functions"] = functions
            self.state.codebase_cache["classes"] = classes

            logger.info(
                f"Codebase loading complete: "
                f"{len(modules)} modules, "
                f"{len(classes)} classes, "
                f"{len(functions)} functions"
            )

            # Consider successful if we loaded any modules
            return len(modules) > 0

        except Exception as e:
            logger.error(f"Failed to load codebase documentation: {e}")
            return False

    def register(self, mcp: FastMCP) -> None:
        """
        Register codebase resources and tools with the MCP server.

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance
        """

        # Register resources
        @mcp.resource("napistu://codebase/summary")
        async def get_codebase_summary():
            """Get a summary of all available codebase information."""
            return {
                "modules": list(self.state.codebase_cache["modules"].keys()),
                "classes": list(self.state.codebase_cache["classes"].keys()),
                "functions": list(self.state.codebase_cache["functions"].keys()),
            }

        @mcp.resource("napistu://codebase/modules/{module_name}")
        async def get_module_details(module_name: str) -> Dict[str, Any]:
            """Get detailed information about a specific module."""
            if module_name not in self.state.codebase_cache["modules"]:
                return {"error": f"Module {module_name} not found"}

            return self.state.codebase_cache["modules"][module_name]

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
            for module_name, info in self.state.codebase_cache["modules"].items():
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
            for class_name, info in self.state.codebase_cache["classes"].items():
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
            for func_name, info in self.state.codebase_cache["functions"].items():
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
            if function_name not in self.state.codebase_cache["functions"]:
                return {"error": f"Function {function_name} not found"}

            return self.state.codebase_cache["functions"][function_name]

        @mcp.tool()
        async def get_class_documentation(class_name: str) -> Dict[str, Any]:
            """
            Get detailed documentation for a specific class.

            Args:
                class_name: Name of the class

            Returns:
                Dictionary with class documentation
            """
            if class_name not in self.state.codebase_cache["classes"]:
                return {"error": f"Class {class_name} not found"}

            return self.state.codebase_cache["classes"][class_name]


# Module-level component instance
_component = CodebaseComponent()


def get_component() -> CodebaseComponent:
    """
    Get the codebase component instance.

    Returns
    -------
    CodebaseComponent
        The codebase component instance
    """
    return _component
