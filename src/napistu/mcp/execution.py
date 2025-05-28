"""
Function execution components for the Napistu MCP server.
"""

from typing import Dict, List, Any, Optional
import inspect

# Global storage for session context and objects
_session_context = {}
_session_objects = {}


async def initialize_components() -> bool:
    """
    Initialize execution components.

    Returns
    -------
    bool
        True if initialization is successful.
    """
    global _session_context, _session_objects
    import napistu

    _session_context["napistu"] = napistu
    return True


def register_object(name: str, obj: Any) -> None:
    """
    Register an object with the execution component.

    Args:
        name: Name to reference the object by
        obj: The object to register
    """
    global _session_objects
    _session_objects[name] = obj
    print(f"Registered object '{name}' with MCP server")


def register_components(mcp, session_context=None, object_registry=None):
    """
    Register function execution components with the MCP server.

    Args:
        mcp: FastMCP server instance
        session_context: Dictionary of the user's current session (e.g., globals())
        object_registry: Dictionary of named objects to make available
    """
    global _session_context, _session_objects

    # Initialize context
    if session_context:
        _session_context = session_context

    if object_registry:
        _session_objects = object_registry

    import napistu

    _session_context["napistu"] = napistu

    # Register resources
    @mcp.resource("napistu-local://registry")
    async def get_registry_summary() -> Dict[str, Any]:
        """
        Get a summary of all objects registered with the server.
        """
        return {
            "object_count": len(_session_objects),
            "object_names": list(_session_objects.keys()),
            "object_types": {
                name: type(obj).__name__ for name, obj in _session_objects.items()
            },
        }

    @mcp.resource("napistu-local://environment")
    async def get_environment_info() -> Dict[str, Any]:
        """
        Get information about the local Python environment.
        """
        try:
            import napistu

            napistu_version = getattr(napistu, "__version__", "unknown")
        except ImportError:
            napistu_version = "not installed"

        import sys

        return {
            "python_version": sys.version,
            "napistu_version": napistu_version,
            "platform": sys.platform,
            "registered_objects": list(_session_objects.keys()),
        }

    # Register tools
    @mcp.tool()
    async def list_registry() -> Dict[str, Any]:
        """
        List all objects registered with the server.

        Returns:
            Dictionary with information about registered objects
        """
        result = {}

        for name, obj in _session_objects.items():
            obj_type = type(obj).__name__

            # Get additional info based on object type
            if hasattr(obj, "shape"):  # For pandas DataFrame or numpy array
                obj_info = {
                    "type": obj_type,
                    "shape": str(obj.shape),
                }
            elif hasattr(obj, "__len__"):  # For lists, dicts, etc.
                obj_info = {
                    "type": obj_type,
                    "length": len(obj),
                }
            else:
                obj_info = {
                    "type": obj_type,
                }

            result[name] = obj_info

        return result

    @mcp.tool()
    async def describe_object(object_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a registered object.

        Args:
            object_name: Name of the registered object

        Returns:
            Dictionary with object information
        """
        if object_name not in _session_objects:
            return {"error": f"Object '{object_name}' not found in registry"}

        obj = _session_objects[object_name]
        obj_type = type(obj).__name__

        # Basic info for all objects
        result = {
            "name": object_name,
            "type": obj_type,
            "methods": [],
            "attributes": [],
        }

        # Add methods and attributes
        for name in dir(obj):
            if name.startswith("_"):
                continue

            try:
                attr = getattr(obj, name)

                if callable(attr):
                    # Method
                    sig = str(inspect.signature(attr))
                    doc = inspect.getdoc(attr) or ""
                    result["methods"].append(
                        {
                            "name": name,
                            "signature": sig,
                            "docstring": doc,
                        }
                    )
                else:
                    # Attribute
                    attr_type = type(attr).__name__
                    result["attributes"].append(
                        {
                            "name": name,
                            "type": attr_type,
                        }
                    )
            except Exception:
                # Skip attributes that can't be accessed
                pass

        return result

    @mcp.tool()
    async def execute_function(
        function_name: str,
        object_name: Optional[str] = None,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Napistu function on a registered object.

        Args:
            function_name: Name of the function to execute
            object_name: Name of the registered object to operate on (if method call)
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary with execution results
        """
        args = args or []
        kwargs = kwargs or {}

        try:
            if object_name:
                # Method call on an object
                if object_name not in _session_objects:
                    return {"error": f"Object '{object_name}' not found in registry"}

                obj = _session_objects[object_name]

                if not hasattr(obj, function_name):
                    return {
                        "error": f"Method '{function_name}' not found on object '{object_name}'"
                    }

                func = getattr(obj, function_name)
                result = func(*args, **kwargs)
            else:
                # Global function call
                if function_name in _session_context:
                    # Function from session context
                    func = _session_context[function_name]
                    result = func(*args, **kwargs)
                else:
                    # Try to find the function in Napistu
                    try:
                        import napistu

                        # Split function name by dots for nested modules
                        parts = function_name.split(".")
                        current = napistu

                        for part in parts[:-1]:
                            current = getattr(current, part)

                        func = getattr(current, parts[-1])
                        result = func(*args, **kwargs)
                    except (ImportError, AttributeError):
                        return {"error": f"Function '{function_name}' not found"}

            # Register result if it's a return value
            if result is not None:
                result_name = f"result_{len(_session_objects) + 1}"
                _session_objects[result_name] = result

                # Basic type conversion for JSON serialization
                if hasattr(result, "to_dict"):
                    # For pandas DataFrame or similar
                    return {
                        "success": True,
                        "result_name": result_name,
                        "result_type": type(result).__name__,
                        "result_preview": (
                            result.to_dict()
                            if hasattr(result, "__len__") and len(result) < 10
                            else "Result too large to preview"
                        ),
                    }
                elif hasattr(result, "to_json"):
                    # For objects with JSON serialization
                    return {
                        "success": True,
                        "result_name": result_name,
                        "result_type": type(result).__name__,
                        "result_preview": result.to_json(),
                    }
                elif hasattr(result, "__dict__"):
                    # For custom objects
                    return {
                        "success": True,
                        "result_name": result_name,
                        "result_type": type(result).__name__,
                        "result_preview": str(result),
                    }
                else:
                    # For simple types
                    return {
                        "success": True,
                        "result_name": result_name,
                        "result_type": type(result).__name__,
                        "result_preview": str(result),
                    }
            else:
                return {
                    "success": True,
                    "result": None,
                }
        except Exception as e:
            import traceback

            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    @mcp.tool()
    async def search_paths(
        source_node: str,
        target_node: str,
        network_object: str,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Find paths between two nodes in a network.

        Args:
            source_node: Source node identifier
            target_node: Target node identifier
            network_object: Name of the registered network object
            max_depth: Maximum path length

        Returns:
            Dictionary with paths found
        """
        if network_object not in _session_objects:
            return {"error": f"Network object '{network_object}' not found in registry"}

        network = _session_objects[network_object]

        try:
            # Import necessary modules
            import napistu

            # Check if the object is a valid network type
            if hasattr(network, "find_paths"):
                # Direct method call
                paths = network.find_paths(
                    source_node, target_node, max_depth=max_depth
                )
            elif hasattr(napistu.graph, "find_paths"):
                # Function call
                paths = napistu.graph.find_paths(
                    network, source_node, target_node, max_depth=max_depth
                )
            else:
                return {"error": "Could not find appropriate path-finding function"}

            # Register result
            result_name = f"paths_{len(_session_objects) + 1}"
            _session_objects[result_name] = paths

            # Return results
            if hasattr(paths, "to_dict"):
                return {
                    "success": True,
                    "result_name": result_name,
                    "paths_found": (
                        len(paths) if hasattr(paths, "__len__") else "unknown"
                    ),
                    "result_preview": (
                        paths.to_dict()
                        if hasattr(paths, "__len__") and len(paths) < 10
                        else "Result too large to preview"
                    ),
                }
            else:
                return {
                    "success": True,
                    "result_name": result_name,
                    "paths_found": (
                        len(paths) if hasattr(paths, "__len__") else "unknown"
                    ),
                    "result_preview": str(paths),
                }
        except Exception as e:
            import traceback

            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
