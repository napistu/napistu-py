"""
Function execution components for the Napistu MCP server.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp.constants import HEALTH_SUMMARIES
from napistu.mcp.semantic_search import SemanticSearch

logger = logging.getLogger(__name__)


class ExecutionState(ComponentState):
    """State management for execution component."""

    def __init__(
        self,
        session_context: Optional[Dict] = None,
        object_registry: Optional[Dict] = None,
    ):
        super().__init__()
        # Session context contains global functions and modules
        self.session_context = session_context or {}
        # Object registry contains user-registered objects
        self.session_objects = object_registry or {}

    def is_healthy(self) -> bool:
        """Component is healthy if it has a session context."""
        return bool(self.session_context)

    def get_health_details(self) -> Dict[str, Any]:
        """Provide execution-specific health details."""

        return {
            HEALTH_SUMMARIES.SESSION_CONTEXT_ITEMS: len(self.session_context),
            HEALTH_SUMMARIES.REGISTERED_OBJECTS: len(self.session_objects),
            HEALTH_SUMMARIES.CONTEXT_KEYS: list(self.session_context.keys()),
            HEALTH_SUMMARIES.OBJECT_NAMES: list(self.session_objects.keys()),
        }

    def register_object(self, name: str, obj: Any) -> None:
        """Register an object with the execution component."""
        self.session_objects[name] = obj
        logger.info(f"Registered object '{name}' with MCP server")


class ExecutionComponent(MCPComponent):
    """MCP component for function execution and object management."""

    def __init__(
        self,
        session_context: Optional[Dict] = None,
        object_registry: Optional[Dict] = None,
    ):
        # Override parent constructor to pass context to state
        self.state = ExecutionState(session_context, object_registry)

    def _create_state(self) -> ExecutionState:
        """This won't be called due to overridden constructor."""
        pass

    async def initialize(self, semantic_search: SemanticSearch = None) -> bool:
        """
        Initialize execution component by setting up the session context.

        Parameters
        ----------
        semantic_search : SemanticSearch, optional
            Shared semantic search instance (unused by execution component).
            Parameter included for consistency with other components.

        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            # Import and add napistu to session context
            import napistu

            self.state.session_context["napistu"] = napistu

            logger.info("Execution component initialized with napistu module")
            return True

        except ImportError as e:
            logger.error(f"Failed to import napistu module: {e}")
            return False

    def register_object(self, name: str, obj: Any) -> None:
        """
        Register an object with the execution component.

        Parameters
        ----------
        name : str
            Name to reference the object by
        obj : Any
            The object to register
        """
        self.state.register_object(name, obj)

    def register(self, mcp: FastMCP) -> None:
        """
        Register execution resources and tools with the MCP server.

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance
        """

        # Register resources
        @mcp.resource("napistu://execution/registry")
        async def get_registry():
            """
            Get a summary of all objects registered with the execution server.

            **USE THIS WHEN:**
            - Checking what Napistu objects are currently available for execution
            - Understanding what data structures, networks, or results are loaded
            - Getting an overview of the current execution session state

            **DO NOT USE FOR:**
            - General information about Napistu classes or functions (use codebase component)
            - Creating new objects (use execute_function tool)
            - Detailed object inspection (use describe_object tool)

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - object_count : int
                    Number of objects currently registered
                - object_names : List[str]
                    Names of all registered objects
                - object_types : Dict[str, str]
                    Mapping of object names to their types

            Examples
            --------
            Use this to see what Napistu objects are available before attempting
            to execute methods on them or analyze their contents.
            """
            return {
                "object_count": len(self.state.session_objects),
                "object_names": list(self.state.session_objects.keys()),
                "object_types": {
                    name: type(obj).__name__
                    for name, obj in self.state.session_objects.items()
                },
            }

        @mcp.resource("napistu://execution/environment")
        async def get_environment_info() -> Dict[str, Any]:
            """
            Get information about the local Python environment and Napistu installation.

            **USE THIS WHEN:**
            - Checking if Napistu is properly installed and available
            - Debugging environment issues with Napistu execution
            - Understanding the Python version and platform context

            **DO NOT USE FOR:**
            - General Napistu documentation (use documentation component)
            - Napistu version comparisons or release notes (use documentation component)
            - Installation instructions (use documentation component)

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - python_version : str
                    Current Python version
                - napistu_version : str
                    Installed Napistu version
                - platform : str
                    Operating system platform
                - registered_objects : List[str]
                    Currently registered object names
                - session_context : List[str]
                    Available session context keys

            Examples
            --------
            Use this to verify Napistu is available before attempting function execution
            or to troubleshoot environment-related execution issues.
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
                "registered_objects": list(self.state.session_objects.keys()),
                "session_context": list(self.state.session_context.keys()),
            }

        # Register tools
        @mcp.tool()
        async def list_registry() -> Dict[str, Any]:
            """
            List all objects registered with the execution server with detailed type information.

            **USE THIS WHEN:**
            - Getting detailed information about registered Napistu objects
            - Understanding object shapes, lengths, or structure before operating on them
            - Deciding which objects to use for analysis or computation

            **DO NOT USE FOR:**
            - Creating new objects (use execute_function tool)
            - Getting method/attribute details (use describe_object tool)
            - General Napistu API documentation (use codebase component)

            Returns
            -------
            Dict[str, Any]
                Dictionary mapping object names to their detailed information including:
                - type : str
                    Object type name
                - shape : str (for arrays/DataFrames)
                    Dimensions of the data structure
                - length : int (for collections)
                    Number of elements in the object

            Examples
            --------
            Use this to understand the structure of loaded datasets, networks, or
            analysis results before performing operations on them.
            """
            result = {}

            for name, obj in self.state.session_objects.items():
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
            Get detailed information about a specific registered Napistu object.

            **USE THIS WHEN:**
            - Exploring methods and attributes available on a Napistu object
            - Understanding what operations can be performed on a loaded dataset or network
            - Getting function signatures and documentation for object methods
            - Planning how to interact with a specific Napistu data structure

            **DO NOT USE FOR:**
            - General Napistu API documentation (use codebase component for class definitions)
            - Creating new objects (use execute_function tool)
            - Objects that aren't registered (check list_registry first)
            - Non-Napistu objects or general Python concepts

            Parameters
            ----------
            object_name : str
                Name of the registered object to describe (from list_registry)

            Returns
            -------
            Dict[str, Any]
                Detailed object information including:
                - name : str
                    Object name in registry
                - type : str
                    Object type name
                - methods : List[Dict]
                    Available methods with signatures and documentation
                - attributes : List[Dict]
                    Available attributes with types

            Examples
            --------
            After loading a consensus network or SBML model, use this to understand
            what analysis methods are available and how to call them.

            >>> describe_object("my_network")
            # Returns methods like find_paths, get_nodes, analyze_topology, etc.
            """
            if object_name not in self.state.session_objects:
                return {"error": f"Object '{object_name}' not found in registry"}

            obj = self.state.session_objects[object_name]
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
            Execute a Napistu function or method on registered objects.

            **USE THIS WHEN:**
            - Calling Napistu functions to create networks, load data, or perform analysis
            - Executing methods on registered Napistu objects (networks, datasets, models)
            - Creating new Napistu objects from loaded data
            - Performing computational analysis using Napistu functionality

            **DO NOT USE FOR:**
            - General Python functions not related to Napistu
            - Functions from other libraries (pandas, numpy, etc.) unless they're part of Napistu workflows
            - Operations that don't involve Napistu objects or functionality
            - File I/O operations outside of Napistu's data loading functions

            **EXAMPLE APPROPRIATE USES:**
            - Creating consensus networks: `execute_function("create_consensus_network", args=[data])`
            - Loading SBML models: `execute_function("sbml.load_model", args=["path/to/model.xml"])`
            - Analyzing networks: `execute_function("analyze_topology", object_name="my_network")`
            - Pathway analysis: `execute_function("find_pathways", object_name="network", args=[source, target])`

            **EXAMPLE INAPPROPRIATE USES:**
            - `execute_function("pandas.read_csv")` (use Napistu's data loading instead)
            - `execute_function("print", args=["hello"])` (not Napistu-related)
            - `execute_function("os.listdir")` (file system operations)

            Parameters
            ----------
            function_name : str
                Name of the Napistu function to execute. Can be:
                - Method name (if object_name provided): "find_paths", "get_nodes"
                - Module function: "sbml.load_model", "consensus.create_network"
                - Top-level function: "load_data", "create_graph"
            object_name : str, optional
                Name of registered object to call method on. If None, treats as global function.
            args : List, optional
                Positional arguments to pass to the function
            kwargs : Dict, optional
                Keyword arguments to pass to the function

            Returns
            -------
            Dict[str, Any]
                Execution result containing:
                - success : bool
                    Whether execution succeeded
                - result_name : str (if successful)
                    Name assigned to result object in registry
                - result_type : str (if successful)
                    Type of the result object
                - result_preview : Any (if successful)
                    Preview of the result data
                - error : str (if failed)
                    Error message describing the failure
                - traceback : str (if failed)
                    Full Python traceback for debugging

            Examples
            --------
            Execute Napistu functions for network analysis workflows:

            >>> # Create a consensus network from data
            >>> execute_function("consensus.create_network", args=[data_object])

            >>> # Analyze network topology
            >>> execute_function("analyze_centrality", object_name="network_1")

            >>> # Load and process SBML model
            >>> execute_function("sbml.load_and_process", args=["model.xml"])

            Notes
            -----
            **RESULT HANDLING:**
            - Successful executions automatically register results with generated names
            - Use list_registry() to see newly created objects
            - Use describe_object() to explore result object capabilities
            - Results persist in the session for further analysis

            **ERROR HANDLING:**
            - Function errors return detailed traceback for debugging
            - Check codebase component for correct function signatures
            - Verify object names exist using list_registry() before method calls
            """
            args = args or []
            kwargs = kwargs or {}

            try:
                if object_name:
                    # Method call on an object
                    if object_name not in self.state.session_objects:
                        return {
                            "error": f"Object '{object_name}' not found in registry"
                        }

                    obj = self.state.session_objects[object_name]

                    if not hasattr(obj, function_name):
                        return {
                            "error": f"Method '{function_name}' not found on object '{object_name}'"
                        }

                    func = getattr(obj, function_name)
                    result = func(*args, **kwargs)
                else:
                    # Global function call
                    if function_name in self.state.session_context:
                        # Function from session context
                        func = self.state.session_context[function_name]
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
                    result_name = f"result_{len(self.state.session_objects) + 1}"
                    self.state.session_objects[result_name] = result

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
            Find paths between two nodes in a registered Napistu network object.

            **USE THIS WHEN:**
            - Analyzing connectivity between specific nodes in a Napistu network
            - Finding pathways in biological networks (metabolic, signaling, etc.)
            - Exploring network structure and shortest paths
            - Conducting pathway analysis in consensus networks

            **DO NOT USE FOR:**
            - General graph algorithms not related to Napistu networks
            - Networks from other libraries (NetworkX, igraph, etc.)
            - Objects that aren't network-like or don't have path-finding capabilities
            - Non-Napistu data structures

            **PREREQUISITES:**
            - Must have a registered network object (check with list_registry)
            - Network object should support path finding (check with describe_object)
            - Node names must exist in the network

            Parameters
            ----------
            source_node : str
                Name/ID of the starting node in the network
            target_node : str
                Name/ID of the destination node in the network
            network_object : str
                Name of the registered network object (from list_registry)
            max_depth : int, optional
                Maximum path length to search (default 3)

            Returns
            -------
            Dict[str, Any]
                Path search results containing:
                - success : bool
                    Whether path search succeeded
                - result_name : str (if successful)
                    Name assigned to paths result in registry
                - paths_found : int (if successful)
                    Number of paths discovered
                - result_preview : Any (if successful)
                    Preview of the path data
                - error : str (if failed)
                    Error message describing the failure

            Examples
            --------
            Find pathways in biological networks:

            >>> # Find metabolic pathways
            >>> search_paths("glucose", "ATP", "metabolic_network", max_depth=5)

            >>> # Explore signaling cascades
            >>> search_paths("receptor", "transcription_factor", "signaling_net")

            >>> # Analyze consensus network connectivity
            >>> search_paths("gene_A", "gene_B", "consensus_net", max_depth=4)

            Notes
            -----
            **NETWORK COMPATIBILITY:**
            - Works with Napistu network objects that have path-finding capabilities
            - Automatically detects whether to use object methods or Napistu graph functions
            - Results are registered for further analysis and visualization

            **PATH INTERPRETATION:**
            - Paths represent functional connections in biological contexts
            - Path length indicates directness of molecular relationships
            - Multiple paths suggest robustness or alternative mechanisms
            """
            if network_object not in self.state.session_objects:
                return {
                    "error": f"Network object '{network_object}' not found in registry"
                }

            network = self.state.session_objects[network_object]

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
                result_name = f"paths_{len(self.state.session_objects) + 1}"
                self.state.session_objects[result_name] = paths

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


# Module-level component instance (will be created by server with proper context)
_component: Optional[ExecutionComponent] = None


def create_component(
    session_context: Optional[Dict] = None, object_registry: Optional[Dict] = None
) -> ExecutionComponent:
    """
    Create and configure the execution component with session context.

    Args:
        session_context: Dictionary of the user's current session (e.g., globals())
        object_registry: Dictionary of named objects to make available

    Returns:
        ExecutionComponent: Configured execution component
    """
    global _component
    _component = ExecutionComponent(session_context, object_registry)
    return _component


def get_component() -> ExecutionComponent:
    """
    Get the execution component instance.

    Returns
    -------
    ExecutionComponent
        The execution component instance

    Raises
    ------
    RuntimeError
        If component hasn't been created yet
    """
    if _component is None:
        raise RuntimeError(
            "Execution component not created. Call create_component() first."
        )
    return _component


def register_object(name: str, obj: Any) -> None:
    """
    Register an object with the execution component (legacy function).

    Args:
        name: Name to reference the object by
        obj: The object to register
    """
    if _component is None:
        raise RuntimeError(
            "Execution component not created. Call create_component() first."
        )
    _component.register_object(name, obj)
