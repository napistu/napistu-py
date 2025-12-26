"""
Codebase exploration components for the Napistu MCP server.
"""

import logging
from typing import Any, Dict

from fastmcp import FastMCP

from napistu.mcp import codebase_utils, inspect_utils
from napistu.mcp.component_base import ComponentState, MCPComponent
from napistu.mcp.constants import (
    CODEBASE_DEFS,
    HEALTH_SUMMARIES,
    MCP_COMPONENTS,
    NAPISTU_PY_READTHEDOCS_API,
    NAPISTU_TORCH_READTHEDOCS_API,
    SEARCH_RESULT_DEFS,
    SEARCH_TYPES,
)
from napistu.mcp.semantic_search import SemanticSearch

logger = logging.getLogger(__name__)


class CodebaseState(ComponentState):
    """
    State management for codebase component with semantic search capabilities.

    Manages cached codebase information and tracks semantic search availability.
    Extends ComponentState to provide standardized health monitoring and status reporting.

    Attributes
    ----------
    codebase_cache : Dict[str, Dict[str, Any]]
        Dictionary containing cached codebase information organized by type:
        - modules: Module documentation and metadata
        - classes: Class documentation and metadata
        - functions: Function documentation, signatures, and metadata
    semantic_search : SemanticSearch or None
        Reference to shared semantic search instance for AI-powered codebase search,
        None if not initialized

    Examples
    --------
    >>> state = CodebaseState()
    >>> state.codebase_cache["functions"]["create_network"] = {...}
    >>> print(state.is_healthy())  # True if any codebase info loaded
    >>> health = state.get_health_details()
    >>> print(health["total_items"])
    """

    def __init__(self):
        super().__init__()
        self.codebase_cache: Dict[str, Dict[str, Any]] = {
            CODEBASE_DEFS.MODULES: {},
            CODEBASE_DEFS.CLASSES: {},
            CODEBASE_DEFS.FUNCTIONS: {},
        }
        self.semantic_search = None

    def is_healthy(self) -> bool:
        """
        Check if component has successfully loaded codebase information.

        Returns
        -------
        bool
            True if any codebase information is loaded, False otherwise

        Notes
        -----
        This method checks for the presence of any codebase content.
        Semantic search availability is not required for health.
        """
        return any(bool(section) for section in self.codebase_cache.values())

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information including codebase element counts.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - modules_count : int
                Number of modules loaded
            - classes_count : int
                Number of classes loaded
            - functions_count : int
                Number of functions loaded
            - total_items : int
                Total number of codebase elements loaded

        Examples
        --------
        >>> state = CodebaseState()
        >>> # ... load content ...
        >>> details = state.get_health_details()
        >>> print(f"Total codebase items: {details['total_items']}")
        """
        return {
            HEALTH_SUMMARIES.MODULES_COUNT: len(
                self.codebase_cache[CODEBASE_DEFS.MODULES]
            ),
            HEALTH_SUMMARIES.CLASSES_COUNT: len(
                self.codebase_cache[CODEBASE_DEFS.CLASSES]
            ),
            HEALTH_SUMMARIES.FUNCTIONS_COUNT: len(
                self.codebase_cache[CODEBASE_DEFS.FUNCTIONS]
            ),
            HEALTH_SUMMARIES.TOTAL_ITEMS: sum(
                len(section) for section in self.codebase_cache.values()
            ),
        }


class CodebaseComponent(MCPComponent):
    """
    MCP component for codebase exploration and search with semantic capabilities.

    Provides access to Napistu codebase documentation including modules, classes, and
    functions with both exact text matching and AI-powered semantic search for natural
    language queries. Loads comprehensive API documentation from ReadTheDocs.

    The component fetches codebase information from the Napistu ReadTheDocs API and
    uses a shared semantic search instance for intelligent code discovery and exploration.

    Public Methods
    --------------
    initialize(semantic_search)
        Initialize codebase component with content loading and semantic indexing.
    register(mcp)
        Register codebase resources and tools with the MCP server.

    Private Methods
    ---------------
    _check_initialized()
        Check if component is initialized, raise clear error if not.
    _create_state()
        Create codebase-specific state instance.
    _initialize_semantic_search()
        Index codebase content into the shared semantic search instance.

    Examples
    --------
    Basic component usage:

    >>> component = CodebaseComponent()
    >>> semantic_search = SemanticSearch()  # Shared instance
    >>> success = await component.safe_initialize(semantic_search)
    >>> if success:
    ...     state = component.get_state()
    ...     print(f"Loaded {state.get_health_details()['total_items']} codebase items")

    Notes
    -----
    The component gracefully handles failures in codebase loading and semantic search
    initialization. If semantic search is not provided, the component continues to
    function with exact text search only.

    **CONTENT SCOPE:**
    Codebase documentation covers Napistu API reference, function signatures, and
    implementation details. Use this component for technical API guidance, not conceptual
    tutorials or general usage patterns.
    """

    async def initialize(self, semantic_search: SemanticSearch = None) -> bool:
        """
        Initialize codebase component with content loading and semantic indexing.

        Performs the following operations:
        1. Loads codebase documentation from ReadTheDocs API
        2. Extracts and organizes modules, classes, and functions
        3. Stores reference to shared semantic search instance
        4. Indexes loaded codebase content if semantic search is available

        Parameters
        ----------
        semantic_search : SemanticSearch, optional
            Shared semantic search instance for AI-powered search capabilities.
            If None, component will operate with exact text search only.

        Returns
        -------
        bool
            True if codebase information was loaded successfully, False if
            loading failed

        Notes
        -----
        ReadTheDocs API failures are logged as errors and cause initialization failure.
        Semantic search indexing failure is logged but doesn't affect the return value -
        the component can function without semantic search.
        """
        try:
            logger.info("Loading codebase documentation from ReadTheDocs...")

            # Load documentation from both ReadTheDocs APIs
            modules_py = await codebase_utils.read_read_the_docs(
                NAPISTU_PY_READTHEDOCS_API
            )
            modules_torch = await codebase_utils.read_read_the_docs(
                NAPISTU_TORCH_READTHEDOCS_API
            )

            # Merge modules from both packages
            modules = {**modules_py, **modules_torch}
            self.state.codebase_cache[CODEBASE_DEFS.MODULES] = modules

            # Extract functions and classes from the merged modules
            functions, classes = (
                codebase_utils.extract_functions_and_classes_from_modules(modules)
            )
            self.state.codebase_cache[CODEBASE_DEFS.FUNCTIONS] = functions
            self.state.codebase_cache[CODEBASE_DEFS.CLASSES] = classes

            # Add stripped names for easier lookup
            codebase_utils.add_stripped_names(functions, classes)

            logger.info(
                f"Codebase loading complete: "
                f"{len(modules)} modules, "
                f"{len(classes)} classes, "
                f"{len(functions)} functions"
            )

            # Store reference to shared semantic search instance
            content_loaded = len(modules) > 0
            if semantic_search and content_loaded:
                self.state.semantic_search = semantic_search
                semantic_success = await self._initialize_semantic_search()
                logger.info(
                    f"Semantic search initialization: {'✅ Success' if semantic_success else '⚠️ Failed'}"
                )

            return content_loaded

        except Exception as e:
            logger.error(f"Failed to load codebase documentation: {e}")
            return False

    def register(self, mcp: FastMCP) -> None:
        """
        Register codebase resources and tools with the MCP server.

        Registers the following MCP endpoints:
        - Resources for accessing codebase summaries and specific API documentation
        - Tools for searching codebase with semantic and exact modes

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance to register endpoints with

        Notes
        -----
        The search tool automatically selects semantic search when available,
        falling back to exact search if semantic search is not initialized.
        """

        # resources
        @mcp.resource("napistu://codebase/summary")
        async def get_codebase_summary():
            self._check_initialized()
            """
            Get a summary of all available Napistu codebase information.

            **USE THIS WHEN:**
            - Getting an overview of available Napistu API documentation
            - Understanding what modules, classes, and functions are documented
            - Checking codebase documentation availability and counts

            **DO NOT USE FOR:**
            - General programming concepts not specific to Napistu
            - Documentation for other libraries or frameworks
            - Conceptual tutorials (use tutorials component instead)
            - Implementation examples (use documentation component for wikis/READMEs)

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - modules : List[str]
                    Names of documented Napistu modules
                - classes : List[str]
                    Names of documented Napistu classes
                - functions : List[str]
                    Names of documented Napistu functions

            Examples
            --------
            Use this to understand what Napistu API documentation is available before
            searching for specific function signatures or class definitions.
            """
            return {
                CODEBASE_DEFS.MODULES: list(
                    self.state.codebase_cache[CODEBASE_DEFS.MODULES].keys()
                ),
                CODEBASE_DEFS.CLASSES: list(
                    self.state.codebase_cache[CODEBASE_DEFS.CLASSES].keys()
                ),
                CODEBASE_DEFS.FUNCTIONS: list(
                    self.state.codebase_cache[CODEBASE_DEFS.FUNCTIONS].keys()
                ),
            }

        @mcp.resource("napistu://codebase/modules/{module_name}")
        async def get_module_details(module_name: str) -> Dict[str, Any]:
            self._check_initialized()
            """
            Get detailed API documentation for a specific Napistu module.

            **USE THIS WHEN:**
            - Reading complete documentation for a specific Napistu module
            - Understanding module structure, classes, and functions
            - Getting detailed API reference information

            **DO NOT USE FOR:**
            - Modules from other libraries (only covers Napistu modules)
            - General programming concepts or tutorials
            - Implementation examples (use tutorials/documentation components)

            Parameters
            ----------
            module_name : str
                Name of the Napistu module (from codebase summary)

            Returns
            -------
            Dict[str, Any]
                Complete module documentation including classes, functions,
                and detailed API information

            Raises
            ------
            Exception
                If the module is not found in the codebase documentation
            """
            if module_name not in self.state.codebase_cache[CODEBASE_DEFS.MODULES]:
                return {"error": f"Module {module_name} not found"}

            return self.state.codebase_cache[CODEBASE_DEFS.MODULES][module_name]

        # tools

        @mcp.tool()
        async def get_class_documentation(class_name: str) -> Dict[str, Any]:
            self._check_initialized()
            """
            Get detailed API documentation for a specific Napistu class.

            **USE THIS WHEN:**
            - Reading complete documentation for a specific Napistu class
            - Understanding class methods, attributes, and inheritance
            - Getting detailed API reference for class usage

            **DO NOT USE FOR:**
            - Classes from other libraries (only covers Napistu classes)
            - General object-oriented programming concepts
            - Usage examples (use tutorials component for implementation guidance)

            Parameters
            ----------
            class_name : str
                Name of the Napistu class (can be short name like "NapistuGraph"
                or full path like "napistu.network.ng_core.NapistuGraph")

            Returns
            -------
            Dict[str, Any]
                Complete class documentation including methods, attributes,
                inheritance, and detailed description, or error message if not found

            Examples
            --------
            >>> # These all work:
            >>> get_class_documentation("NapistuGraph")
            >>> get_class_documentation("napistu.network.ng_core.NapistuGraph")
            >>> get_class_documentation("SBML_dfs")
            """
            result = codebase_utils.find_item_by_name(
                class_name, self.state.codebase_cache[CODEBASE_DEFS.CLASSES]
            )
            if result is None:
                return {
                    "error": f"Class '{class_name}' not found. Try searching for similar names."
                }

            full_name, class_info = result
            # Add the full name to the response for clarity
            class_info["full_name"] = full_name
            return class_info

        @mcp.tool()
        async def get_function_documentation(function_name: str) -> Dict[str, Any]:
            self._check_initialized()
            """
            Get detailed API documentation for a specific Napistu function.

            **USE THIS WHEN:**
            - Reading complete documentation for a specific Napistu function
            - Understanding function signatures, parameters, and return types
            - Getting detailed API reference for function implementation

            **DO NOT USE FOR:**
            - Functions from other libraries (only covers Napistu functions)
            - General programming concepts or tutorials
            - Usage examples (use tutorials component for implementation guidance)

            Parameters
            ----------
            function_name : str
                Name of the Napistu function (can be short name like "create_network"
                or full path like "napistu.network.create_network")

            Returns
            -------
            Dict[str, Any]
                Complete function documentation including signature, parameters,
                return type, and detailed description, or error message if not found

            Examples
            --------
            >>> # These all work:
            >>> get_function_documentation("create_network")
            >>> get_function_documentation("napistu.network.create_network")
            >>> get_function_documentation("create_consensus")
            """
            result = codebase_utils.find_item_by_name(
                function_name, self.state.codebase_cache[CODEBASE_DEFS.FUNCTIONS]
            )
            if result is None:
                return {
                    "error": f"Function '{function_name}' not found. Try searching for similar names."
                }

            full_name, func_info = result
            # Add the full name to the response for clarity
            func_info["full_name"] = full_name
            return func_info

        @mcp.tool()
        async def inspect_class(
            class_name: str, package_name: str = "napistu", include_init: bool = True
        ) -> Dict[str, Any]:
            """
            Get runtime inspection of an installed class with actual source code.

            This complements get_class_documentation by providing:
            - Actual __init__ source code (not available in ReadTheDocs)
            - Runtime type hints and defaults
            - Works for any installed package (napistu, napistu_torch, etc.)

            The key value is seeing the __init__ implementation which shows:
            - Initialization logic and validation
            - Default parameter values
            - Required vs optional parameters
            - Setup and configuration code

            **USE THIS WHEN:**
            - You need to see how to properly instantiate a class
            - Understanding __init__ parameters and their defaults
            - Seeing what happens during class initialization
            - ReadTheDocs documentation is insufficient

            **DO NOT USE FOR:**
            - Just getting method signatures (use get_class_documentation)
            - Classes from other libraries
            - Understanding class methods (use get_method_source for that)

            Parameters
            ----------
            class_name : str
                Class name (e.g., "NapistuGraph" or "network.NapistuGraph")
            package_name : str, optional
                Package to inspect (default: "napistu")
            include_init : bool, optional
                Include __init__ source code (default: True)

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - success : bool
                - name : str
                - module : str
                - docstring : str
                - file_path : str
                - line_number : int
                - init_signature : str
                - init_source : str (actual __init__ code if include_init=True)
                - methods : Dict[str, MethodInfo] (signatures only, no source)
                - error : str (if failed)

            Examples
            --------
            >>> inspect_class("SBML_dfs")
            >>> inspect_class("network.NapistuGraph", "napistu")
            >>> inspect_class("napistu_data.NapistuData", "napistu_torch")
            >>> inspect_class("torch.Tensor", "torch")
            """
            try:
                # Resolve class name from cache if needed
                resolved_class_name = codebase_utils._resolve_name_from_cache(
                    class_name,
                    self.state.codebase_cache[CODEBASE_DEFS.CLASSES],
                    package_name,
                )

                # Import the class
                cls, error = inspect_utils.import_object(
                    resolved_class_name, package_name
                )
                if error:
                    return {
                        "success": False,
                        "error": error,
                        "suggestion": "Try using search_codebase first to find the correct class path",
                    }

                # Get class info using Pydantic model
                info = inspect_utils.ClassInfo.from_class(
                    cls, include_init=include_init
                )

                return {
                    "success": True,
                    "class_name": class_name,
                    "package": package_name,
                    **info.model_dump(),
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "suggestion": "Verify the class exists and is installed",
                }

        @mcp.tool()
        async def inspect_cli_command(
            command_path: str, cli_name: str = "napistu"
        ) -> Dict[str, Any]:
            """
            Get detailed information about a specific CLI command.

            **USE THIS WHEN:**
            - Understanding how to use a specific CLI command
            - Seeing all arguments and options for a command
            - Getting help text and usage examples

            Parameters
            ----------
            command_path : str
                Full command path (e.g., "napistu ingestion reactome")
            cli_name : str, optional
                Which CLI to inspect ("napistu" or "napistu.mcp")

            Returns
            -------
            Dict[str, Any]
                Detailed command information with arguments, options, help text

            Examples
            --------
            >>> inspect_cli_command("napistu ingestion reactome")
            >>> inspect_cli_command("napistu consensus create")
            >>> inspect_cli_command("napistu-torch train")
            """
            try:
                if cli_name == "napistu":
                    from napistu.__main__ import cli
                elif cli_name == "napistu.mcp":
                    from napistu.mcp.__main__ import cli
                elif cli_name == "napistu-torch":
                    try:
                        from napistu_torch.__main__ import cli
                    except ImportError:
                        return {
                            "success": False,
                            "error": "napistu-torch is not installed",
                            "suggestion": "Install napistu-torch using `pip install napistu-torch`",
                        }
                else:
                    return {"error": f"Unknown CLI: {cli_name}"}

                from napistu.mcp import cli_utils

                structure = cli_utils.CLIStructure.from_cli_group(cli, cli_name)

                # Find the command
                if command_path not in structure.commands:
                    return {
                        "success": False,
                        "error": f"Command '{command_path}' not found",
                        "available_commands": list(structure.commands.keys())[:10],
                    }

                return {
                    "success": True,
                    **structure.commands[command_path].model_dump(),
                }

            except Exception as e:
                return {"success": False, "error": str(e)}

        @mcp.tool()
        async def inspect_function(
            function_name: str, package_name: str = "napistu"
        ) -> Dict[str, Any]:
            """
            Get runtime inspection of an installed function with actual source code.

            This complements get_function_documentation by providing:
            - Actual function implementation (source code)
            - Runtime signature with defaults
            - Works for any installed package (napistu, napistu_torch, etc.)

            **USE THIS WHEN:**
            - You need to see the actual implementation of a function
            - ReadTheDocs documentation is insufficient
            - You want to understand how a function works internally
            - Debugging or understanding implementation details

            **DO NOT USE FOR:**
            - Just getting the function signature (use get_function_documentation)
            - Functions that don't exist in installed packages
            - Built-in Python functions or standard library

            Parameters
            ----------
            function_name : str
                Function name (e.g., "create_network" or "network.create_network")
            package_name : str, optional
                Package to inspect (default: "napistu")

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - success : bool
                - source : str (actual function code)
                - signature : str
                - docstring : str
                - file_path : str
                - line_number : int
                - error : str (if failed)

            Examples
            --------
            >>> inspect_function("consensus.construct_consensus_model")
            >>> inspect_function("SBML_dfs.parse_model", "napistu")
            >>> inspect_function("utils.tensor_utils.compute_cosine_distances_torch", "napistu_torch")
            >>> inspect_function("igraph.Graph", "igraph")
            """
            try:
                # Resolve function name from cache if needed
                resolved_function_name = codebase_utils._resolve_name_from_cache(
                    function_name,
                    self.state.codebase_cache[CODEBASE_DEFS.FUNCTIONS],
                    package_name,
                )

                # Import the function
                func, error = inspect_utils.import_object(
                    resolved_function_name, package_name
                )
                if error:
                    return {
                        "success": False,
                        "error": error,
                        "suggestion": "Try using search_codebase first to find the correct function path",
                    }

                # Get function info using Pydantic model
                info = inspect_utils.FunctionInfo.from_function(func)

                return {
                    "success": True,
                    "function_name": function_name,
                    "package": package_name,
                    **info.model_dump(),
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "suggestion": "Verify the function exists and is installed",
                }

        @mcp.tool()
        async def inspect_method(
            class_name: str, method_name: str, package_name: str = "napistu"
        ) -> Dict[str, Any]:
            """
            Get actual source code for a specific method of a class.

            Use this after inspect_class when you want to see the implementation
            of a specific method. This provides the actual code, not just the signature.

            **USE THIS WHEN:**
            - You need to see how a specific method is implemented
            - Understanding method logic and algorithm
            - Debugging or learning from implementation
            - ReadTheDocs documentation is insufficient

            **DO NOT USE FOR:**
            - Just getting method signatures (use get_class_documentation or inspect_class)
            - Methods from other libraries
            - Understanding what methods exist (use inspect_class for that)

            Parameters
            ----------
            class_name : str
                Class name (e.g., "SBML_dfs" or "sbml_dfs_core.SBML_dfs")
            method_name : str
                Name of the method (e.g., "get_reactions", "parse_model")
            package_name : str, optional
                Package to inspect (default: "napistu")

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - success : bool
                - method_name : str
                - class_name : str
                - source : str (actual method code)
                - signature : str
                - docstring : str
                - line_number : int
                - error : str (if failed)

            Examples
            --------
            >>> inspect_method("SBML_dfs", "get_identifiers")
            >>> inspect_method("NapistuGraph", "reverse_edges", "napistu")
            >>> inspect_method("network.NapistuGraph", "transform_edges")
            """
            try:
                # Resolve class name from cache if needed
                resolved_class_name = codebase_utils._resolve_name_from_cache(
                    class_name,
                    self.state.codebase_cache[CODEBASE_DEFS.CLASSES],
                    package_name,
                )

                # Import the class
                cls, error = inspect_utils.import_object(
                    resolved_class_name, package_name
                )
                if error:
                    return {
                        "success": False,
                        "error": error,
                        "suggestion": "Try using search_codebase first to find the correct class path",
                    }

                # Get method source using Pydantic model
                info = inspect_utils.MethodSourceInfo.from_method(cls, method_name)

                # Check if there was an error finding the method
                if info.error:
                    return {
                        "success": False,
                        "error": info.error,
                        "suggestion": f"Use inspect_class('{class_name}') to see available methods",
                    }

                return {"success": True, **info.model_dump(exclude={"error"})}

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "suggestion": "Verify the class and method exist and are installed",
                }

        @mcp.tool()
        async def list_cli_commands(cli_name: str = "napistu") -> Dict[str, Any]:
            """
            List all available CLI commands for Napistu.

            **USE THIS WHEN:**
            - User wants to know what command-line operations are available
            - Looking for data ingestion, integration, or export commands
            - Understanding the CLI structure and hierarchy

            Parameters
            ----------
            cli_name : str, optional
                Which CLI to inspect ("napistu" or "napistu.mcp")

            Returns
            -------
            Dict[str, Any]
                Dictionary containing all CLI commands with their paths, arguments, options
            """
            try:
                if cli_name == "napistu":
                    from napistu.__main__ import cli
                elif cli_name == "napistu.mcp":
                    from napistu.mcp.__main__ import cli
                else:
                    return {"error": f"Unknown CLI: {cli_name}"}

                from napistu.mcp import cli_utils

                structure = cli_utils.CLIStructure.from_cli_group(cli, cli_name)

                # Convert Pydantic models to dicts for JSON serialization
                commands_dict = {
                    path: cmd_info.model_dump()
                    for path, cmd_info in structure.commands.items()
                }

                return {
                    "success": True,
                    "cli_name": cli_name,
                    "total_commands": len(structure.commands),
                    "commands": commands_dict,
                }

            except Exception as e:
                return {"success": False, "error": str(e)}

        @mcp.tool()
        async def list_installed_packages() -> Dict[str, Any]:
            """
            List all installed Python packages available on the server.

            **USE THIS WHEN:**
            - Determining which packages are available for inspection
            - Finding package names to use with inspect_class, inspect_function, or inspect_method
            - Understanding what libraries are installed in the server environment
            - Checking if a specific package is available before attempting inspection

            **DO NOT USE FOR:**
            - Getting package documentation (use inspect_class/inspect_function for that)
            - Checking package versions for compatibility (this shows versions but isn't for validation)
            - Installing packages (this is read-only)

            Returns
            -------
            Dict[str, Any]
                Dictionary containing:
                - success : bool
                    Whether the operation succeeded
                - total_packages : int
                    Total number of installed packages
                - packages : List[Dict[str, str]]
                    List of package dictionaries, each containing:
                    - name : str
                        Package name
                    - version : str
                        Installed version

            Examples
            --------
            >>> list_installed_packages()
            # Returns all installed packages like:
            # {
            #   "success": True,
            #   "total_packages": 150,
            #   "packages": [
            #     {"name": "napistu", "version": "0.7.7"},
            #     {"name": "numpy", "version": "1.26.0"},
            #     ...
            #   ]
            # }

            Notes
            -----
            This tool helps identify which packages are available for use with
            inspect_class, inspect_function, and inspect_method. Use the package
            names returned here as the package_name parameter in those tools.
            """
            try:
                from napistu.mcp import inspect_utils

                packages = inspect_utils.list_installed_packages()

                return {
                    "success": True,
                    "total_packages": len(packages),
                    "packages": packages,
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "suggestion": "Unable to list installed packages. Check Python environment.",
                }

        @mcp.tool()
        async def search_codebase(
            query: str,
            search_type: str = SEARCH_TYPES.SEMANTIC,
            n_results: int = 5,
            max_exact_results: int = 20,
        ) -> Dict[str, Any]:
            self._check_initialized()
            """
            Search Napistu codebase documentation with intelligent search strategy.

            Provides flexible search capabilities for finding relevant Napistu API documentation
            using either AI-powered semantic search for natural language queries or exact text
            matching for precise keyword searches. Covers modules, classes, and functions from
            the Napistu codebase.

            **USE THIS WHEN:**
            - Looking for specific Napistu functions, classes, or modules
            - Finding API documentation for Napistu features
            - Searching for function signatures, parameters, or return types
            - Understanding Napistu class hierarchies and method documentation
            - Finding implementation details for Napistu functionality

            **DO NOT USE FOR:**
            - General programming concepts not specific to Napistu
            - Documentation for other libraries or frameworks
            - Conceptual tutorials or usage examples (use tutorials component)
            - Installation or setup instructions (use documentation component)
            - Academic research not involving Napistu implementation

            **EXAMPLE APPROPRIATE QUERIES:**
            - "consensus network creation functions"
            - "SBML parsing classes"
            - "pathway analysis methods"
            - "graph algorithms in Napistu"
            - "data ingestion API"

            **EXAMPLE INAPPROPRIATE QUERIES:**
            - "how to install Python" (not Napistu-specific)
            - "general graph theory" (too broad, not API-focused)
            - "pandas DataFrame methods" (wrong library)

            Parameters
            ----------
            query : str
                Search term or natural language question about Napistu API.
                Should be specific to Napistu functions, classes, or modules.
            search_type : str, optional
                Search strategy to use:
                - "semantic" (default): AI-powered search using embeddings
                - "exact": Traditional text matching search
                Default is "semantic".
            n_results : int, optional
                Maximum number of results to return. Results are ranked by similarity score.
                Default is 5.
            max_exact_results : int, optional
                Only applicable when search_type is "exact". If more than max_exact_results are found,
                an error will be returned rather than returning all results.

            Returns
            -------
            Dict[str, Any]
                Search results dictionary containing:
                - query : str
                    Original search query
                - search_type : str
                    Actual search type used ("semantic" or "exact")
                - results : List[Dict] or Dict[str, List]
                    Search results. Format depends on search type:
                    - Semantic: List of result dictionaries with content, metadata, source, similarity_score
                    - Exact: Dictionary organized by code element type (modules, classes, functions)
                - tip : str
                    Helpful guidance for improving search results

            Examples
            --------
            Natural language semantic search for Napistu API:

            >>> results = await search_codebase("functions for creating networks")
            >>> print(results["search_type"])  # "semantic"
            >>> for result in results["results"]:
            ...     score = result['similarity_score']
            ...     print(f"Score: {score:.3f} - {result['source']}")

            Exact keyword search for specific API elements:

            >>> results = await search_codebase("create_consensus", search_type="exact")
            >>> print(len(results["results"]["functions"]))  # Number of matching functions

            Notes
            -----
            **CONTENT SCOPE:**
            This tool searches only Napistu API documentation including:
            - Function signatures, parameters, and return types
            - Class definitions, methods, and attributes
            - Module organization and structure
            - Technical API reference information

            **SEARCH TYPE GUIDANCE:**
            - Use semantic (default) for conceptual API queries and natural language
            - Use exact for precise function names, class names, or known API terms

            **RESULT INTERPRETATION:**
            - Semantic results include similarity scores (0.8-1.0 = very relevant)
            - Results may include chunked sections from long documentation for precision
            - Follow up with get_function_documentation() or get_class_documentation() for complete details

            The function automatically handles semantic search failures by falling back
            to exact search, ensuring reliable results even if AI components are unavailable.
            """
            if search_type == SEARCH_TYPES.SEMANTIC and self.state.semantic_search:
                # Use shared semantic search instance
                results = self.state.semantic_search.search(
                    query, MCP_COMPONENTS.CODEBASE, n_results=n_results
                )
                return {
                    SEARCH_RESULT_DEFS.QUERY: query,
                    SEARCH_RESULT_DEFS.SEARCH_TYPE: SEARCH_TYPES.SEMANTIC,
                    SEARCH_RESULT_DEFS.RESULTS: results,
                    SEARCH_RESULT_DEFS.TIP: "For Napistu API documentation only. Try different phrasings if results aren't relevant, or use search_type='exact' for precise keyword matching",
                }
            else:
                # Fall back to exact search
                return codebase_utils._exact_search_codebase(
                    query, self.state.codebase_cache, max_exact_results
                )

    def _check_initialized(self) -> None:
        """
        Check if component is initialized, raise clear error if not.

        Distinguishes between:
        - Still initializing (initialized=False)
        - Failed initialization (initialized=True, initialization_error set)
        """
        if not self.state.initialized:
            raise RuntimeError(
                "CodebaseComponent is still initializing. "
                "This component loads documentation from ReadTheDocs, which may take several minutes. "
                "Please wait a moment and try again. "
                "You can check initialization status using the health check endpoint."
            )
        elif self.state.initialization_error:
            raise RuntimeError(
                f"CodebaseComponent failed to initialize: {self.state.initialization_error}. "
                "This component requires ReadTheDocs documentation to function. "
                "Please check the server logs for details or try again later."
            )

    def _create_state(self) -> CodebaseState:
        """
        Create codebase-specific state instance.

        Returns
        -------
        CodebaseState
            New state instance for managing codebase content and semantic search
        """
        return CodebaseState()

    async def _initialize_semantic_search(self) -> bool:
        """
        Index codebase content into the shared semantic search instance.

        Uses the shared semantic search instance (stored in self.state.semantic_search)
        to index this component's codebase content into the "codebase" collection.

        Returns
        -------
        bool
            True if content was successfully indexed, False if indexing failed

        Notes
        -----
        Assumes self.state.semantic_search has already been set to a valid
        SemanticSearch instance during initialize().

        Failure to index content is not considered a critical error.
        The component continues to function with exact text search if semantic
        search indexing fails.
        """
        try:
            if not self.state.semantic_search:
                logger.warning("No semantic search instance available")
                return False

            logger.info("Indexing codebase content for semantic search...")

            # Index codebase content using the shared semantic search instance
            self.state.semantic_search.index_content(
                MCP_COMPONENTS.CODEBASE, self.state.codebase_cache
            )

            logger.info("✅ Codebase content indexed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to index codebase content: {e}")
            return False


# Module-level component instance
_component = CodebaseComponent()


def get_component() -> CodebaseComponent:
    """
    Get the codebase component instance.

    Returns
    -------
    CodebaseComponent
        Singleton codebase component instance for use across the MCP server.
        The same instance is returned on every call to ensure consistent state.

    Notes
    -----
    This function provides the standard interface for accessing the codebase
    component. The component must be initialized via safe_initialize() before use.
    """
    return _component
