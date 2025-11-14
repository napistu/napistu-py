"""
Utilities for runtime Python package inspection.

Provides functions to introspect installed Python packages and extract
source code, signatures, and metadata from functions, classes, and methods.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class FunctionInfo(BaseModel):
    """Information about a function."""

    source: Optional[str] = Field(None, description="Full source code")
    signature: Optional[str] = Field(
        None, description="Function signature with type hints"
    )
    docstring: Optional[str] = Field(None, description="Function docstring")
    file_path: Optional[str] = Field(
        None, description="Path to file containing the function"
    )
    line_number: Optional[int] = Field(None, description="Starting line number")

    @classmethod
    def from_function(cls, func: Any) -> "FunctionInfo":
        """
        Create FunctionInfo from a function object.

        Parameters
        ----------
        func : Any
            Function object to inspect

        Returns
        -------
        FunctionInfo
            Pydantic model containing function details
        """
        # Get source code
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = None

        # Get signature
        try:
            signature = str(inspect.signature(func))
        except (ValueError, TypeError):
            signature = None

        # Get docstring
        docstring = inspect.getdoc(func)

        # Get file location
        try:
            file_path = inspect.getfile(func)
            line_number = inspect.getsourcelines(func)[1]
        except (OSError, TypeError):
            file_path = None
            line_number = None

        return cls(
            source=source,
            signature=signature,
            docstring=docstring,
            file_path=file_path,
            line_number=line_number,
        )


class MethodInfo(BaseModel):
    """Information about a single method."""

    signature: Optional[str] = Field(None, description="Method signature")
    docstring: Optional[str] = Field(None, description="Method docstring")


class ClassInfo(BaseModel):
    """Information about a class."""

    name: str = Field(description="Class name")
    module: str = Field(description="Module where class is defined")
    docstring: Optional[str] = Field(None, description="Class docstring")
    file_path: Optional[str] = Field(
        None, description="Path to file containing the class"
    )
    line_number: Optional[int] = Field(None, description="Starting line number")
    init_signature: Optional[str] = Field(None, description="__init__ method signature")
    init_source: Optional[str] = Field(None, description="__init__ source code")
    methods: Dict[str, MethodInfo] = Field(
        default_factory=dict, description="Dictionary of methods"
    )

    @classmethod
    def from_class(cls, class_obj: Any, include_init: bool = True) -> "ClassInfo":
        """
        Create ClassInfo from a class object.

        Parameters
        ----------
        class_obj : Any
            Class object to inspect
        include_init : bool, optional
            Whether to include __init__ source code (default: True)

        Returns
        -------
        ClassInfo
            Pydantic model containing class details
        """
        # Get docstring
        docstring = inspect.getdoc(class_obj)

        # Get file location
        try:
            file_path = inspect.getfile(class_obj)
            line_number = inspect.getsourcelines(class_obj)[1]
        except (OSError, TypeError):
            file_path = None
            line_number = None

        # Get __init__ info
        init_signature = None
        init_source = None
        if hasattr(class_obj, "__init__"):
            init_method = class_obj.__init__
            try:
                init_signature = str(inspect.signature(init_method))
            except (ValueError, TypeError):
                pass

            if include_init:
                try:
                    init_source = inspect.getsource(init_method)
                except (OSError, TypeError):
                    pass

        # Get methods (excluding private unless __init__)
        methods = {}
        for name, method in inspect.getmembers(class_obj, inspect.isfunction):
            if not name.startswith("_") or name == "__init__":
                try:
                    sig = str(inspect.signature(method))
                except (ValueError, TypeError):
                    sig = None

                methods[name] = MethodInfo(
                    signature=sig, docstring=inspect.getdoc(method)
                )

        return cls(
            name=class_obj.__name__,
            module=class_obj.__module__,
            docstring=docstring,
            file_path=file_path,
            line_number=line_number,
            init_signature=init_signature,
            init_source=init_source,
            methods=methods,
        )


class MethodSourceInfo(BaseModel):
    """Detailed source information for a specific method."""

    method_name: str = Field(description="Name of the method")
    class_name: str = Field(description="Name of the containing class")
    source: Optional[str] = Field(None, description="Method source code")
    signature: Optional[str] = Field(None, description="Method signature")
    docstring: Optional[str] = Field(None, description="Method docstring")
    line_number: Optional[int] = Field(None, description="Starting line number")
    error: Optional[str] = Field(None, description="Error message if method not found")

    @classmethod
    def from_method(cls, class_obj: Any, method_name: str) -> "MethodSourceInfo":
        """
        Create MethodSourceInfo from a class and method name.

        Parameters
        ----------
        class_obj : Any
            Class object containing the method
        method_name : str
            Name of the method to extract

        Returns
        -------
        MethodSourceInfo
            Pydantic model containing method details
        """
        # Check if method exists
        if not hasattr(class_obj, method_name):
            return cls(
                method_name=method_name,
                class_name=class_obj.__name__,
                error=f"Method '{method_name}' not found on class '{class_obj.__name__}'",
            )

        method = getattr(class_obj, method_name)

        # Get source
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError):
            source = None

        # Get signature
        try:
            signature = str(inspect.signature(method))
        except (ValueError, TypeError):
            signature = None

        # Get docstring
        docstring = inspect.getdoc(method)

        # Get line number
        try:
            line_number = inspect.getsourcelines(method)[1]
        except (OSError, TypeError):
            line_number = None

        return cls(
            method_name=method_name,
            class_name=class_obj.__name__,
            source=source,
            signature=signature,
            docstring=docstring,
            line_number=line_number,
        )


def import_object(
    full_path: str, package_name: str = "napistu"
) -> Tuple[Any, Optional[str]]:
    """
    Import and return an object from a dotted path.

    Parameters
    ----------
    full_path : str
        Dotted path to the object. Can be:
        - Short form: "network.create_consensus" (uses package_name)
        - Full form: "napistu.network.create_consensus" (ignores package_name)
        - Class form: "sbml_dfs_core.SBML_dfs" (uses package_name)
    package_name : str, optional
        Default package name if full_path doesn't include it (default: "napistu")

    Returns
    -------
    Tuple[Any, Optional[str]]
        (object, error_message) where error_message is None on success

    Examples
    --------
    >>> obj, error = import_object("network.create_consensus", "napistu")
    >>> if error:
    ...     print(f"Import failed: {error}")
    ... else:
    ...     print(f"Got object: {obj}")

    >>> cls, error = import_object("sbml_dfs_core.SBML_dfs", "napistu")
    >>> if not error:
    ...     print(f"Got class: {cls.__name__}")
    """
    # If the path doesn't start with a known package, prepend package_name
    if not full_path.startswith(package_name):
        full_path = f"{package_name}.{full_path}"

    # Split into module path and object name
    parts = full_path.split(".")

    # Try progressively shorter module paths, starting from longest
    # This handles cases like:
    # - napistu.network.create_consensus (module: napistu.network, obj: create_consensus)
    # - napistu.sbml_dfs_core.SBML_dfs (module: napistu.sbml_dfs_core, obj: SBML_dfs)
    last_error = None

    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        obj_path = parts[i:]

        try:
            # Try to import the module
            module = importlib.import_module(module_path)

            # If no remaining path, return the module itself
            if not obj_path:
                return module, None

            # Navigate to the object through the remaining path
            obj = module
            for attr_name in obj_path:
                obj = getattr(obj, attr_name)

            return obj, None

        except ImportError as e:
            # Module doesn't exist, try shorter path
            last_error = e
            continue
        except AttributeError as e:
            # Module exists but doesn't have the attribute, try shorter path
            last_error = e
            continue

    # If we got here, nothing worked
    error_msg = f"Could not import '{full_path}'. "
    if last_error:
        error_msg += f"Last error: {str(last_error)}"
    else:
        error_msg += "Module or object not found."

    return None, error_msg


def list_installed_packages() -> List[Dict[str, str]]:
    """
    List all installed Python packages with their versions.

    Returns
    -------
    List[Dict[str, str]]
        List of package dictionaries, each containing:
        - name : str
            Package name
        - version : str
            Installed version
        Packages are sorted alphabetically by name.

    Examples
    --------
    >>> packages = list_installed_packages()
    >>> pandas_pkg = next((p for p in packages if p["name"] == "pandas"), None)
    >>> if pandas_pkg:
    ...     print(f"pandas version: {pandas_pkg['version']}")
    """
    try:
        # Python 3.8+ - preferred method
        from importlib.metadata import distributions

        packages = []
        for dist in distributions():
            try:
                name = dist.metadata["Name"]
                version = dist.metadata["Version"]
                if name:  # Skip packages without names
                    packages.append({"name": name, "version": version})
            except (KeyError, AttributeError):
                # Skip packages with missing metadata
                continue

        # Sort by name for consistent output
        packages.sort(key=lambda x: x["name"].lower())
        return packages

    except ImportError:
        # Fallback to pkg_resources for older Python versions
        import pkg_resources

        packages = []
        for dist in pkg_resources.working_set:
            packages.append({"name": dist.project_name, "version": dist.version})

        packages.sort(key=lambda x: x["name"].lower())
        return packages
