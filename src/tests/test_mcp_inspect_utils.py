"""Tests for inspect_utils module."""

from napistu.mcp.inspect_utils import (
    ClassInfo,
    FunctionInfo,
    MethodSourceInfo,
    import_object,
    list_installed_packages,
)


def test_function_info_from_function():
    """Test FunctionInfo.from_function with a real function."""
    func, error = import_object("consensus.construct_consensus_model", "napistu")
    assert error is None, f"Failed to import function: {error}"

    info = FunctionInfo.from_function(func)

    assert info.signature is not None
    assert "sbml_dfs_dict" in info.signature  # Check for a parameter name
    assert info.source is not None
    assert "def construct_consensus_model" in info.source
    assert info.file_path is not None
    assert info.line_number is not None
    assert isinstance(info.line_number, int)


def test_class_info_from_class():
    """Test ClassInfo.from_class with include_init=True and include_init=False."""
    cls, error = import_object("sbml_dfs_core.SBML_dfs", "napistu")
    assert error is None, f"Failed to import class: {error}"

    # Test with include_init=True
    info = ClassInfo.from_class(cls, include_init=True)

    assert info.name == "SBML_dfs"
    assert info.module is not None
    assert "sbml_dfs_core" in info.module
    assert info.init_source is not None
    assert "__init__" in info.init_source or "def __init__" in info.init_source
    assert isinstance(info.methods, dict)
    assert len(info.methods) > 0
    assert "get_identifiers" in info.methods

    # Test with include_init=False
    info_no_init = ClassInfo.from_class(cls, include_init=False)

    assert info_no_init.name == "SBML_dfs"
    assert info_no_init.init_signature is not None
    assert info_no_init.init_source is None


def test_method_source_info_from_method():
    """Test MethodSourceInfo.from_method with both existing and non-existent methods."""
    cls, error = import_object("sbml_dfs_core.SBML_dfs", "napistu")
    assert error is None, f"Failed to import class: {error}"

    # Test with an existing public method
    method_info = MethodSourceInfo.from_method(cls, "get_identifiers")

    assert method_info.method_name == "get_identifiers"
    assert method_info.class_name == "SBML_dfs"
    assert method_info.source is not None
    assert "def get_identifiers" in method_info.source
    assert method_info.signature is not None
    assert method_info.line_number is not None
    assert isinstance(method_info.line_number, int)
    assert method_info.error is None

    # Test with an existing private method
    private_method_info = MethodSourceInfo.from_method(
        cls, "_get_non_interactor_reactions"
    )

    assert private_method_info.method_name == "_get_non_interactor_reactions"
    assert private_method_info.class_name == "SBML_dfs"
    assert private_method_info.source is not None
    assert "def _get_non_interactor_reactions" in private_method_info.source
    assert private_method_info.signature is not None
    assert private_method_info.error is None

    # Test with a non-existent method
    method_info_not_found = MethodSourceInfo.from_method(cls, "nonexistent_method")

    assert method_info_not_found.method_name == "nonexistent_method"
    assert method_info_not_found.class_name == "SBML_dfs"
    assert method_info_not_found.error is not None
    assert "not found" in method_info_not_found.error.lower()
    assert method_info_not_found.source is None


def test_list_installed_packages():
    """Test list_installed_packages returns packages and validates igraph and pandas."""
    packages = list_installed_packages()

    assert isinstance(packages, list)
    assert len(packages) > 0

    # Check that all packages have the expected structure
    for pkg in packages:
        assert "name" in pkg
        assert "version" in pkg
        assert isinstance(pkg["name"], str)
        assert isinstance(pkg["version"], str)

    # Validate that igraph and pandas are available
    package_names = [pkg["name"].lower() for pkg in packages]

    assert "igraph" in package_names, "igraph package should be installed"
    assert "pandas" in package_names, "pandas package should be installed"

    # Get the actual package info for verification
    igraph_pkg = next((p for p in packages if p["name"].lower() == "igraph"), None)
    pandas_pkg = next((p for p in packages if p["name"].lower() == "pandas"), None)

    assert igraph_pkg is not None, "igraph package not found in installed packages"
    assert pandas_pkg is not None, "pandas package not found in installed packages"
    assert igraph_pkg["version"] is not None, "igraph should have a version"
    assert pandas_pkg["version"] is not None, "pandas should have a version"
