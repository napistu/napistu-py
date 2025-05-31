"""
Tests to validate MCP tool and resource naming conventions.
"""

import re
from typing import List, Tuple
from fastmcp import FastMCP

from napistu.mcp import (
    documentation,
    codebase,
    tutorials,
    execution,
    health,
)

# Regex patterns for validation
VALID_RESOURCE_PATH = re.compile(
    r"^napistu://[a-zA-Z][a-zA-Z0-9_]*(?:/[a-zA-Z0-9_{}.-]+)*$"
)
VALID_TOOL_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def collect_resources_and_tools(mcp: FastMCP) -> Tuple[List[str], List[str]]:
    """
    Collect all registered resource paths and tool names from an MCP server instance.

    Args:
        mcp: FastMCP server instance

    Returns:
        Tuple of (resource_paths, tool_names)
    """
    # Get all registered resources and tools
    resources = mcp._resource_manager.get_resources()
    tool_names = mcp._tool_manager.get_tools()

    # Extract resource paths from the resource objects
    resource_paths = []

    # Add all resources (including parameterized ones)
    for resource in resources.values():
        resource_paths.append(str(resource.uri))

    return resource_paths, tool_names


def test_documentation_naming():
    """Test that documentation component uses valid names."""
    mcp = FastMCP("test")
    documentation.register_components(mcp)

    resource_paths, tool_names = collect_resources_and_tools(mcp)

    # Check resource paths
    for path in resource_paths:
        assert VALID_RESOURCE_PATH.match(path), f"Invalid resource path: {path}"

    # Check tool names
    for name in tool_names:
        assert VALID_TOOL_NAME.match(name), f"Invalid tool name: {name}"


def test_codebase_naming():
    """Test that codebase component uses valid names."""
    mcp = FastMCP("test")
    codebase.register_components(mcp)

    resource_paths, tool_names = collect_resources_and_tools(mcp)

    # Check resource paths
    for path in resource_paths:
        assert VALID_RESOURCE_PATH.match(path), f"Invalid resource path: {path}"

    # Check tool names
    for name in tool_names:
        assert VALID_TOOL_NAME.match(name), f"Invalid tool name: {name}"


def test_tutorials_naming():
    """Test that tutorials component uses valid names."""
    mcp = FastMCP("test")
    tutorials.register_components(mcp)

    resource_paths, tool_names = collect_resources_and_tools(mcp)

    # Check resource paths
    for path in resource_paths:
        assert VALID_RESOURCE_PATH.match(path), f"Invalid resource path: {path}"

    # Check tool names
    for name in tool_names:
        assert VALID_TOOL_NAME.match(name), f"Invalid tool name: {name}"


def test_execution_naming():
    """Test that execution component uses valid names."""
    mcp = FastMCP("test")
    execution.register_components(mcp)

    resource_paths, tool_names = collect_resources_and_tools(mcp)

    # Check resource paths
    for path in resource_paths:
        assert VALID_RESOURCE_PATH.match(path), f"Invalid resource path: {path}"

    # Check tool names
    for name in tool_names:
        assert VALID_TOOL_NAME.match(name), f"Invalid tool name: {name}"


def test_health_naming():
    """Test that health component uses valid names."""
    mcp = FastMCP("test")
    health.register_components(mcp)

    resource_paths, tool_names = collect_resources_and_tools(mcp)

    # Check resource paths
    for path in resource_paths:
        assert VALID_RESOURCE_PATH.match(path), f"Invalid resource path: {path}"

    # Check tool names
    for name in tool_names:
        assert VALID_TOOL_NAME.match(name), f"Invalid tool name: {name}"


def test_all_components_naming():
    """Test that all components together use valid names without conflicts."""
    mcp = FastMCP("test")

    # Register all components
    documentation.register_components(mcp)
    codebase.register_components(mcp)
    tutorials.register_components(mcp)
    execution.register_components(mcp)
    health.register_components(mcp)

    resource_paths, tool_names = collect_resources_and_tools(mcp)

    # Check for duplicate resource paths
    path_counts = {}
    for path in resource_paths:
        assert VALID_RESOURCE_PATH.match(path), f"Invalid resource path: {path}"
        path_counts[path] = path_counts.get(path, 0) + 1
        assert path_counts[path] == 1, f"Duplicate resource path: {path}"

    # Check for duplicate tool names
    name_counts = {}
    for name in tool_names:
        assert VALID_TOOL_NAME.match(name), f"Invalid tool name: {name}"
        name_counts[name] = name_counts.get(name, 0) + 1
        assert name_counts[name] == 1, f"Duplicate tool name: {name}"


def test_expected_resources_exist():
    """Test that all expected resources are registered."""
    mcp = FastMCP("test")

    # Register all components
    documentation.register_components(mcp)
    codebase.register_components(mcp)
    tutorials.register_components(mcp)
    execution.register_components(mcp)
    health.register_components(mcp)

    resource_paths, _ = collect_resources_and_tools(mcp)

    # Debug: Print all registered resources
    print("\nRegistered resources:")
    for path in sorted(resource_paths):
        print(f"  {path}")
    print()

    # List of expected base resources (no templates)
    expected_resources = {
        "napistu://documentation/summary",
        "napistu://codebase/summary",
        "napistu://tutorials/index",
        "napistu://execution/registry",
        "napistu://execution/environment",
        "napistu://health",
    }

    # Check that each expected resource exists
    for resource in expected_resources:
        assert resource in resource_paths, f"Missing expected resource: {resource}"


def test_expected_tools_exist():
    """Test that all expected tools are registered."""
    mcp = FastMCP("test")

    # Register all components
    documentation.register_components(mcp)
    codebase.register_components(mcp)
    tutorials.register_components(mcp)
    execution.register_components(mcp)
    health.register_components(mcp)

    _, tool_names = collect_resources_and_tools(mcp)

    # List of expected tools
    expected_tools = {
        "search_documentation",
        "search_codebase",
        "get_function_documentation",
        "get_class_documentation",
        "search_tutorials",
        "list_registry",
        "describe_object",
        "execute_function",
        "search_paths",
        "check_health",
    }

    # Check that all expected tools exist
    for tool in expected_tools:
        assert tool in tool_names, f"Missing expected tool: {tool}"
