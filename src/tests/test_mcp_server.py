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
    documentation_component = documentation.get_component()
    documentation_component.register(mcp)

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
    codebase_component = codebase.get_component()
    codebase_component.register(mcp)

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
    tutorials_component = tutorials.get_component()
    tutorials_component.register(mcp)

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

    # Create execution component with test session context
    test_session_context = {}
    test_object_registry = {}
    execution_component = execution.create_component(
        session_context=test_session_context, object_registry=test_object_registry
    )
    execution_component.register(mcp)

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

    # Register all components using new class-based API
    documentation_component = documentation.get_component()
    documentation_component.register(mcp)

    codebase_component = codebase.get_component()
    codebase_component.register(mcp)

    tutorials_component = tutorials.get_component()
    tutorials_component.register(mcp)

    # Create execution component with test context
    test_session_context = {}
    test_object_registry = {}
    execution_component = execution.create_component(
        session_context=test_session_context, object_registry=test_object_registry
    )
    execution_component.register(mcp)

    # Health component still uses legacy registration
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

    # Register all components using new class-based API
    documentation_component = documentation.get_component()
    documentation_component.register(mcp)

    codebase_component = codebase.get_component()
    codebase_component.register(mcp)

    tutorials_component = tutorials.get_component()
    tutorials_component.register(mcp)

    # Create execution component with test context
    test_session_context = {}
    test_object_registry = {}
    execution_component = execution.create_component(
        session_context=test_session_context, object_registry=test_object_registry
    )
    execution_component.register(mcp)

    # Health component still uses legacy registration
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

    # Register all components using new class-based API
    documentation_component = documentation.get_component()
    documentation_component.register(mcp)

    codebase_component = codebase.get_component()
    codebase_component.register(mcp)

    tutorials_component = tutorials.get_component()
    tutorials_component.register(mcp)

    # Create execution component with test context
    test_session_context = {}
    test_object_registry = {}
    execution_component = execution.create_component(
        session_context=test_session_context, object_registry=test_object_registry
    )
    execution_component.register(mcp)

    # Health component still uses legacy registration
    health.register_components(mcp)

    _, tool_names = collect_resources_and_tools(mcp)

    # Debug: Print all registered tools
    print("\nRegistered tools:")
    for name in sorted(tool_names):
        print(f"  {name}")
    print()

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


def test_component_state_health():
    """Test that component states can provide health information."""
    # Test documentation component state
    doc_component = documentation.get_component()
    doc_state = doc_component.get_state()
    health_status = doc_state.get_health_status()
    assert "status" in health_status
    assert health_status["status"] in [
        "initializing",
        "healthy",
        "inactive",
        "unavailable",
    ]

    # Test codebase component state
    codebase_component = codebase.get_component()
    codebase_state = codebase_component.get_state()
    health_status = codebase_state.get_health_status()
    assert "status" in health_status
    assert health_status["status"] in [
        "initializing",
        "healthy",
        "inactive",
        "unavailable",
    ]

    # Test tutorials component state
    tutorials_component = tutorials.get_component()
    tutorials_state = tutorials_component.get_state()
    health_status = tutorials_state.get_health_status()
    assert "status" in health_status
    assert health_status["status"] in [
        "initializing",
        "healthy",
        "inactive",
        "unavailable",
    ]

    # Test execution component state
    test_session_context = {"test": "value"}
    test_object_registry = {}
    execution_component = execution.create_component(
        session_context=test_session_context, object_registry=test_object_registry
    )
    execution_state = execution_component.get_state()
    health_status = execution_state.get_health_status()
    assert "status" in health_status
    assert health_status["status"] in [
        "initializing",
        "healthy",
        "inactive",
        "unavailable",
    ]


def test_execution_object_registration():
    """Test that execution component can register and track objects."""
    test_session_context = {}
    test_object_registry = {}
    execution_component = execution.create_component(
        session_context=test_session_context, object_registry=test_object_registry
    )

    # Register a test object
    test_object = {"test": "data"}
    execution_component.register_object("test_obj", test_object)

    # Check that object was registered
    state = execution_component.get_state()
    assert "test_obj" in state.session_objects
    assert state.session_objects["test_obj"] == test_object

    # Check health details include the object
    health_details = state.get_health_details()
    assert "registered_objects" in health_details
    assert health_details["registered_objects"] == 1
    assert "test_obj" in health_details["object_names"]
