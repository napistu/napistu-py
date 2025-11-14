"""Tests for cli_utils module using the actual Napistu CLI."""

from napistu.__main__ import cli
from napistu.mcp import __main__ as mcp_main
from napistu.mcp.cli_utils import CLICommandInfo, CLIStructure


def test_cli_command_info_from_click_command():
    """Test CLICommandInfo.from_click_command with a real CLI command."""
    # Get the ingestion group
    ingestion_group = cli.commands.get("ingestion")
    assert ingestion_group is not None, "ingestion command not found"

    # Test with the ingestion group
    info = CLICommandInfo.from_click_command(ingestion_group, "napistu")

    assert info.name == "ingestion"
    assert info.path == "napistu ingestion"
    assert info.help_text is not None
    assert isinstance(info.subcommands, list)
    assert len(info.subcommands) > 0


def test_cli_command_info_with_reactome_command():
    """Test CLICommandInfo.from_click_command with a specific command."""
    # Get the reactome command under ingestion
    ingestion_group = cli.commands.get("ingestion")
    assert ingestion_group is not None

    reactome_cmd = ingestion_group.commands.get("reactome")
    assert reactome_cmd is not None, "reactome command not found"

    info = CLICommandInfo.from_click_command(reactome_cmd, "napistu ingestion")

    assert info.name == "reactome"
    assert info.path == "napistu ingestion reactome"
    assert isinstance(info.arguments, list)
    assert isinstance(info.options, list)
    # Check that arguments/options have the expected structure
    for arg in info.arguments:
        assert "name" in arg
        assert "type" in arg
        assert "required" in arg
    for opt in info.options:
        assert "name" in opt
        assert "flags" in opt
        assert "type" in opt


def test_cli_structure_from_cli_group():
    """Test CLIStructure.from_cli_group with the actual Napistu CLI."""
    structure = CLIStructure.from_cli_group(cli, "napistu")

    assert isinstance(structure.commands, dict)
    assert len(structure.commands) > 0

    # Check that we can find the ingestion reactome command
    assert "napistu ingestion reactome" in structure.commands

    # Verify the structure of a command
    reactome_info = structure.commands["napistu ingestion reactome"]
    assert isinstance(reactome_info, CLICommandInfo)
    assert reactome_info.name == "reactome"
    assert reactome_info.path == "napistu ingestion reactome"


def test_cli_command_info_with_mcp_health_command():
    """Test CLICommandInfo.from_click_command with MCP CLI health command."""
    mcp_cli = mcp_main.cli
    health_cmd = mcp_cli.commands.get("health")
    assert health_cmd is not None, "health command not found in MCP CLI"

    info = CLICommandInfo.from_click_command(health_cmd, "napistu-mcp")

    assert info.name == "health"
    assert info.path == "napistu-mcp health"
    assert isinstance(info.options, list)
    assert len(info.options) > 0
    # Check that options have the expected structure
    for opt in info.options:
        assert "name" in opt
        assert "flags" in opt
        assert "type" in opt
        assert "required" in opt


def test_cli_command_info_with_mcp_server_start_command():
    """Test CLICommandInfo.from_click_command with MCP CLI server start command."""
    mcp_cli = mcp_main.cli
    server_group = mcp_cli.commands.get("server")
    assert server_group is not None, "server command not found in MCP CLI"

    start_cmd = server_group.commands.get("start")
    assert start_cmd is not None, "server start command not found"

    info = CLICommandInfo.from_click_command(start_cmd, "napistu-mcp server")

    assert info.name == "start"
    assert info.path == "napistu-mcp server start"
    assert isinstance(info.options, list)
    # Check that the profile option exists with a default value
    profile_opt = next((opt for opt in info.options if opt["name"] == "profile"), None)
    assert profile_opt is not None, "profile option not found"
    assert profile_opt["default"] == "docs", "profile default should be 'docs'"


def test_cli_structure_from_mcp_cli():
    """Test CLIStructure.from_cli_group with the MCP CLI."""
    mcp_cli = mcp_main.cli
    structure = CLIStructure.from_cli_group(mcp_cli, "napistu-mcp")

    assert isinstance(structure.commands, dict)
    assert len(structure.commands) > 0

    # Check that we can find the health command
    assert "napistu-mcp health" in structure.commands

    # Check that we can find the server start command
    assert "napistu-mcp server start" in structure.commands

    # Verify the structure of a command
    health_info = structure.commands["napistu-mcp health"]
    assert isinstance(health_info, CLICommandInfo)
    assert health_info.name == "health"
    assert health_info.path == "napistu-mcp health"
