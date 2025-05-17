"""
MCP (Model Context Protocol) Server for Napistu.
"""

import os
import sys
import asyncio
import logging
import click
import click_logging

import napistu
from napistu.mcp.profiles import get_profile, ServerProfile
from napistu.mcp.server import create_server

logger = logging.getLogger(napistu.__name__)
click_logging.basic_config(logger)

@click.group()
def cli():
    """The Napistu MCP (Model Context Protocol) Server CLI"""
    pass

@click.group()
def server():
    """Start and manage MCP servers."""
    pass

@server.command(name="start")
@click.option(
    "--profile", 
    type=click.Choice(["local", "remote", "full"]), 
    default="remote",
    help="Predefined configuration profile"
)
@click.option(
    "--server-name", 
    type=str, 
    help="Name of the MCP server"
)
@click_logging.simple_verbosity_option(logger)
def start_server(profile, server_name):
    """Start an MCP server with the specified profile."""
    # Collect configuration
    config = {}
    if server_name:
        config["server_name"] = server_name
    
    # Get profile with overrides
    server_profile = get_profile(profile, **config)
    
    # Create and start the server
    logger.info(f"Starting Napistu MCP Server with {profile} profile...")
    server = create_server(server_profile)
    asyncio.run(server.start())

@server.command(name="local")
@click.option(
    "--server-name", 
    type=str, 
    default="napistu-local",
    help="Name of the MCP server"
)
@click_logging.simple_verbosity_option(logger)
def start_local(server_name):
    """Start a local MCP server optimized for function execution."""
    # Get profile with overrides
    server_profile = get_profile("local", server_name=server_name)
    
    # Create and start the server
    logger.info("Starting Napistu local MCP Server...")
    server = create_server(server_profile)
    asyncio.run(server.start())

@server.command(name="remote")
@click.option(
    "--server-name", 
    type=str, 
    default="napistu-docs",
    help="Name of the MCP server"
)
@click.option(
    "--codebase-path", 
    type=str,
    help="Path to the Napistu codebase"
)
@click.option(
    "--docs-paths", 
    type=str,
    help="Comma-separated list of paths to documentation files"
)
@click.option(
    "--tutorials-path", 
    type=str,
    help="Path to the tutorials directory"
)
@click_logging.simple_verbosity_option(logger)
def start_remote(server_name, codebase_path, docs_paths, tutorials_path):
    """Start a remote MCP server for documentation and codebase exploration."""
    # Collect configuration
    config = {"server_name": server_name}
    if codebase_path:
        config["codebase_path"] = codebase_path
    if docs_paths:
        config["docs_paths"] = docs_paths.split(",")
    if tutorials_path:
        config["tutorials_path"] = tutorials_path
    
    # Get profile with overrides
    server_profile = get_profile("remote", **config)
    
    # Create and start the server
    logger.info("Starting Napistu remote MCP Server...")
    server = create_server(server_profile)
    asyncio.run(server.start())

@click.group()
def component():
    """Enable or disable specific MCP server components."""
    pass

@component.command(name="list")
def list_components():
    """List available MCP server components."""
    click.echo("Available MCP server components:")
    click.echo("  - documentation: Documentation components")
    click.echo("  - codebase: Codebase exploration components")
    click.echo("  - execution: Function execution components")
    click.echo("  - tutorials: Tutorial components")

@component.command(name="custom")
@click.option(
    "--enable-documentation/--disable-documentation", 
    default=None,
    help="Enable/disable documentation components"
)
@click.option(
    "--enable-codebase/--disable-codebase", 
    default=None,
    help="Enable/disable codebase exploration components"
)
@click.option(
    "--enable-execution/--disable-execution", 
    default=None,
    help="Enable/disable function execution components"
)
@click.option(
    "--enable-tutorials/--disable-tutorials", 
    default=None,
    help="Enable/disable tutorial components"
)
@click.option(
    "--server-name", 
    type=str, 
    default="napistu-custom",
    help="Name of the MCP server"
)
@click.option(
    "--codebase-path", 
    type=str,
    help="Path to the Napistu codebase"
)
@click.option(
    "--docs-paths", 
    type=str,
    help="Comma-separated list of paths to documentation files"
)
@click.option(
    "--tutorials-path", 
    type=str,
    help="Path to the tutorials directory"
)
@click_logging.simple_verbosity_option(logger)
def custom_server(enable_documentation, enable_codebase, enable_execution, 
                  enable_tutorials, server_name, codebase_path, docs_paths, 
                  tutorials_path):
    """Start an MCP server with custom component configuration."""
    # Collect configuration
    config = {"server_name": server_name}
    if enable_documentation is not None:
        config["enable_documentation"] = enable_documentation
    if enable_codebase is not None:
        config["enable_codebase"] = enable_codebase
    if enable_execution is not None:
        config["enable_execution"] = enable_execution
    if enable_tutorials is not None:
        config["enable_tutorials"] = enable_tutorials
    if codebase_path:
        config["codebase_path"] = codebase_path
    if docs_paths:
        config["docs_paths"] = docs_paths.split(",")
    if tutorials_path:
        config["tutorials_path"] = tutorials_path
    
    # Create a custom profile
    server_profile = ServerProfile(**config)
    
    # Create and start the server
    logger.info("Starting Napistu custom MCP Server...")
    server = create_server(server_profile)
    asyncio.run(server.start())

# Add command groups to the CLI
cli.add_command(server)
cli.add_command(component)

if __name__ == "__main__":
    cli()