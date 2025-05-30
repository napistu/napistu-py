# src/napistu/mcp/__main__.py
"""
MCP (Model Context Protocol) Server CLI for Napistu.
"""

import click
import click_logging
import logging

import napistu
from napistu.mcp.server import start_mcp_server

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
@click.option("--profile", type=click.Choice(["local", "remote", "full"]), default="remote")
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=8080)
@click.option("--server-name", type=str)
@click.option("--transport", type=click.Choice(["stdio", "http"]), default="stdio", help="Transport type")
@click_logging.simple_verbosity_option(logger)
def start_server(profile, host, port, server_name, transport):
    """Start an MCP server with the specified profile."""
    start_mcp_server(profile, host, port, server_name, transport)


@server.command(name="local")
@click.option("--server-name", type=str, default="napistu-local")
@click_logging.simple_verbosity_option(logger)
def start_local(server_name):
    """Start a local MCP server optimized for function execution."""
    start_mcp_server("local", "127.0.0.1", 8765, server_name, "stdio")


@server.command(name="remote")
@click.option("--server-name", type=str, default="napistu-docs")
@click_logging.simple_verbosity_option(logger)
def start_remote(server_name):
    """Start a remote MCP server for documentation and codebase exploration."""
    start_mcp_server("remote", "0.0.0.0", 8080, server_name, "http")


@server.command(name="full")
@click.option("--server-name", type=str, default="napistu-full")
@click_logging.simple_verbosity_option(logger)
def start_full(server_name):
    """Start a full MCP server with all components enabled (local debugging)."""
    start_mcp_server("full", "127.0.0.1", 8765, server_name, "stdio")


# Add command groups to the CLI
cli.add_command(server)


if __name__ == "__main__":
    cli()