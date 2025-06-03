# src/napistu/mcp/__main__.py
"""
MCP (Model Context Protocol) Server CLI for Napistu.
"""

import asyncio
import click
import click_logging
import logging

import napistu
from napistu.mcp.server import start_mcp_server
from napistu.mcp.client import (
    check_server_health,
    print_health_status,
    list_server_resources,
    read_server_resource,
)

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
    "--profile", type=click.Choice(["local", "remote", "full"]), default="remote"
)
@click.option("--host", type=str, default="127.0.0.1")
@click.option("--port", type=int, default=8765)
@click.option("--server-name", type=str)
@click_logging.simple_verbosity_option(logger)
def start_server(profile, host, port, server_name):
    """Start an MCP server with the specified profile."""
    start_mcp_server(profile, host, port, server_name)


@server.command(name="local")
@click.option("--server-name", type=str, default="napistu-local")
@click_logging.simple_verbosity_option(logger)
def start_local(server_name):
    """Start a local MCP server optimized for function execution."""
    start_mcp_server("local", "127.0.0.1", 8765, server_name)


@server.command(name="full")
@click.option("--server-name", type=str, default="napistu-full")
@click_logging.simple_verbosity_option(logger)
def start_full(server_name):
    """Start a full MCP server with all components enabled (local debugging)."""
    start_mcp_server("full", "127.0.0.1", 8765, server_name)


@cli.command()
@click.option("--url", default="http://127.0.0.1:8765", help="Server URL")
@click_logging.simple_verbosity_option(logger)
def health(url):
    """Quick health check of MCP server."""

    async def run_health_check():
        print("🏥 Napistu MCP Server Health Check")
        print("=" * 40)
        print(f"Server URL: {url}")
        print()

        health = await check_server_health(server_url=url)
        print_health_status(health)

    asyncio.run(run_health_check())


@cli.command()
@click.option("--url", default="http://127.0.0.1:8765", help="Server URL")
@click_logging.simple_verbosity_option(logger)
def resources(url):
    """List all available resources on the MCP server."""

    async def run_list_resources():
        print("📋 Napistu MCP Server Resources")
        print("=" * 40)
        print(f"Server URL: {url}")
        print()

        resources = await list_server_resources(server_url=url)

        if resources:
            print(f"Found {len(resources)} resources:")
            for resource in resources:
                print(f"  📄 {resource.uri}")
                if resource.name != resource.uri:
                    print(f"      Name: {resource.name}")
                if hasattr(resource, "description") and resource.description:
                    print(f"      Description: {resource.description}")
                print()
        else:
            print("❌ Could not retrieve resources")

    asyncio.run(run_list_resources())


@cli.command()
@click.argument("resource_uri")
@click.option("--url", default="http://127.0.0.1:8765", help="Server URL")
@click.option(
    "--output", type=click.File("w"), default="-", help="Output file (default: stdout)"
)
@click_logging.simple_verbosity_option(logger)
def read(resource_uri, url, output):
    """Read a specific resource from the MCP server."""

    async def run_read_resource():
        print(
            f"📖 Reading Resource: {resource_uri}",
            file=output if output.name != "<stdout>" else None,
        )
        print(f"Server URL: {url}", file=output if output.name != "<stdout>" else None)
        print("=" * 50, file=output if output.name != "<stdout>" else None)

        content = await read_server_resource(resource_uri, server_url=url)

        if content:
            print(content, file=output)
        else:
            print(
                "❌ Could not read resource",
                file=output if output.name != "<stdout>" else None,
            )

    asyncio.run(run_read_resource())


@cli.command()
@click.option("--local-url", default="http://127.0.0.1:8765", help="Local server URL")
@click.option("--remote-url", required=True, help="Remote server URL")
@click_logging.simple_verbosity_option(logger)
def compare(local_url, remote_url):
    """Compare health between local and remote servers."""

    async def run_comparison():
        print("🔍 Local vs Remote Server Comparison")
        print("=" * 50)

        print(f"\n📍 Local Server: {local_url}")
        local_health = await check_server_health(server_url=local_url)
        print_health_status(local_health)

        print(f"\n🌐 Remote Server: {remote_url}")
        remote_health = await check_server_health(server_url=remote_url)
        print_health_status(remote_health)

        # Compare results
        print("\n📊 Comparison Summary:")
        if local_health and remote_health:
            local_components = local_health.get("components", {})
            remote_components = remote_health.get("components", {})

            all_components = set(local_components.keys()) | set(
                remote_components.keys()
            )

            for component in sorted(all_components):
                local_status = local_components.get(component, {}).get(
                    "status", "missing"
                )
                remote_status = remote_components.get(component, {}).get(
                    "status", "missing"
                )

                if local_status == remote_status == "healthy":
                    icon = "✅"
                elif local_status != remote_status:
                    icon = "⚠️ "
                else:
                    icon = "❌"

                print(
                    f"  {icon} {component}: Local={local_status}, Remote={remote_status}"
                )
        else:
            print("  ❌ Cannot compare - one or both servers unreachable")

    asyncio.run(run_comparison())


# Add commands to the CLI
cli.add_command(server)
cli.add_command(health)
cli.add_command(resources)
cli.add_command(read)
cli.add_command(compare)


if __name__ == "__main__":
    cli()
