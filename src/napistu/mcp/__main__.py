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
from napistu.mcp.config import (
    validate_server_config_flags,
    validate_client_config_flags,
    server_config_options,
    client_config_options,
    local_server_config,
    local_client_config,
    production_client_config,
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
    "--profile", type=click.Choice(["execution", "docs", "full"]), default="docs"
)
@server_config_options
@click_logging.simple_verbosity_option(logger)
def start_server(profile, production, local, host, port, server_name):
    """Start an MCP server with the specified profile."""
    try:
        config = validate_server_config_flags(
            local, production, host, port, server_name
        )

        click.echo("Starting server with configuration:")
        click.echo(f"  Profile: {profile}")
        click.echo(f"  Host: {config.host}")
        click.echo(f"  Port: {config.port}")
        click.echo(f"  Server Name: {config.server_name}")

        start_mcp_server(profile, config)

    except click.BadParameter as e:
        raise click.ClickException(str(e))


@server.command(name="local")
@click_logging.simple_verbosity_option(logger)
def start_local():
    """Start a local MCP server optimized for function execution."""
    config = local_server_config()
    click.echo("Starting local development server (execution profile)")
    click.echo(f"  Host: {config.host}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Server Name: {config.server_name}")

    start_mcp_server("execution", config)


@server.command(name="full")
@click_logging.simple_verbosity_option(logger)
def start_full():
    """Start a full MCP server with all components enabled (local debugging)."""
    config = local_server_config()
    # Override server name for full profile
    config.server_name = "napistu-full"

    click.echo("Starting full development server (all components)")
    click.echo(f"  Host: {config.host}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Server Name: {config.server_name}")

    start_mcp_server("full", config)


@cli.command()
@client_config_options
@click_logging.simple_verbosity_option(logger)
def health(production, local, host, port, https):
    """Quick health check of MCP server."""

    async def run_health_check():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            print("🏥 Napistu MCP Server Health Check")
            print("=" * 40)
            print(f"Server URL: {config.base_url}")
            print()

            health = await check_server_health(config)
            print_health_status(health)

        except click.BadParameter as e:
            raise click.ClickException(str(e))

    asyncio.run(run_health_check())


@cli.command()
@client_config_options
@click_logging.simple_verbosity_option(logger)
def resources(production, local, host, port, https):
    """List all available resources on the MCP server."""

    async def run_list_resources():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            print("📋 Napistu MCP Server Resources")
            print("=" * 40)
            print(f"Server URL: {config.base_url}")
            print()

            resources = await list_server_resources(config)

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

        except click.BadParameter as e:
            raise click.ClickException(str(e))

    asyncio.run(run_list_resources())


@cli.command()
@click.argument("resource_uri")
@client_config_options
@click.option(
    "--output", type=click.File("w"), default="-", help="Output file (default: stdout)"
)
@click_logging.simple_verbosity_option(logger)
def read(resource_uri, production, local, host, port, https, output):
    """Read a specific resource from the MCP server."""

    async def run_read_resource():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            print(
                f"📖 Reading Resource: {resource_uri}",
                file=output if output.name != "<stdout>" else None,
            )
            print(
                f"Server URL: {config.base_url}",
                file=output if output.name != "<stdout>" else None,
            )
            print("=" * 50, file=output if output.name != "<stdout>" else None)

            content = await read_server_resource(resource_uri, config)

            if content:
                print(content, file=output)
            else:
                print(
                    "❌ Could not read resource",
                    file=output if output.name != "<stdout>" else None,
                )

        except click.BadParameter as e:
            raise click.ClickException(str(e))

    asyncio.run(run_read_resource())


@cli.command()
@click_logging.simple_verbosity_option(logger)
def compare():
    """Compare health between local development and production servers."""

    async def run_comparison():

        local_config = local_client_config()
        production_config = production_client_config()

        print("🔍 Local vs Production Server Comparison")
        print("=" * 50)

        print(f"\n📍 Local Server: {local_config.base_url}")
        local_health = await check_server_health(local_config)
        print_health_status(local_health)

        print(f"\n🌐 Production Server: {production_config.base_url}")
        production_health = await check_server_health(production_config)
        print_health_status(production_health)

        # Compare results
        print("\n📊 Comparison Summary:")
        if local_health and production_health:
            local_components = local_health.get("components", {})
            production_components = production_health.get("components", {})

            all_components = set(local_components.keys()) | set(
                production_components.keys()
            )

            for component in sorted(all_components):
                local_status = local_components.get(component, {}).get(
                    "status", "missing"
                )
                production_status = production_components.get(component, {}).get(
                    "status", "missing"
                )

                if local_status == production_status == "healthy":
                    icon = "✅"
                elif local_status != production_status:
                    icon = "⚠️ "
                else:
                    icon = "❌"

                print(
                    f"  {icon} {component}: Local={local_status}, Production={production_status}"
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
