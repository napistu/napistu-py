"""
MCP (Model Context Protocol) Server CLI for Napistu.
"""

import asyncio
import json

import click
import httpx

from napistu._cli import setup_logging, verbosity_option
from napistu.mcp.client import (
    check_server_health,
    list_server_resources,
    print_health_status,
    read_server_resource,
    search_all,
    search_component,
)
from napistu.mcp.config import (
    client_config_options,
    local_client_config,
    local_server_config,
    production_client_config,
    server_config_options,
    validate_client_config_flags,
    validate_server_config_flags,
)
from napistu.mcp.constants import (
    API_ENDPOINTS,
    HEALTH_CHECK_DEFS,
    HEALTH_SUMMARIES,
    MCP_COMPONENTS,
    MCP_DEFAULTS,
    MCP_PROFILES,
    SEARCH_TYPES,
)
from napistu.mcp.server import start_mcp_server

# Module-level logger and console - will be initialized when CLI is invoked
logger = None
console = None


@click.group()
def cli():
    """The Napistu MCP (Model Context Protocol) Server CLI"""
    # Set up logging only when CLI is actually invoked, not at import time
    # This prevents interfering with pytest's caplog fixture during tests
    global logger, console
    if logger is None:
        logger, console = setup_logging()


@click.group()
def server():
    """Start and manage MCP servers."""
    pass


@server.command(name="start")
@click.option(
    "--profile",
    type=click.Choice([MCP_PROFILES.EXECUTION, MCP_PROFILES.DOCS, MCP_PROFILES.FULL]),
    default=MCP_PROFILES.DOCS,
)
@server_config_options
@verbosity_option
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
@verbosity_option
def start_local():
    """Start a local MCP server optimized for function execution."""
    config = local_server_config()
    click.echo("Starting local development server (execution profile)")
    click.echo(f"  Host: {config.host}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Server Name: {config.server_name}")

    start_mcp_server(MCP_PROFILES.EXECUTION, config)


@server.command(name="full")
@verbosity_option
def start_full():
    """Start a full MCP server with all components enabled (local debugging)."""
    config = local_server_config()
    # Override server name for full profile
    config.server_name = MCP_DEFAULTS.FULL_SERVER_NAME

    click.echo("Starting full development server (all components)")
    click.echo(f"  Host: {config.host}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Server Name: {config.server_name}")

    start_mcp_server(MCP_PROFILES.FULL, config)


@cli.command()
@client_config_options
@verbosity_option
def health(production, local, host, port, https):
    """Quick health check of MCP server."""

    async def run_health_check():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            logger.info("🏥 Napistu MCP Server Health Check")
            logger.info("=" * 40)
            logger.info(f"Server URL: {config.base_url}")
            logger.info("")

            health = await check_server_health(config)
            print_health_status(health)

        except click.BadParameter as e:
            raise click.ClickException(str(e))

    asyncio.run(run_health_check())


@cli.command()
@client_config_options
@verbosity_option
def resources(production, local, host, port, https):
    """List all available resources on the MCP server."""

    async def run_list_resources():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            logger.info("📋 Napistu MCP Server Resources")
            logger.info("=" * 40)
            logger.info(f"Server URL: {config.base_url}")
            logger.info("")

            resources = await list_server_resources(config)

            if resources:
                logger.info(f"Found {len(resources)} resources:")
                for resource in resources:
                    logger.info(f"  📄 {resource.uri}")
                    if resource.name != resource.uri:
                        logger.info(f"      Name: {resource.name}")
                    if hasattr(resource, "description") and resource.description:
                        logger.info(f"      Description: {resource.description}")
                    logger.info("")
            else:
                logger.warning("❌ Could not retrieve resources")

        except click.BadParameter as e:
            raise click.ClickException(str(e))

    asyncio.run(run_list_resources())


@cli.command()
@click.argument("resource_uri")
@client_config_options
@click.option(
    "--output", type=click.File("w"), default="-", help="Output file (default: stdout)"
)
@verbosity_option
def read(resource_uri, production, local, host, port, https, output):
    """Read a specific resource from the MCP server."""

    async def run_read_resource():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            if output.name == "<stdout>":
                logger.info(f"📖 Reading Resource: {resource_uri}")
                logger.info(f"Server URL: {config.base_url}")
                logger.info("=" * 50)
            else:
                print(
                    f"📖 Reading Resource: {resource_uri}",
                    file=output,
                )
                print(
                    f"Server URL: {config.base_url}",
                    file=output,
                )
                print("=" * 50, file=output)

            content = await read_server_resource(resource_uri, config)

            if content:
                if output.name == "<stdout>":
                    logger.info(content)
                else:
                    print(content, file=output)
            else:
                if output.name == "<stdout>":
                    logger.info("❌ Could not read resource")
                else:
                    print("❌ Could not read resource", file=output)

        except click.BadParameter as e:
            raise click.ClickException(str(e))

    asyncio.run(run_read_resource())


@cli.command()
@verbosity_option
def compare():
    """Compare health between local development and production servers."""

    async def run_comparison():

        local_config = local_client_config()
        production_config = production_client_config()

        logger.info("🔍 Local vs Production Server Comparison")
        logger.info("=" * 50)

        logger.info("")
        logger.info(f"📍 Local Server: {local_config.base_url}")
        local_health = await check_server_health(local_config)
        print_health_status(local_health)

        logger.info("")
        logger.info(f"🌐 Production Server: {production_config.base_url}")
        production_health = await check_server_health(production_config)
        print_health_status(production_health)

        # Compare results
        logger.info("")
        logger.info("📊 Comparison Summary:")
        if local_health and production_health:
            local_components = local_health.get(HEALTH_SUMMARIES.COMPONENTS, {})
            production_components = production_health.get(
                HEALTH_SUMMARIES.COMPONENTS, {}
            )

            all_components = set(local_components.keys()) | set(
                production_components.keys()
            )

            for component in sorted(all_components):
                local_status = local_components.get(component, {}).get(
                    HEALTH_CHECK_DEFS.STATUS, "missing"
                )
                production_status = production_components.get(component, {}).get(
                    HEALTH_CHECK_DEFS.STATUS, "missing"
                )

                if local_status == production_status == HEALTH_CHECK_DEFS.HEALTHY:
                    icon = "✅"
                elif local_status != production_status:
                    icon = "⚠️ "
                else:
                    icon = "❌"

                logger.info(
                    f"  {icon} {component}: Local={local_status}, Production={production_status}"
                )
        else:
            logger.info("  ❌ Cannot compare - one or both servers unreachable")

    asyncio.run(run_comparison())


@cli.command()
@click.argument(
    "component",
    type=click.Choice(
        [
            MCP_COMPONENTS.DOCUMENTATION,
            MCP_COMPONENTS.TUTORIALS,
            MCP_COMPONENTS.CODEBASE,
            "all",
        ]
    ),
)
@click.argument("query")
@click.option(
    "--search-type",
    type=click.Choice([SEARCH_TYPES.SEMANTIC, SEARCH_TYPES.EXACT]),
    default=SEARCH_TYPES.SEMANTIC,
    help="Search strategy to use (default: semantic)",
)
@click.option(
    "--show-scores",
    is_flag=True,
    help="Show similarity scores for semantic search results",
)
@click.option(
    "--max-results",
    type=int,
    default=None,
    help="Maximum number of results to return (default: 10 for 'all', 5 for individual components)",
)
@client_config_options
@verbosity_option
def search(
    component,
    query,
    search_type,
    show_scores,
    max_results,
    production,
    local,
    host,
    port,
    https,
):
    """Search Napistu documentation, tutorials, codebase, or all components."""

    async def run_search():
        try:
            config = validate_client_config_flags(local, production, host, port, https)

            # Determine default n_results based on component
            n_results = (
                max_results
                if max_results is not None
                else (10 if component == "all" else 5)
            )

            if component == "all":
                logger.info(f"🔍 Searching all components for: '{query}'")
                logger.info("=" * 50)
                logger.info(f"Server URL: {config.base_url}")
                logger.info(f"Search Type: {search_type}")
                logger.info(f"Max Results: {n_results}")
                logger.info("")

                result = await search_all(query, search_type, n_results, config)
            else:
                logger.info(f"🔍 Searching {component.title()} for: '{query}'")
                logger.info("=" * 50)
                logger.info(f"Server URL: {config.base_url}")
                logger.info(f"Search Type: {search_type}")
                logger.info(f"Max Results: {n_results}")
                logger.info("")

                result = await search_component(
                    component, query, search_type, n_results, config
                )

            if not result:
                logger.info("❌ Search failed - check server connection")
                return

            # Display results
            results = result.get("results", [])
            actual_search_type = result.get("search_type", search_type)

            if not results:
                logger.info("🔍 No results found")
                if result.get("tip"):
                    logger.info(f"💡 Tip: {result['tip']}")
                return

            logger.info(f"📋 Found {len(results)} result(s):")
            logger.info("")

            # Format results based on search type
            if actual_search_type == SEARCH_TYPES.SEMANTIC and isinstance(
                results, list
            ):
                # Semantic search results with scores
                # Group by component for unified search
                if component == "all":
                    # Show component labels for unified search
                    for i, r in enumerate(results, 1):
                        comp = r.get("component", HEALTH_CHECK_DEFS.UNKNOWN)
                        source = r.get("source", "Unknown")
                        content = (
                            r.get("content", "")[:100] + "..."
                            if len(r.get("content", "")) > 100
                            else r.get("content", "")
                        )

                        if show_scores and "similarity_score" in r:
                            score = r["similarity_score"]
                            logger.info(
                                f"{i}. [{comp.upper()}] {source} (Score: {score:.3f})"
                            )
                        else:
                            logger.info(f"{i}. [{comp.upper()}] {source}")

                        if content:
                            logger.info(f"   {content}")
                        logger.info("")
                else:
                    # Component-specific search (no component label needed)
                    for i, r in enumerate(results, 1):
                        source = r.get("source", "Unknown")
                        content = (
                            r.get("content", "")[:100] + "..."
                            if len(r.get("content", "")) > 100
                            else r.get("content", "")
                        )

                        if show_scores and "similarity_score" in r:
                            score = r["similarity_score"]
                            logger.info(f"{i}. {source} (Score: {score:.3f})")
                        else:
                            logger.info(f"{i}. {source}")

                        if content:
                            logger.info(f"   {content}")
                        logger.info("")

            elif actual_search_type == SEARCH_TYPES.EXACT and isinstance(results, dict):
                # Exact search results organized by type
                total_found = 0
                for result_type, items in results.items():
                    if items:
                        logger.info(f"📁 {result_type.title()}:")
                        for item in items:
                            name = item.get("name", "Unknown")
                            snippet = (
                                item.get("snippet", "")[:100] + "..."
                                if len(item.get("snippet", "")) > 100
                                else item.get("snippet", "")
                            )
                            logger.info(f"  • {name}")
                            if snippet:
                                logger.info(f"    {snippet}")
                        logger.info("")
                        total_found += len(items)

                if total_found == 0:
                    logger.info("🔍 No results found")

            else:
                # Fallback formatting
                logger.info(f"Results: {results}")

            # Show tip if available
            if result.get("tip"):
                logger.info(f"💡 Tip: {result['tip']}")

        except click.BadParameter as e:
            raise click.ClickException(str(e))
        except Exception as e:
            raise click.ClickException(f"Search failed: {str(e)}")

    asyncio.run(run_search())


@cli.command()
@click.argument("message")
@client_config_options
@verbosity_option
def chat(message, production, local, host, port, https):
    """Send a message to the chat API."""

    async def run_chat():
        try:
            config = validate_client_config_flags(local, production, host, port, https)
            api_url = f"{config.base_url}{API_ENDPOINTS.API}/{API_ENDPOINTS.CHAT}"

            logger.info("💬 Sending message to chat API")
            logger.info("=" * 50)
            logger.info(f"Server URL: {config.base_url}")
            logger.info(f"Message: {message[:50]}{'...' if len(message) > 50 else ''}")
            logger.info("")

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json={"content": message},
                )
                response.raise_for_status()
                result = response.json()

                if "response" in result:
                    logger.info("Response:")
                    logger.info("-" * 50)
                    logger.info(result["response"])
                    if "usage" in result:
                        logger.info("")
                        logger.info("Usage:")
                        logger.info(
                            f"  Input tokens: {result['usage'].get('input_tokens', 'N/A')}"
                        )
                        logger.info(
                            f"  Output tokens: {result['usage'].get('output_tokens', 'N/A')}"
                        )
                else:
                    logger.info(json.dumps(result, indent=2))

        except click.BadParameter as e:
            raise click.ClickException(str(e))
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", str(e))
            except (ValueError, KeyError):
                error_detail = str(e)
            raise click.ClickException(f"Chat API error: {error_detail}")
        except Exception as e:
            raise click.ClickException(f"Chat request failed: {str(e)}")

    asyncio.run(run_chat())


@cli.command(name="api-stats")
@client_config_options
@verbosity_option
def api_stats(production, local, host, port, https):
    """Get usage statistics from the chat API."""

    async def run_stats():
        try:
            config = validate_client_config_flags(local, production, host, port, https)
            api_url = f"{config.base_url}{API_ENDPOINTS.API}/{API_ENDPOINTS.STATS}"

            logger.info("📊 Chat API Statistics")
            logger.info("=" * 50)
            logger.info(f"Server URL: {config.base_url}")
            logger.info("")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(api_url)
                response.raise_for_status()
                result = response.json()

                if "budget" in result:
                    budget = result["budget"]
                    logger.info("Budget:")
                    logger.info(f"  Daily limit: ${budget.get('daily_limit', 'N/A')}")
                    logger.info(f"  Spent today: ${budget.get('spent_today', 'N/A')}")
                    logger.info(f"  Remaining: ${budget.get('remaining', 'N/A')}")
                    logger.info("")

                if "rate_limits" in result:
                    rate_limits = result["rate_limits"]
                    logger.info("Rate Limits:")
                    logger.info(f"  Per hour: {rate_limits.get('per_hour', 'N/A')}")
                    logger.info(f"  Per day: {rate_limits.get('per_day', 'N/A')}")
                else:
                    logger.info(json.dumps(result, indent=2))

        except click.BadParameter as e:
            raise click.ClickException(str(e))
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", str(e))
            except (ValueError, KeyError):
                error_detail = str(e)
            raise click.ClickException(f"Stats API error: {error_detail}")
        except Exception as e:
            raise click.ClickException(f"Stats request failed: {str(e)}")

    asyncio.run(run_stats())


@cli.command(name="api-health")
@client_config_options
@verbosity_option
def api_health(production, local, host, port, https):
    """Check the health of the chat API."""

    async def run_api_health():
        try:
            config = validate_client_config_flags(local, production, host, port, https)
            api_url = f"{config.base_url}{API_ENDPOINTS.API}/{API_ENDPOINTS.HEALTH}"

            logger.info("🏥 Chat API Health Check")
            logger.info("=" * 50)
            logger.info(f"Server URL: {config.base_url}")
            logger.info("")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(api_url)
                response.raise_for_status()
                result = response.json()

                status = result.get("status", "unknown")
                status_icon = "✅" if status == "healthy" else "❌"
                logger.info(f"{status_icon} Status: {status}")

                if "chat_api" in result:
                    logger.info(f"Chat API: {result['chat_api']}")
                if "budget_ok" in result:
                    budget_status = "✅ OK" if result["budget_ok"] else "❌ Exceeded"
                    logger.info(f"Budget: {budget_status}")

                if "error" in result:
                    logger.info(f"Error: {result['error']}")

        except click.BadParameter as e:
            raise click.ClickException(str(e))
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", str(e))
            except (ValueError, KeyError):
                error_detail = str(e)
            raise click.ClickException(f"API health check error: {error_detail}")
        except Exception as e:
            raise click.ClickException(f"API health check failed: {str(e)}")

    asyncio.run(run_api_health())


@cli.command()
@client_config_options
@verbosity_option
def api_test_mcp(production, local, host, port, https):
    """Test MCP server connectivity from chat API."""

    async def run_test():
        try:
            config = validate_client_config_flags(local, production, host, port, https)
            api_url = f"{config.base_url}{API_ENDPOINTS.API}/test-mcp"

            logger.info("🔧 Testing MCP server connectivity")
            logger.info("=" * 50)
            logger.info(f"Server URL: {config.base_url}")
            logger.info("")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(api_url)
                response.raise_for_status()
                result = response.json()

                logger.info("MCP Test Results:")
                logger.info("-" * 50)
                logger.info(json.dumps(result, indent=2))

        except click.BadParameter as e:
            raise click.ClickException(str(e))
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", str(e))
            except (ValueError, KeyError):
                error_detail = str(e)
            raise click.ClickException(f"MCP test failed: {error_detail}")
        except Exception as e:
            raise click.ClickException(f"MCP test request failed: {str(e)}")

    asyncio.run(run_test())


# Add commands to the CLI
cli.add_command(server)
cli.add_command(health)
cli.add_command(resources)
cli.add_command(read)
cli.add_command(compare)
cli.add_command(search)
cli.add_command(chat)
cli.add_command(api_stats)
cli.add_command(api_health)
cli.add_command(api_test_mcp)


if __name__ == "__main__":
    cli()
