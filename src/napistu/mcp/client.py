# src/napistu/mcp/client.py
"""
MCP client for testing and interacting with Napistu MCP servers.
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any

# Import MCP client components
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

async def check_server_health(transport: str = "stdio", server_url: str = "http://127.0.0.1:8080"):
    """
    Simple health check of an MCP server.
    
    Args:
        transport: Transport type ('stdio' or 'http')
        server_url: Server URL for HTTP transport
    """
    if not MCP_AVAILABLE:
        print("❌ MCP client not available. Install with: pip install mcp")
        return
    
    try:
        if transport == "stdio":
            # Start server process and connect via stdio
            # Use python -m to run the module
            server_params = StdioServerParameters(
                command="python",
                args=["-m", "napistu.mcp", "server", "start", "--profile", "full"]
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # Get health status
                    result = await session.read_resource("napistu://health")
                    if result and result.contents:
                        for content in result.contents:
                            if hasattr(content, 'text'):
                                try:
                                    health = json.loads(content.text)
                                    return health
                                except json.JSONDecodeError:
                                    continue
        else:
            # Connect via HTTP/SSE
            sse_url = f"{server_url}/sse"
            async with sse_client(sse_url) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # Get health status
                    result = await session.read_resource("napistu://health")
                    if result and result.contents:
                        for content in result.contents:
                            if hasattr(content, 'text'):
                                try:
                                    health = json.loads(content.text)
                                    return health
                                except json.JSONDecodeError:
                                    continue
        
        return None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return None

def print_health_status(health: Optional[Dict[str, Any]]):
    """Pretty print health status"""
    if not health:
        print("❌ Could not get health status")
        return
    
    status = health.get('status', 'unknown')
    print(f"\nServer Status: {status}")
    
    components = health.get('components', {})
    if components:
        print("\nComponents:")
        for name, comp_status in components.items():
            icon = "✅" if comp_status.get('status') == 'healthy' else "❌"
            print(f"  {icon} {name}: {comp_status.get('status', 'unknown')}")

async def main():
    """Main entry point for testing"""
    print("🏥 Testing Napistu MCP Server")
    print("=" * 50)
    
    # Test stdio transport
    print("\nTesting stdio transport...")
    health = await check_server_health(transport="stdio")
    print_health_status(health)
    
    # Test HTTP transport
    print("\nTesting HTTP transport...")
    health = await check_server_health(transport="http")
    print_health_status(health)

if __name__ == "__main__":
    asyncio.run(main())