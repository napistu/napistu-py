"""
MCP (Model Context Protocol) Server for Napistu.

This module requires optional dependencies. Install with:
pip install napistu[mcp]
"""

import asyncio
from typing import Dict, Any

__all__ = ["start_server", "register_object", "is_available"]

# Check if MCP dependencies are available
try:
    __import__("mcp")
    is_available = True
except ImportError:
    is_available = False

if is_available:
    from napistu.mcp.server import create_server
    from napistu.mcp.profiles import get_profile
    from napistu.mcp.constants import MCP_PROFILES

    def start_server(
        profile_name: str = MCP_PROFILES.EXECUTION, **kwargs
    ) -> Dict[str, Any]:
        """
        Start an MCP server with a specific profile.

        Args:
            profile_name: Name of the profile ('execution', 'docs', or 'full')
            **kwargs: Additional configuration options

        Returns:
            Server control dictionary
        """
        profile = get_profile(profile_name, **kwargs)
        server = create_server(profile)

        # Start the server
        asyncio.create_task(server.start())

        # Return control interface
        return {
            "status": "running",
            "server": server,
            "profile": profile_name,
            "stop": server.stop,
            "register_object": (
                register_object if profile.get_config()["enable_execution"] else None
            ),
        }

    # Helper function for registering objects with a running server
    def register_object(name, obj):
        """Register an object with the execution component."""
        from .execution import register_object as _register

        return _register(name, obj)

else:
    # Stubs for when MCP is not available
    def start_server(*args, **kwargs):
        raise ImportError(
            "MCP support not installed. Install with 'pip install napistu[mcp]'"
        )

    def register_object(*args, **kwargs):
        raise ImportError(
            "MCP support not installed. Install with 'pip install napistu[mcp]'"
        )
