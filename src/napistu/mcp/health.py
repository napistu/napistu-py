# src/napistu/mcp/health.py
"""
Health check endpoint for the MCP server when deployed to Cloud Run.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def register_health_endpoint(mcp):
    """
    Register health check endpoint with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("napistu://health")
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint for deployment monitoring.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Basic health check - verify server is responsive
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": _get_version(),
                "components": await _check_components()
            }
            
            # Check if any components failed
            failed_components = [
                name for name, status in health_status["components"].items() 
                if status["status"] != "healthy"
            ]
            
            if failed_components:
                health_status["status"] = "degraded"
                health_status["failed_components"] = failed_components
                
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    @mcp.tool()
    async def get_server_info() -> Dict[str, Any]:
        """
        Get detailed server information for monitoring.
        
        Returns:
            Dictionary with server configuration and status
        """
        import os
        import sys
        
        return {
            "server_name": os.getenv("MCP_SERVER_NAME", "unknown"),
            "profile": os.getenv("MCP_PROFILE", "unknown"),
            "python_version": sys.version,
            "platform": sys.platform,
            "environment": {
                "PORT": os.getenv("PORT"),
                "HOST": os.getenv("HOST"),
                "MCP_PROFILE": os.getenv("MCP_PROFILE"),
                "MCP_SERVER_NAME": os.getenv("MCP_SERVER_NAME")
            },
            "startup_time": datetime.utcnow().isoformat()
        }


def _check_component_health(component_name: str, module_name: str, cache_attr: str) -> Dict[str, str]:
    """
    Check the health of a single MCP component.
    
    Args:
        component_name: Name of the component (for importing)
        module_name: Full module path for importing
        cache_attr: Name of the cache/context attribute to check
        
    Returns:
        Dictionary with component status and optional error message
    """
    try:
        module = __import__(module_name, fromlist=[component_name])
        if hasattr(module, cache_attr) and getattr(module, cache_attr):
            return {"status": "healthy"}
        return {"status": "not_initialized"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}

async def _check_components() -> Dict[str, Dict[str, Any]]:
    """Check the health of individual MCP components."""
    # Define component configurations
    component_configs = {
        "documentation": ("napistu.mcp.documentation", "_docs_cache"),
        "codebase": ("napistu.mcp.codebase", "_codebase_cache"),
        "tutorials": ("napistu.mcp.tutorials", "_tutorial_cache"),
        "execution": ("napistu.mcp.execution", "_session_context")
    }
    
    # Check each component
    return {
        name: _check_component_health(name, module_path, cache_attr)
        for name, (module_path, cache_attr) in component_configs.items()
    }


def _get_version() -> str:
    """Get the Napistu version."""
    try:
        import napistu
        return getattr(napistu, "__version__", "unknown")
    except ImportError:
        return "unknown"