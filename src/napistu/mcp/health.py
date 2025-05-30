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


async def _check_components() -> Dict[str, Dict[str, Any]]:
    """Check the health of individual MCP components."""
    components = {}
    
    # Check documentation component
    try:
        from napistu.mcp import documentation
        # Simple check - verify cache is initialized
        if hasattr(documentation, '_docs_cache') and documentation._docs_cache:
            components["documentation"] = {"status": "healthy"}
        else:
            components["documentation"] = {"status": "not_initialized"}
    except Exception as e:
        components["documentation"] = {"status": "error", "error": str(e)}
    
    # Check codebase component
    try:
        from napistu.mcp import codebase
        # Simple check - verify cache is initialized
        if hasattr(codebase, '_codebase_cache') and codebase._codebase_cache:
            components["codebase"] = {"status": "healthy"}
        else:
            components["codebase"] = {"status": "not_initialized"}
    except Exception as e:
        components["codebase"] = {"status": "error", "error": str(e)}
    
    # Check tutorials component
    try:
        from napistu.mcp import tutorials
        # Simple check - verify cache is initialized
        if hasattr(tutorials, '_tutorial_cache') and tutorials._tutorial_cache:
            components["tutorials"] = {"status": "healthy"}
        else:
            components["tutorials"] = {"status": "not_initialized"}
    except Exception as e:
        components["tutorials"] = {"status": "error", "error": str(e)}
    
    # Check execution component
    try:
        from napistu.mcp import execution
        # Simple check - verify context is initialized
        if hasattr(execution, '_session_context') and execution._session_context:
            components["execution"] = {"status": "healthy"}
        else:
            components["execution"] = {"status": "not_initialized"}
    except Exception as e:
        components["execution"] = {"status": "error", "error": str(e)}
    
    return components


def _get_version() -> str:
    """Get the Napistu version."""
    try:
        import napistu
        return getattr(napistu, "__version__", "unknown")
    except ImportError:
        return "unknown"