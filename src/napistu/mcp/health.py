# src/napistu/mcp/health.py
"""
Health check endpoint for the MCP server when deployed to Cloud Run.
"""

import logging
from typing import Dict, Any, TypeVar
from datetime import datetime
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Type variable for the FastMCP decorator return type
T = TypeVar("T")

# Global cache for component health status
_health_cache = {"status": "initializing", "components": {}, "last_check": None}


def register_components(mcp: FastMCP) -> None:
    """
    Register health check components with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        FastMCP server instance to register the health endpoint with.
    """

    @mcp.resource("napistu://health")
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint for deployment monitoring.
        Returns current cached health status.
        """
        return _health_cache

    @mcp.tool()
    async def check_health() -> Dict[str, Any]:
        """
        Tool to actively check current component health.
        This performs real-time checks and updates the cached status.
        """
        global _health_cache
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": _get_version(),
                "components": await _check_components(),
            }

            # Check if any components failed
            failed_components = [
                name
                for name, status in health_status["components"].items()
                if status["status"] == "unavailable"
            ]

            if failed_components:
                health_status["status"] = "degraded"
                health_status["failed_components"] = failed_components

            # Update the global cache with latest status
            health_status["last_check"] = datetime.utcnow().isoformat()
            _health_cache.update(health_status)
            logger.info(f"Updated health cache - Status: {health_status['status']}")

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            error_status = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "last_check": datetime.utcnow().isoformat(),
            }
            # Update cache even on error
            _health_cache.update(error_status)
            return error_status


async def initialize_components() -> bool:
    """
    Initialize health check components.
    Performs initial health check and caches the result.

    Returns
    -------
    bool
        True if initialization is successful
    """
    global _health_cache

    logger.info("Initializing health check components...")

    try:
        # Check initial component health
        component_status = await _check_components()

        # Update cache
        _health_cache.update(
            {
                "status": "healthy",
                "components": component_status,
                "timestamp": datetime.utcnow().isoformat(),
                "version": _get_version(),
                "last_check": datetime.utcnow().isoformat(),
            }
        )

        # Check for failed components
        failed_components = [
            name
            for name, status in component_status.items()
            if status["status"] == "unavailable"
        ]

        if failed_components:
            _health_cache["status"] = "degraded"
            _health_cache["failed_components"] = failed_components

        logger.info(f"Health check initialization complete: {_health_cache['status']}")
        return True

    except Exception as e:
        logger.error(f"Health check initialization failed: {e}")
        _health_cache["status"] = "unhealthy"
        _health_cache["error"] = str(e)
        return False


def _check_component_health(component_name: str, module_path: str) -> Dict[str, Any]:
    """
    Check the health of a single MCP component using the component class pattern.

    Parameters
    ----------
    component_name : str
        Name of the component (for logging)
    module_path : str
        Full module path for importing the component

    Returns
    -------
    Dict[str, Any]
        Dictionary containing component health status from the component's state
    """
    try:
        # Import the component module
        module = __import__(module_path, fromlist=[component_name])

        # Use the new component class pattern
        if hasattr(module, "get_component"):
            try:
                component = module.get_component()
                state = component.get_state()
                health_status = state.get_health_status()
                logger.info(f"{component_name} health: {health_status}")
                return health_status
            except RuntimeError as e:
                # Handle execution component that might not be created yet
                if "not created" in str(e):
                    logger.warning(f"{component_name} not initialized yet")
                    return {
                        "status": "initializing",
                        "message": "Component not created",
                    }
                else:
                    raise
        else:
            # Component doesn't follow the new pattern
            logger.warning(f"{component_name} doesn't use component class pattern")
            return {"status": "unknown", "message": "Component using legacy pattern"}

    except ImportError as e:
        logger.error(f"Could not import {component_name}: {str(e)}")
        return {"status": "unavailable", "error": f"Import failed: {str(e)}"}
    except Exception as e:
        logger.error(f"{component_name} health check failed: {str(e)}")
        return {"status": "unavailable", "error": str(e)}


async def _check_components() -> Dict[str, Dict[str, Any]]:
    """
    Check the health of individual MCP components using their component classes.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping component names to their health status
    """
    # Define component configurations
    component_configs = {
        "documentation": "napistu.mcp.documentation",
        "codebase": "napistu.mcp.codebase",
        "tutorials": "napistu.mcp.tutorials",
        "execution": "napistu.mcp.execution",
    }

    logger.info("Starting component health checks...")
    logger.info(f"Checking components: {list(component_configs.keys())}")

    # Check each component using their state objects
    results = {
        name: _check_component_health(name, module_path)
        for name, module_path in component_configs.items()
    }

    logger.info(f"Health check results: {results}")
    return results


def _get_version() -> str:
    """
    Get the Napistu version.

    Returns
    -------
    str
        Version string of the Napistu package, or 'unknown' if not available.
    """
    try:
        import napistu

        return getattr(napistu, "__version__", "unknown")
    except ImportError:
        return "unknown"
