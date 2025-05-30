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

    @mcp.tool("napistu://health/check")
    async def check_current_health() -> Dict[str, Any]:
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


def _check_component_health(
    component_name: str, module_name: str, cache_attr: str
) -> Dict[str, str]:
    """
    Check the health of a single MCP component by verifying its cache is initialized and contains data.

    Parameters
    ----------
    component_name : str
        Name of the component (for importing)
    module_name : str
        Full module path for importing
    cache_attr : str
        Name of the cache/context attribute to check

    Returns
    -------
    Dict[str, str]
        Dictionary containing component health status:
            - status : str
                One of: 'healthy', 'inactive', or 'unavailable'
            - error : str, optional
                Error message if status is 'unavailable'
    """
    try:
        module = __import__(module_name, fromlist=[component_name])
        cache = getattr(module, cache_attr, None)
        logger.info(
            f"Checking {component_name} health - Cache exists: {cache is not None}"
        )

        # Component specific checks for actual data
        if cache:
            if component_name == "documentation":
                # Check if any documentation section has content
                has_data = any(bool(section) for section in cache.values())
                logger.info(f"Documentation sections: {list(cache.keys())}")
            elif component_name == "codebase":
                # Check if any codebase section has content
                has_data = any(bool(section) for section in cache.values())
                logger.info(f"Codebase sections: {list(cache.keys())}")
            elif component_name == "tutorials":
                # Check if tutorials section has content
                has_data = bool(cache.get("tutorials", {}))
                logger.info(f"Tutorials cache: {bool(cache.get('tutorials', {}))}")
            elif component_name == "execution":
                # Check if session context has more than just napistu module
                has_data = len(cache) > 0
                logger.info(f"Execution context: {list(cache.keys())}")
            else:
                has_data = bool(cache)

            if has_data:
                logger.info(f"{component_name} is healthy")
                return {"status": "healthy"}

        logger.info(f"{component_name} is inactive")
        return {"status": "inactive"}
    except Exception as e:
        logger.error(f"{component_name} check failed: {str(e)}")
        return {"status": "unavailable", "error": str(e)}


async def _check_components() -> Dict[str, Dict[str, Any]]:
    """
    Check the health of individual MCP components by verifying their caches.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping component names to their health status:
            - {component_name} : Dict[str, str]
                Health status for each component, containing:
                    - status : str
                        One of: 'healthy', 'inactive', or 'unavailable'
                    - error : str, optional
                        Error message if status is 'unavailable'
    """
    # Define component configurations - cache vars that indicate initialization
    component_configs = {
        "documentation": ("napistu.mcp.documentation", "_docs_cache"),
        "codebase": ("napistu.mcp.codebase", "_codebase_cache"),
        "tutorials": ("napistu.mcp.tutorials", "_tutorial_cache"),
        "execution": ("napistu.mcp.execution", "_session_context"),
    }

    logger.info("Starting component health checks...")
    logger.info(f"Checking components: {list(component_configs.keys())}")

    # Check each component's cache
    results = {
        name: _check_component_health(name, module_path, cache_attr)
        for name, (module_path, cache_attr) in component_configs.items()
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
