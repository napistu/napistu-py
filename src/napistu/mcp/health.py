# src/napistu/mcp/health.py
"""
Health check endpoint for the MCP server when deployed to Cloud Run.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, TypeVar

from fastmcp import FastMCP

from napistu.mcp.constants import (
    HEALTH_CHECK_DEFS,
    HEALTH_SUMMARIES,
    MCP_COMPONENTS,
    PROFILE_DEFS,
)

logger = logging.getLogger(__name__)

# Type variable for the FastMCP decorator return type
T = TypeVar("T")

# Global cache for component health status
_health_cache = {
    HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.INITIALIZING,
    HEALTH_SUMMARIES.COMPONENTS: {},
    HEALTH_SUMMARIES.LAST_CHECK: None,
}

# Global profile configuration (set when components are registered)
_profile_config: Dict[str, Any] = {}


def register_components(mcp: FastMCP, profile_config: Dict[str, Any]) -> None:
    """
    Register health check components with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        FastMCP server instance to register the health endpoint with.
    profile_config : Dict[str, Any]
        Profile configuration dictionary indicating which components are enabled.
        Health checks will only check enabled components.
    """
    global _profile_config
    _profile_config = profile_config

    @mcp.resource("napistu://health")
    async def health_check() -> Dict[str, Any]:
        """
        Get cached health status of the Napistu MCP server and all components.

        **USE THIS WHEN:**
        - Checking if Napistu MCP server is running and operational
        - Verifying that Napistu components (documentation, tutorials, codebase, execution) are loaded
        - Getting a quick overview of server status without triggering active checks
        - Monitoring server health for deployment or troubleshooting

        **DO NOT USE FOR:**
        - Real-time health verification (use check_health tool for active checking)
        - General Napistu installation questions (use documentation component)
        - Component-specific issues (use individual component health details)
        - Performance monitoring (this only shows component availability)

        Returns
        -------
        Dict[str, Any]
            Cached health status containing:
            - status : str
                Overall server status ("healthy", "degraded", "unhealthy", "initializing")
            - timestamp : str
                ISO timestamp of when health was last assessed
            - version : str
                Napistu package version
            - components : Dict[str, Dict]
                Status of each component (documentation, tutorials, codebase, execution)
            - failed_components : List[str] (if any)
                Names of components that are unavailable
            - last_check : str
                ISO timestamp of last active health check

        Examples
        --------
        Use this to quickly verify the MCP server is operational before attempting
        to search documentation, execute functions, or access tutorials.

        **Component Status Meanings:**
        - "healthy": Component loaded successfully with data
        - "degraded": Component partially functional but missing some data
        - "unavailable": Component failed to initialize or crashed
        - "initializing": Component still loading (check again later)

        Notes
        -----
        This returns cached status for fast response. If components haven't been
        checked yet (empty dict), it will perform a quick check to populate them.
        Use check_health() tool for real-time verification if you suspect issues.
        """
        # If components haven't been checked yet, do a quick check now
        # This ensures component statuses are always visible, even during initialization
        if not _health_cache.get(HEALTH_SUMMARIES.COMPONENTS):
            try:
                component_statuses = await _check_components()
                _health_cache[HEALTH_SUMMARIES.COMPONENTS] = component_statuses
            except Exception as e:
                logger.warning(f"Failed to check components in health resource: {e}")
                # Return what we have - components will be empty but that's better than failing

        return _health_cache

    @mcp.tool()
    async def check_health() -> Dict[str, Any]:
        """
        Actively check current health of all Napistu MCP server components.

        **USE THIS WHEN:**
        - Diagnosing why Napistu MCP server seems unresponsive or broken
        - Verifying components are working after troubleshooting
        - Getting real-time status when cached health might be stale
        - Confirming server recovery after errors or restarts

        **DO NOT USE FOR:**
        - Routine status checks (use health resource for cached status)
        - Component functionality testing (use component-specific tools)
        - Performance benchmarking (this only tests basic availability)
        - Frequent monitoring (use sparingly to avoid resource overhead)

        **WHEN TO EXPECT DIFFERENT STATUSES:**
        - "healthy": All components loaded successfully with data
        - "degraded": Some components failed but core functionality available
        - "unhealthy": Critical failures preventing normal operation
        - "initializing": Server still starting up (normal during deployment)

        Returns
        -------
        Dict[str, Any]
            Real-time health status containing:
            - status : str
                Overall server status after active checking
            - timestamp : str
                ISO timestamp of this health check
            - version : str
                Napistu package version
            - components : Dict[str, Dict]
                Current status of each component with detailed info
            - failed_components : List[str] (if any)
                Names of components that are currently unavailable
            - last_check : str
                ISO timestamp of this check (same as timestamp)

        Examples
        --------
        Use this when Napistu searches are failing or components seem unavailable:

        >>> result = await check_health()
        >>> if result["status"] != "healthy":
        ...     print(f"Issues found: {result.get('failed_components', [])}")

        **Interpreting Component Details:**
        Each component status includes:
        - Initialization success/failure
        - Data loading counts (tutorials, documentation items, etc.)
        - Error messages for failed components
        - Semantic search availability

        Notes
        -----
        This performs active checks and updates the health cache. May take
        several seconds to complete as it verifies each component's data
        and functionality.
        """
        global _health_cache
        try:
            health_status = {
                HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.HEALTHY,
                HEALTH_SUMMARIES.TIMESTAMP: _get_current_time(),
                HEALTH_SUMMARIES.VERSION: _get_version(),
                HEALTH_SUMMARIES.COMPONENTS: await _check_components(),
            }

            # Check if any components failed
            failed_components = [
                name
                for name, status in health_status[HEALTH_SUMMARIES.COMPONENTS].items()
                if status[HEALTH_SUMMARIES.STATUS] == HEALTH_CHECK_DEFS.UNAVAILABLE
            ]

            if failed_components:
                health_status[HEALTH_SUMMARIES.STATUS] = HEALTH_CHECK_DEFS.DEGRADED
                health_status[HEALTH_SUMMARIES.FAILED_COMPONENTS] = failed_components

            # Update the global cache with latest status
            health_status[HEALTH_SUMMARIES.LAST_CHECK] = _get_current_time()
            _health_cache.update(health_status)
            logger.info(
                f"Updated health cache - Status: {health_status[HEALTH_SUMMARIES.STATUS]}"
            )

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            error_status = {
                HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNHEALTHY,
                HEALTH_CHECK_DEFS.ERROR: str(e),
                HEALTH_SUMMARIES.TIMESTAMP: _get_current_time(),
                HEALTH_SUMMARIES.LAST_CHECK: _get_current_time(),
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
                HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.HEALTHY,
                HEALTH_SUMMARIES.COMPONENTS: component_status,
                HEALTH_SUMMARIES.TIMESTAMP: _get_current_time(),
                HEALTH_SUMMARIES.VERSION: _get_version(),
                HEALTH_SUMMARIES.LAST_CHECK: _get_current_time(),
            }
        )

        # Check for failed components
        failed_components = [
            name
            for name, status in component_status.items()
            if status[HEALTH_SUMMARIES.STATUS] == HEALTH_CHECK_DEFS.UNAVAILABLE
        ]

        if failed_components:
            _health_cache[HEALTH_SUMMARIES.STATUS] = HEALTH_CHECK_DEFS.DEGRADED
            _health_cache[HEALTH_SUMMARIES.FAILED_COMPONENTS] = failed_components

        logger.info(
            f"Health check initialization complete: {_health_cache[HEALTH_SUMMARIES.STATUS]}"
        )
        return True

    except Exception as e:
        logger.error(f"Health check initialization failed: {e}")
        _health_cache[HEALTH_SUMMARIES.STATUS] = HEALTH_CHECK_DEFS.UNHEALTHY
        _health_cache[HEALTH_CHECK_DEFS.ERROR] = str(e)
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
                # get_health_status() handles initialization gracefully, so we can always call it
                health_status = state.get_health_status()
                logger.info(f"{component_name} health: {health_status}")
                return health_status
            except RuntimeError as e:
                error_msg = str(e)
                # Handle components that are still initializing
                if "still initializing" in error_msg.lower():
                    logger.info(f"{component_name} still initializing: {error_msg}")
                    return {
                        HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.INITIALIZING,
                        HEALTH_SUMMARIES.MESSAGE: error_msg,
                    }
                # Handle failed initialization
                elif "failed to initialize" in error_msg.lower():
                    logger.warning(
                        f"{component_name} initialization failed: {error_msg}"
                    )
                    return {
                        HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNAVAILABLE,
                        HEALTH_CHECK_DEFS.ERROR: error_msg,
                    }
                else:
                    # Other RuntimeErrors - try to peek at state anyway
                    try:
                        component = module.get_component()
                        state = component.get_state()
                        # If we can get the state, use its health status
                        return state.get_health_status()
                    except (RuntimeError, AttributeError, TypeError):
                        # If we can't peek at state, return original error
                        logger.warning(
                            f"{component_name} error accessing state: {error_msg}"
                        )
                        return {
                            HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNAVAILABLE,
                            HEALTH_CHECK_DEFS.ERROR: error_msg,
                        }
        else:
            # Component doesn't follow the new pattern
            logger.warning(f"{component_name} doesn't use component class pattern")
            return {
                HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNKNOWN,
                HEALTH_SUMMARIES.MESSAGE: "Component using legacy pattern",
            }

    except ImportError as e:
        logger.error(f"Could not import {component_name}: {str(e)}")
        return {
            HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNAVAILABLE,
            HEALTH_CHECK_DEFS.ERROR: f"Import failed: {str(e)}",
        }
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        error_msg = str(e)
        logger.error(f"{component_name} health check failed: {error_msg}")
        # Try to provide a more helpful error message
        if "Could not get" in error_msg or "health status" in error_msg.lower():
            # This might be a component that's still initializing
            # Try to peek at the component state if possible
            try:
                module = __import__(module_path, fromlist=[component_name])
                if hasattr(module, "get_component"):
                    component = module.get_component()
                    state = component.get_state()
                    # If we can access state, use its health status
                    return state.get_health_status()
            except (RuntimeError, AttributeError, TypeError, ImportError):
                pass  # Fall through to return error

        return {
            HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNAVAILABLE,
            HEALTH_CHECK_DEFS.ERROR: error_msg,
        }


async def _check_components() -> Dict[str, Dict[str, Any]]:
    """
    Check the health of individual MCP components using their component classes.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping component names to their health status
    """
    # Define component configurations with their enable flags
    component_configs = {
        MCP_COMPONENTS.DOCUMENTATION: (
            "napistu.mcp.documentation",
            PROFILE_DEFS.ENABLE_DOCUMENTATION,
        ),
        MCP_COMPONENTS.CODEBASE: (
            "napistu.mcp.codebase",
            PROFILE_DEFS.ENABLE_CODEBASE,
        ),
        MCP_COMPONENTS.TUTORIALS: (
            "napistu.mcp.tutorials",
            PROFILE_DEFS.ENABLE_TUTORIALS,
        ),
        MCP_COMPONENTS.EXECUTION: (
            "napistu.mcp.execution",
            PROFILE_DEFS.ENABLE_EXECUTION,
        ),
    }

    logger.info("Starting component health checks...")

    results = {}
    for name, (module_path, enable_key) in component_configs.items():
        # Check if component is enabled in profile
        is_enabled = _profile_config.get(enable_key, False)
        if is_enabled:
            logger.info(f"Checking enabled component: {name}")
            results[name] = _check_component_health(name, module_path)
        else:
            logger.info(f"Skipping disabled component: {name}")
            results[name] = {
                HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.INACTIVE,
                HEALTH_SUMMARIES.MESSAGE: "Component disabled in server profile",
            }

    # Semantic search is always checked (it's shared across components)
    results["semantic_search"] = _check_semantic_search_health()

    logger.info(f"Health check results: {results}")
    return results


def _check_semantic_search_health() -> Dict[str, Any]:
    """Check health of shared semantic search instance"""
    try:
        # Try components in order until we find one with semantic search
        component_modules = [
            "napistu.mcp.documentation",
            "napistu.mcp.tutorials",
            "napistu.mcp.codebase",
        ]

        for module_path in component_modules:
            try:
                module = __import__(module_path, fromlist=["get_component"])
                component = module.get_component()
                if (
                    hasattr(component.state, "semantic_search")
                    and component.state.semantic_search
                ):
                    shared_instance = component.state.semantic_search
                    break
            except Exception:
                continue
        else:
            return {
                HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNAVAILABLE,
                HEALTH_SUMMARIES.MESSAGE: "No semantic search instance found",
            }

        collections = shared_instance.collections
        return {
            HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.HEALTHY,
            HEALTH_SUMMARIES.COLLECTIONS: list(collections.keys()),
            HEALTH_SUMMARIES.TOTAL_COLLECTIONS: len(collections),
        }

    except Exception as e:
        return {
            HEALTH_SUMMARIES.STATUS: HEALTH_CHECK_DEFS.UNAVAILABLE,
            HEALTH_CHECK_DEFS.ERROR: str(e),
        }


def _get_current_time() -> str:
    """
    Get current UTC time as ISO format string.

    Returns
    -------
    str
        Current UTC time in ISO 8601 format (e.g., "2024-01-01T12:00:00.000000+00:00")

    Notes
    -----
    Uses datetime.now(timezone.utc) instead of deprecated datetime.utcnow().
    """
    return datetime.now(timezone.utc).isoformat()


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
