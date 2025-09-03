"""Lazy loading for ingestion modules to avoid import-time side effects."""

import importlib
import logging
from functools import lru_cache
from typing import Any, Dict

# Mapping of package names to their extras (if any)
INGESTION_PACKAGE_TO_EXTRA: Dict[str, str] = {
    "omnipath": "ingestion",  # or whatever extra contains omnipath
    # Add other problematic packages here as needed
    # "some_other_package": "extra_name",
}

logger = logging.getLogger(__name__)


def _configure_package_logging(package_name: str) -> None:
    """Configure logging for a package before import to prevent pollution."""
    if package_name == "omnipath":
        # Silence omnipath before it gets imported
        logging.getLogger("omnipath").setLevel(logging.CRITICAL)
        # Ensure root logger doesn't get polluted during omnipath import
        root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)
        return root_level
    # Add other package-specific logging configurations here
    return None


def _restore_logging(package_name: str, original_level: Any) -> None:
    """Restore original logging configuration after import."""
    if package_name == "omnipath" and original_level is not None:
        logging.getLogger().setLevel(original_level)


def get_package(package_name: str) -> Any:
    """Import a package with pre-configured logging to avoid side effects."""
    try:
        # Configure logging before import
        original_level = _configure_package_logging(package_name)

        # Import the package
        package = importlib.import_module(package_name)

        # Restore logging
        _restore_logging(package_name, original_level)

        return package
    except ImportError:
        if package_name in INGESTION_PACKAGE_TO_EXTRA:
            extra = INGESTION_PACKAGE_TO_EXTRA[package_name]
            raise ImportError(
                f"Package {package_name} is required. "
                f"Install with: pip install napistu[{extra}]"
            )
        else:
            raise ImportError(f"Package {package_name} is not available")


def create_package_getter(package_name: str):
    """Create a cached package getter function."""

    @lru_cache(maxsize=1)
    def _get_package():
        return get_package(package_name)

    return _get_package


# Create getters for ingestion packages
get_omnipath = create_package_getter("omnipath")
get_omnipath_interactions = create_package_getter("omnipath.interactions")

# Add more as needed:
# get_other_package = create_package_getter("other_package")


def require_package(package_name: str):
    """Decorator to ensure a package is available."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # This will trigger the lazy import with proper logging config
            get_package(package_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators
require_omnipath = require_package("omnipath")
