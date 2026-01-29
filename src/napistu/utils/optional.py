"""
Utilities for handling optional dependencies in Napistu.

This module provides lazy loading functionality for optional dependencies,
allowing the package to be imported even when optional dependencies are not installed.
It includes logging configuration to prevent import-time side effects from noisy packages.

Function Factories
------------------
create_package_importer:
    Create a cached package importer function.
import_package:
    Import a package with pre-configured logging to avoid side effects.
require_package:
    Decorator to ensure a package is available before calling a function.

Convenience Functions
---------------------
import_anndata:
    Import and return anndata, raising an informative error if missing.
import_gseapy:
    Import and return gseapy, raising an informative error if missing.
import_mudata:
    Import and return mudata, raising an informative error if missing.
import_omnipath:
    Import and return omnipath, raising an informative error if missing.
import_omnipath_interactions:
    Import and return omnipath.interactions, raising an informative error if missing.
import_statsmodels:
    Import and return statsmodels, raising an informative error if missing.
import_statsmodels_multitest:
    Import and return statsmodels.stats.multitest.multipletests, raising an informative error if missing.

Decorators
----------
require_anndata:
    Decorator ensuring anndata is available before calling a function.
require_gseapy:
    Decorator ensuring gseapy is available before calling a function.
require_mudata:
    Decorator ensuring mudata is available before calling a function.
require_omnipath:
    Decorator ensuring omnipath is available before calling a function.
require_statsmodels:
    Decorator ensuring statsmodels is available before calling a function.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

from napistu.utils.constants import (
    CRITICAL_LOGGING_ONLY_PACKAGES,
    IMPORTABLE_PACKAGES,
    PACKAGE_TO_EXTRA,
)

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def create_package_importer(package_name: str):
    """
    Create a cached package importer function.

    This creates a function that lazily imports and caches a package.
    The cache ensures the package is only imported once per Python session.

    Parameters
    ----------
    package_name : str
        Name of the package to create an importer for.

    Returns
    -------
    Callable
        A cached function that returns the imported package.
    """

    @lru_cache(maxsize=1)
    def _import_package():
        return import_package(package_name)

    return _import_package


def import_package(package_name: str) -> Any:
    """
    Import a package with pre-configured logging to avoid side effects.

    This function handles optional dependencies by:
    1. Configuring logging to prevent import-time noise
    2. Importing the package
    3. Restoring original logging configuration
    4. Providing helpful error messages if the package is missing

    Parameters
    ----------
    package_name : str
        Name of the package to import (e.g., "omnipath", "gseapy").

    Returns
    -------
    Any
        The imported package module.

    Raises
    ------
    ImportError
        If the package cannot be imported, with a helpful message about
        which extra to install if the package is in PACKAGE_TO_EXTRA.
    """
    try:
        # Configure logging before import
        original_level = _configure_package_logging(package_name)

        # Import the package
        package = importlib.import_module(package_name)

        # Restore logging
        _restore_logging(package_name, original_level)

        return package

    except ImportError:
        if package_name in PACKAGE_TO_EXTRA:
            extra = PACKAGE_TO_EXTRA[package_name]
            raise ImportError(
                f"Package {package_name} is required. "
                f"Install with: pip install napistu[{extra}]"
            )
        else:
            raise ImportError(f"Package {package_name} is not available")


def require_package(package_name: str):
    """
    Decorator to ensure a package is available before calling a function.

    Use this decorator for functions that require an optional dependency.
    The package will be imported (with proper logging configuration) when
    the function is called, not at import time.

    Parameters
    ----------
    package_name : str
        Name of the package that must be available.

    Returns
    -------
    Callable
        A decorator function.

    Examples
    --------
    >>> @require_package("omnipath")
    >>> def process_omnipath_data():
    ...     # Uses omnipath
    ...     pass
    """

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            # This will trigger the lazy import with proper logging config
            import_package(package_name)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# Create import functions for packages using create_package_importer
import_omnipath = create_package_importer(IMPORTABLE_PACKAGES.OMNIPATH)
import_omnipath_interactions = create_package_importer(
    f"{IMPORTABLE_PACKAGES.OMNIPATH}.interactions"
)
import_anndata = create_package_importer(IMPORTABLE_PACKAGES.ANNDATA)
import_mudata = create_package_importer(IMPORTABLE_PACKAGES.MUDATA)
import_gseapy = create_package_importer(IMPORTABLE_PACKAGES.GSEAPY)
import_statsmodels = create_package_importer(IMPORTABLE_PACKAGES.STATSMODELS)
import_statsmodels_multitest = create_package_importer(
    f"{IMPORTABLE_PACKAGES.STATSMODELS}.stats.multitest"
)

# Convenience decorators
require_anndata = require_package(IMPORTABLE_PACKAGES.ANNDATA)
require_gseapy = require_package(IMPORTABLE_PACKAGES.GSEAPY)
require_mudata = require_package(IMPORTABLE_PACKAGES.MUDATA)
require_omnipath = require_package(IMPORTABLE_PACKAGES.OMNIPATH)
require_statsmodels = require_package(IMPORTABLE_PACKAGES.STATSMODELS)


def _configure_package_logging(package_name: str) -> Any:
    """Configure logging for a package before import to prevent pollution.

    Parameters
    ----------
    package_name : str
        Name of the package to configure logging for.

    Returns
    -------
    Any
        Original logging level that was configured, or None if no configuration was needed.
    """
    if package_name in CRITICAL_LOGGING_ONLY_PACKAGES:
        # Silence the package before it gets imported
        logging.getLogger(package_name).setLevel(logging.CRITICAL)
        # Ensure root logger doesn't get polluted during import
        root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)
        return root_level

    # Add other package-specific logging configurations here
    return None


def _restore_logging(package_name: str, original_level: Any) -> None:
    """
    Restore original logging configuration after import.

    Parameters
    ----------
    package_name : str
        Name of the package that was imported.
    original_level : Any
        Original logging level to restore, or None if no restoration is needed.
    """
    if package_name in CRITICAL_LOGGING_ONLY_PACKAGES and original_level is not None:
        logging.getLogger().setLevel(original_level)
