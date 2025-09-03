"""
napistu.rpy2
============

This subpackage provides utilities for interacting with R and the rpy2 bridge, including:
- Checking rpy2 availability
- Importing and caching core and extended rpy2 modules
- Handling R session information and error reporting
- Decorators for requiring rpy2 and reporting R-related exceptions

All rpy2-related imports are performed lazily and cached, so that the package can be imported even if rpy2 is not installed.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pyarrow
    import rpy2.robjects
    import rpy2.robjects.conversion
    import rpy2.robjects.ListVector
    import rpy2.robjects.packages
    import rpy2.robjects.pandas2ri
    import rpy2_arrow.arrow

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_rpy2_availability() -> bool:
    """
    Check if rpy2 is available in the current environment.

    Returns
    -------
    bool
        True if rpy2 is importable, False otherwise.
    """
    try:
        import rpy2  # noqa: F401 - needed for rpy2 availability check

        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"rpy2 initialization failed: {e}")
        return False


@lru_cache(maxsize=1)
def get_rpy2_core_modules() -> tuple[
    "rpy2.robjects.conversion.Converter",
    "rpy2.robjects.conversion.Converter",
    "rpy2.robjects.packages.importr",
]:
    """
    Import and cache core rpy2 modules (conversion, default_converter, importr).

    Returns
    -------
    tuple
        (conversion, default_converter, importr) from rpy2.robjects

    Raises
    ------
    ImportError
        If rpy2 is not available or import fails.
    """
    if not get_rpy2_availability():
        raise ImportError(
            "This function requires `rpy2`. "
            "Please install `napistu` with the `rpy2` extra dependencies. "
            "For example: `pip install napistu[rpy2]`"
        )
    try:
        from rpy2.robjects import conversion, default_converter
        from rpy2.robjects.packages import importr

        return conversion, default_converter, importr
    except Exception as e:
        logger.error(f"Failed to import core rpy2 modules: {e}")
        raise


@lru_cache(maxsize=1)
def get_rpy2_extended_modules() -> tuple[
    "rpy2.robjects.pandas2ri",
    "pyarrow",
    "rpy2_arrow.arrow",
    "rpy2.robjects",
    "rpy2.robjects.ListVector",
]:
    """
    Import and cache extended rpy2 modules (pandas2ri, pyarrow, rpy2_arrow, etc.).

    Returns
    -------
    tuple
        (pandas2ri, pyarrow, rpy2_arrow.arrow, ro, ListVector)

    Raises
    ------
    ImportError
        If rpy2 or dependencies are not available or import fails.
    """
    if not get_rpy2_availability():
        raise ImportError(
            "This function requires `rpy2`. "
            "Please install `napistu` with the `rpy2` extra dependencies. "
            "For example: `pip install napistu[rpy2]`"
        )
    try:
        import pyarrow
        from rpy2.robjects import pandas2ri

        # loading rpy2_arrow checks whether the R arrow package is found
        # this is the first time when a non-standard R package is loaded
        # so a bad R setup can cause issues at this stage
        try:
            import rpy2_arrow.arrow as pyra
        except Exception as e:
            rsession_info()
            raise e
        import rpy2.rinterface  # noqa: F401 - needed for R interface initialization
        import rpy2.robjects as ro
        import rpy2.robjects.conversion  # noqa: F401 - needed for R conversion setup
        from rpy2.robjects import ListVector

        return pandas2ri, pyarrow, pyra, ro, ListVector
    except Exception as e:
        logger.error(f"Failed to import extended rpy2 modules: {e}")
        raise


@lru_cache(maxsize=1)
def get_napistu_r_package() -> Any:
    """
    Import and cache the napistu R package using rpy2.

    Returns
    -------
    Any
        The imported napistu.r R package object.

    Raises
    ------
    ImportError
        If rpy2 or the napistu R package is not available.
    """
    conversion, default_converter, importr = get_rpy2_core_modules()
    try:
        napistu_r = importr("napistu.r")
        return napistu_r
    except Exception as e:
        logger.error(f"Failed to import napistu.r R package: {e}")
        raise


def require_rpy2(func: Callable) -> Callable:
    """
    Decorator to ensure rpy2 is available before calling the decorated function.

    Raises
    ------
    ImportError
        If rpy2 is not available.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not get_rpy2_availability():
            raise ImportError(
                f"Function '{func.__name__}' requires `rpy2`. "
                "Please install `napistu` with the `rpy2` extra dependencies. "
                "For example: `pip install napistu[rpy2]`"
            )
        return func(*args, **kwargs)

    return wrapper


def report_r_exceptions(func: Callable) -> Callable:
    """
    Decorator to provide helpful error reporting for R-related exceptions.

    If an exception occurs, logs the error and prints R session info.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not get_rpy2_availability():
            raise ImportError(
                f"Function '{func.__name__}' requires `rpy2`. "
                "Please install `napistu` with the `rpy2` extra dependencies. "
                "For example: `pip install napistu[rpy2]`"
            )
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            rsession_info()
            raise e

    return wrapper


def rsession_info() -> None:
    """
    Report summaries of the R installation found by rpy2.

    This function logs the R version, library paths, and session info using rpy2.
    If R is not available or an error occurs, logs a warning.
    """
    try:
        conversion, default_converter, importr = get_rpy2_core_modules()
        with conversion.localconverter(default_converter):
            base = importr("base")
            utils = importr("utils")
            lib_paths = base._libPaths()
            session_info = utils.sessionInfo()
            logger.warning(
                "An exception occurred when running some rpy2-related functionality\n"
                "Here is a summary of your R session\n"
                f"Using R version in {base.R_home()[0]}\n"
                ".libPaths ="
            )
            logger.warning("\n".join(lib_paths))
            logger.warning(f"sessionInfo = {session_info}")
            logger.warning(_r_homer_warning())
    except Exception as e:
        logger.warning(f"Could not generate R session info: {e}")


def _r_homer_warning() -> str:
    """
    Utility function to suggest installation directions for R based on environment.

    Returns
    -------
    str
        Installation instructions for R in the current environment.
    """
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    if is_conda:
        r_lib_path = os.path.join(sys.prefix, "lib", "R")
        if os.path.isdir(r_lib_path):
            return (
                "You seem to be working in a conda environment with R installed.\n"
                "If this version was not located by rpy2 then try to set R_HOME using:\n"
                f"os.environ['R_HOME'] = '{r_lib_path}'"
            )
        else:
            return (
                "You seem to be working in a conda environment but R is NOT installed.\n"
                "If this is the case then install R, the CPR R package and the R arrow package into your\n"
                "conda environment and then set the R_HOME environmental variable using:\n"
                "os.environ['R_HOME'] = '<<PATH_TO_R_lib/R>>'"
            )
    else:
        return (
            "If you don't have R installed or if your desired R library does not match the\n"
            "one above, then set your R_HOME environmental variable using:\n"
            "os.environ['R_HOME'] = '<<PATH_TO_lib/R>>'"
        )
