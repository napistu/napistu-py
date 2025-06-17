from __future__ import annotations

import functools
import logging
import os
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_rpy2_availability():
    """Check if rpy2 is available. Cached to avoid repeated checks."""
    try:
        import rpy2  # noqa: F401 - needed for rpy2 availability check

        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"rpy2 initialization failed: {e}")
        return False


@lru_cache(maxsize=1)
def get_rpy2_core_modules():
    """Import and cache core rpy2 modules."""
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
def get_rpy2_extended_modules():
    """Import and cache extended rpy2 modules (pandas2ri, arrow, etc.)."""
    if not get_rpy2_availability():
        raise ImportError(
            "This function requires `rpy2`. "
            "Please install `napistu` with the `rpy2` extra dependencies. "
            "For example: `pip install napistu[rpy2]`"
        )

    try:
        from rpy2.robjects import pandas2ri
        import pyarrow

        # loading rpy2_arrow checks whether the R arrow package is found
        # this is the first time when a non-standard R package is loaded
        # so a bad R setup can cause issues at this stage
        try:
            import rpy2_arrow.arrow as pyra
        except Exception as e:
            rsession_info()
            raise e

        import rpy2.robjects.conversion  # noqa: F401 - needed for R conversion setup
        import rpy2.rinterface  # noqa: F401 - needed for R interface initialization
        import rpy2.robjects as ro
        from rpy2.robjects import ListVector

        return pandas2ri, pyarrow, pyra, ro, ListVector
    except Exception as e:
        logger.error(f"Failed to import extended rpy2 modules: {e}")
        raise


@lru_cache(maxsize=1)
def get_napistu_r_package():
    """Import and cache the napistu R package."""
    conversion, default_converter, importr = get_rpy2_core_modules()

    try:
        napistu_r = importr("napistu.r")
        return napistu_r
    except Exception as e:
        logger.error(f"Failed to import napistu.r R package: {e}")
        raise


def require_rpy2(func):
    """Decorator to ensure rpy2 is available before calling function."""

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


def report_r_exceptions(func):
    """Decorator to provide helpful error reporting for R-related exceptions."""

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
    """Report summaries of the R installation found by rpy2."""
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
    """Utility function to suggest installation directions for R."""
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
