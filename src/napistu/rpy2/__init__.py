from __future__ import annotations

import functools
import logging
import os
import sys

logger = logging.getLogger(__name__)

try:
    import rpy2  # noqa

    has_rpy2 = True

    from rpy2.robjects import conversion, default_converter  # noqa
    from rpy2.robjects.packages import importr  # noqa

except ImportError:
    has_rpy2 = False
    logger.warning(
        "rpy2 is not installed. "
        "Some functions will not work. "
        "Consider installing `cpr[rpy2]`."
    )
except Exception as e:
    has_rpy2 = False
    print(e)
    logger.warning("rpy2 initialization failed with an unrecognized exception.")


def warn_if_no_rpy2(func):
    @functools.wraps(func)
    def warn_if_no_rpy2_wrapper(*args, **kwargs):
        if not has_rpy2:
            raise ImportError(
                "This function requires `rpy2`. \n"
                "Please install `cpr` with the `rpy2` extra dependencies. \n"
                "For example: `pip install cpr[rpy2]`\n"
            )
        return func(*args, **kwargs)

    return warn_if_no_rpy2_wrapper


def rsession_info() -> None:
    # report summaries of the R installation found by rpy2
    # default converters bundled with rpy2 are used
    # for this step rather than those bundled with rpy2_arrow
    # because rpy2_arrow requires the arrow R package so
    # it can be difficult to import this package without
    # a valid R setup.

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
        # suggest a fix
        logger.warning(_r_homer_warning())

    return None


def _r_homer_warning() -> None:
    # utility function to suggest installation directions for R
    # as part of rsession

    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    if is_conda:
        r_lib_path = os.path.join(sys.prefix, "lib", "R")
        if os.path.isdir(r_lib_path):
            logging.warning(
                "You seem to be working in a conda environment with R installed.\n"
                "If this version was not located by rpy2 then then try to set R_HOME using:\n"
                f"os.environ['R_HOME'] = {r_lib_path}"
            )
        else:
            logging.warning(
                "You seem to be working in a conda environment but R is NOT installed.\n"
                "If this is the case then install R, the CPR R package and the R arrow package into your\n"
                "conda environment and then set the R_HOME environmental variable using:\n"
                "os.environ['R_HOME'] = <<PATH_TO_R_lib/R>>"
            )
    else:
        logging.warning(
            "If you don't have R installed or if your desired R library does not match the\n"
            "one above, then set your R_HOME environmental variable using:\n"
            "os.environ['R_HOME'] = <<PATH_TO_lib/R>>"
        )

    return None


def report_r_exceptions(function):
    @functools.wraps(function)
    def report_r_exceptions_wrapper(*args, **kwargs):
        if not has_rpy2:
            raise ImportError(
                "This function requires `rpy2`. \n"
                "Please install `cpr` with the `rpy2` extra dependencies. \n"
                "For example: `pip install cpr[rpy2]`\n"
            )
        try:
            return function(*args, **kwargs)
        except Exception as e:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__

            logger.warning(err)
            # report session info
            rsession_info()

            # re-raise the exception
            raise e

    return report_r_exceptions_wrapper
