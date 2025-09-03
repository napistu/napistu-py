from __future__ import annotations

import pandas as pd

from napistu.rpy2 import (
    get_napistu_r_package,
    get_rpy2_core_modules,
    get_rpy2_extended_modules,
    report_r_exceptions,
    require_rpy2,
)


@require_rpy2
@report_r_exceptions
def get_napistu_r(r_paths: list[str] | None = None):
    """
    Get Napistu R package.

    Args:
        r_paths (list[str], optional): Paths to add to .libPaths() in R

    Returns:
        napistu.r R package
    """
    _ = get_rbase(r_paths)
    return get_napistu_r_package()


@require_rpy2
@report_r_exceptions
def bioconductor_org_r_function(
    object_type: str, species: str, r_paths: list[str] | None = None
):
    """
    Bioconductor Organism R Function

    Calls "bioconductor_org_function" from the R cpr package to pull a mapping object
    out of a species specific library.

    Parameters:
    object_type (str): Type of function to call
    species (str): Species name
    r_paths (list[str], optional): Paths to add to .libPaths() in R.
        Alternatively consider setting the R_HOME env variable.

    Returns:
    pd.DataFrame or a function for non-tabular results
    """
    _ = get_rbase(r_paths)
    napistu_r = get_napistu_r_package()
    results = napistu_r.bioconductor_org_function(object_type, species)
    return results


@require_rpy2
@report_r_exceptions
def get_rbase(r_paths: list[str] | None = None):
    """Get the base R package.

    Args:
        r_paths (list[str], optional): Optional additional r_paths. Defaults to None.

    Returns:
        Base R package
    """
    conversion, default_converter, importr = get_rpy2_core_modules()
    base = importr("base")
    if r_paths is not None:
        base._libPaths(r_paths)
    return base


@require_rpy2
@report_r_exceptions
def pandas_to_r_dataframe(df: pd.DataFrame):
    """Convert a pandas dataframe to an R dataframe.

    This uses the rpy2-arrow functionality to increase the performance
    of conversion by orders of magnitude.

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        rpy2.robjects.DataFrame: R dataframe
    """
    pandas2ri, pyarrow, pyra, ro, ListVector = get_rpy2_extended_modules()

    conv = _get_py2rpy_pandas_conv()
    with (ro.default_converter + conv).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)
    return r_df


@require_rpy2
@report_r_exceptions
def r_dataframe_to_pandas(rdf):
    """Convert an R dataframe to a pandas dataframe.

    Args:
        rdf (rpy2.robjects.DataFrame): R dataframe

    Returns:
        pd.DataFrame: Pandas dataframe
    """
    pandas2ri, pyarrow, pyra, ro, ListVector = get_rpy2_extended_modules()

    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(rdf)
    return df


@require_rpy2
@report_r_exceptions
def _get_py2rpy_pandas_conv():
    """Get the py2rpy arrow converter for pandas.

    This is a high-performance converter using the rpy2-arrow functionality:
    https://rpy2.github.io/rpy2-arrow/version/main/html/index.html

    Returns:
        Callable: The converter function
    """
    pandas2ri, pyarrow, pyra, ro, ListVector = get_rpy2_extended_modules()

    base = get_rbase()
    # We use the converter included in rpy2-arrow as template.
    conv = ro.conversion.Converter("Pandas to data.frame", template=pyra.converter)

    @conv.py2rpy.register(pd.DataFrame)
    def py2rpy_pandas(dataf):
        pa_tbl = pyarrow.Table.from_pandas(dataf)
        # pa_tbl is a pyarrow table, and this is something
        # that the converter shipping with rpy2-arrow knows
        # how to handle.
        return base.as_data_frame(pa_tbl)

    return conv
