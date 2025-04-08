from __future__ import annotations

import pandas as pd
from napistu.rpy2 import has_rpy2
from napistu.rpy2 import report_r_exceptions
from napistu.rpy2 import rsession_info
from napistu.rpy2 import warn_if_no_rpy2

if has_rpy2:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.packages import InstalledSTPackage, InstalledPackage
    import pyarrow

    # loading rpy2_arrow checks whether the R arrow package is found
    # this is the first time when a non-standard R package is loaded
    # so a bad R setup can cause issues at this stage
    # rsession_info() adds some helpful debugging information
    try:
        import rpy2_arrow.arrow as pyra
    except Exception as e:
        rsession_info()
        raise e
    import rpy2.robjects.conversion
    import rpy2.rinterface
    import rpy2.robjects as ro


@warn_if_no_rpy2
@report_r_exceptions
def get_rcpr(
    r_paths: list[str] | None = None,
):
    """
    Get rcpr

    Gets the rcpr R package

    Args:
        r_paths (list[str]):
            Paths to add to .libPaths() in R

    Returns:
        rcpr R package
    """

    _ = get_rbase(r_paths)

    # connect the cpr R package
    rcpr = importr("rcpr")
    return rcpr


@warn_if_no_rpy2
@report_r_exceptions
def bioconductor_org_r_function(
    object_type: str, species: str, r_paths: list[str] | None = None
):
    """
    Bioconuctor Organism R Function

    Calls "bioconductor_org_function" from the R cpr package to pull a mapping object
    out of a species specific library.

    Parameters:
    object_type (str):
        Type of function to call
    species (str):
        Species name
    r_paths: list(str):
        Paths to add to .libPaths() in R. Alternatively consider setting the R_HOME env variable.

    Returns:
    pd.DataFrame or a function for non-tabular results
    """

    _ = get_rbase(r_paths)

    # connect the cpr R package
    cpr = importr("rcpr")

    results = cpr.bioconductor_org_function(object_type, species)

    return results


@report_r_exceptions
def get_rbase(
    r_paths: list[str] | None = None,
) -> InstalledSTPackage | InstalledPackage:
    """Get the base R package

    Args:
        r_paths (list[str], optional): Optional additional
            r_paths. Defaults to None.

    Returns:
        _type_: _description_
    """
    base = importr("base")
    if r_paths is not None:
        base._libPaths(r_paths)
    return base


@warn_if_no_rpy2
@report_r_exceptions
def pandas_to_r_dataframe(df: pd.DataFrame) -> rpy2.robjects.DataFrame:
    """Convert a pandas dataframe to an R dataframe

    This uses the rpy2-arrow functionality
    to increase the performance of conversion orders of magnitude.

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        rpy2.robjects.DataFrame: R dataframe
    """
    conv = _get_py2rpy_pandas_conv()
    with (ro.default_converter + conv).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)
    return r_df


@warn_if_no_rpy2
@report_r_exceptions
def r_dataframe_to_pandas(rdf: rpy2.robjects.DataFrame) -> pd.DataFrame:
    """Convert an R dataframe to a pandas dataframe

    Args:
        rdf (rpy2.robjects.DataFrame): R dataframe

    Returns:
        pd.DataFrame: Pandas dataframe
    """
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(rdf)
    return df


@warn_if_no_rpy2
@report_r_exceptions
def _get_py2rpy_pandas_conv():
    """Get the py2rpy arrow converter for pandas

    This is a high-performance converter using
    the rpy2-arrow functionality:
    https://rpy2.github.io/rpy2-arrow/version/main/html/index.html

    Returns:
        Callable: The converter function
    """
    base = get_rbase()
    # We use the converter included in rpy2-arrow as template.
    conv = rpy2.robjects.conversion.Converter(
        "Pandas to data.frame", template=pyra.converter
    )

    @conv.py2rpy.register(pd.DataFrame)
    def py2rpy_pandas(dataf):
        pa_tbl = pyarrow.Table.from_pandas(dataf)
        # pa_tbl is a pyarrow table, and this is something
        # that the converter shipping with rpy2-arrow knows
        # how to handle.
        return base.as_data_frame(pa_tbl)

    return conv
