"""Module for comparing observed values to null distributions."""

import logging
from typing import Any, Union

import numpy as np
import pandas as pd

from napistu.statistics.constants import QUANTILE_METHODS, VALID_QUANTILE_METHODS

logger = logging.getLogger(__name__)


def calculate_quantiles(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    *,
    method: str = QUANTILE_METHODS.DENSE,
    comparison_dtype: Any = np.float32,
) -> pd.DataFrame:
    """
    Calculate quantiles of observed scores relative to null distributions using
    the standard midrank method for tie handling.

    This implements the same approach as R's quantile function (Type 7), which
    handles ties by averaging the ranks of tied values. For an observed value
    with tied null values, the quantile is calculated as:
    (count_less_than + count_equal_to/2) / total_count

    This approach ensures proper statistical behavior: if an observed value of 0.5
    is compared to null values [0.3, 0.5, 0.7], the result is (1 + 1/2)/3 = 0.5,
    meaning the observed value falls at the 50th percentile.

    Parameters
    ----------
    observed_df : pd.DataFrame
        DataFrame with features as index and attributes as columns containing
        observed scores.
    null_df : pd.DataFrame
        DataFrame with null scores, features as index (multiple rows per feature)
        and attributes as columns.
    method : str
        ``per_feature`` (default): one null block per feature, linear memory in
        ``n_null_samples * n_attributes`` (no 3D tensor). Best for large graphs
        and many nulls when memory is the constraint.

        ``dense``: materialize a padded 3D array
        ``(n_features, n_null_samples, n_attributes)`` and use vectorized
        comparisons. Faster when the full array fits in RAM, but can increase
        peak memory substantially.

    comparison_dtype
        numpy float dtype for `<` and `==` (default float32 for parity with
        :func:`napistu.utils.pd_utils.downcast_float_dataframe` in the
        propagation pipeline). The dense path stores the
        3D block in this dtype when it is float32, float16, or float64. Use
        float64 to mirror legacy all-float64 midrank numerics.

    Returns
    -------
    pd.DataFrame
        DataFrame with same structure as observed_df containing quantiles.
        Each value represents the proportion of null values relative to observed value
        using the midrank method for handling ties. Returns NaN when the observed
        value and all null values are identical (no meaningful quantile can be computed).

    Notes
    -----
    The midrank method is the standard statistical approach used in R and other
    major statistical software packages. When all values (observed + nulls) for
    a feature-attribute combination are identical, NaN is returned since no
    meaningful ranking is possible.

    The ``per_feature`` method processes one feature at a time; the ``dense`` method
    is the historical vectorized implementation optimized for speed at the cost
    of a large temporary 3D buffer.
    """

    _assert_quantile_inputs(observed_df, null_df)

    if method == QUANTILE_METHODS.PER_FEATURE:
        return _calculate_quantiles_per_feature(
            observed_df, null_df, comparison_dtype=comparison_dtype
        )
    elif method == QUANTILE_METHODS.DENSE:
        return _calculate_quantiles_dense(
            observed_df, null_df, comparison_dtype=comparison_dtype
        )
    else:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: {VALID_QUANTILE_METHODS}"
        )


def _calculate_quantiles_dense(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    *,
    comparison_dtype: Any,
) -> pd.DataFrame:
    """Midrank quantiles: stack all nulls into a dense 3D array and vectorize.

    Favors CPU throughput when the problem fits in memory; materializes
    shape ``(n_features, n_null_samples, n_attributes)`` (padded with NaN where
    per-feature null counts differ).
    """
    out_dtype = np.dtype(comparison_dtype)
    null_grouped = null_df.groupby(level=0)
    sizes = [len(g) for _, g in null_grouped]
    if not sizes:
        raise ValueError("null_df has no rows")
    max_null_samples = max(sizes)
    n_col = len(observed_df.columns)
    n_row = len(observed_df)
    null_array = np.full((n_row, max_null_samples, n_col), np.nan, dtype=out_dtype)
    actual_sample_counts = np.zeros(n_row, dtype=int)

    for _, (feature, group) in enumerate(null_grouped):
        feature_idx = observed_df.index.get_loc(feature)
        block = np.asarray(group.values, dtype=out_dtype, order="C")
        if block.ndim == 1:
            block = block.reshape(1, -1)
        n_g = block.shape[0]
        if not isinstance(feature_idx, (int, np.integer)):
            raise NotImplementedError(
                "Non-scalar index from observed index (e.g. duplicate labels) is not "
                "supported for the dense quantile method; use method='per_feature'."
            )
        null_array[feature_idx, :n_g, :] = block
        actual_sample_counts[feature_idx] = n_g

    observed_cmp = np.asarray(observed_df.values, dtype=out_dtype, order="C")
    less_than = np.nansum(null_array < observed_cmp[:, np.newaxis, :], axis=1)
    equal_to = np.nansum(null_array == observed_cmp[:, np.newaxis, :], axis=1)
    all_identical = equal_to == actual_sample_counts[:, np.newaxis]
    with np.errstate(invalid="ignore", divide="ignore"):
        quantiles = (less_than + equal_to / 2.0) / actual_sample_counts[:, np.newaxis]
    quantiles[all_identical] = np.nan
    return pd.DataFrame(quantiles, index=observed_df.index, columns=observed_df.columns)


def _calculate_quantiles_per_feature(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    *,
    comparison_dtype: Any,
) -> pd.DataFrame:
    """Midrank quantiles: one (n_null, n_attr) block per feature; no 3D tensor."""
    out_dtype = np.dtype(comparison_dtype)
    observed_values = np.asarray(observed_df.values, dtype=np.float64)
    n_features = len(observed_df)
    quantiles_out = np.empty((n_features, len(observed_df.columns)), dtype=np.float64)

    for i, feature in enumerate(observed_df.index):
        null_block = _coerce_to_2d_null_block(
            null_df.loc[feature], out_dtype
        )  # (n_null, n_attributes)
        obs_row = np.asarray(observed_values[i, :], dtype=out_dtype)
        n = int(null_block.shape[0])
        less_than = (null_block < obs_row).sum(axis=0, dtype=np.int64)
        equal_to = (null_block == obs_row).sum(axis=0, dtype=np.int64)
        all_identical = equal_to == n
        with np.errstate(invalid="ignore"):
            qrow = (less_than + equal_to / 2.0) / float(n)
        qrow = qrow.astype(np.float64, copy=False)
        qrow[all_identical] = np.nan
        quantiles_out[i, :] = qrow

    return pd.DataFrame(
        quantiles_out, index=observed_df.index, columns=observed_df.columns
    )


def _assert_quantile_inputs(observed_df: pd.DataFrame, null_df: pd.DataFrame) -> None:
    if not observed_df.columns.equals(null_df.columns):
        raise ValueError("Column names must match between observed and null data")

    missing_features = set(observed_df.index) - set(null_df.index)
    if missing_features:
        raise ValueError(f"Missing features in null data: {missing_features}")

    if observed_df.isna().any().any():
        raise ValueError("NaN values found in observed data")
    if null_df.isna().any().any():
        raise ValueError("NaN values found in null data")

    null_grouped = null_df.groupby(level=0)
    sample_counts = {name: len(group) for name, group in null_grouped}
    if len(set(sample_counts.values())) > 1:
        logger.warning("Unequal null sample counts per feature may affect results")


def _coerce_to_2d_null_block(
    null_slice: Union[pd.DataFrame, pd.Series], dtype: Any
) -> np.ndarray:
    """Return (n_null_samples, n_attributes) array for one feature's null draws."""
    if isinstance(null_slice, pd.Series):
        arr = null_slice.to_numpy(dtype=dtype, copy=False)
        return arr.reshape(1, -1)
    return null_slice.to_numpy(dtype=dtype, copy=False)
