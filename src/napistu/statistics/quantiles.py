"""Module for comparing observed values to null distributions."""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_quantiles(
    observed_df: pd.DataFrame, null_df: pd.DataFrame
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
    """

    if not observed_df.columns.equals(null_df.columns):
        raise ValueError("Column names must match between observed and null data")

    # Validate all features present
    missing_features = set(observed_df.index) - set(null_df.index)
    if missing_features:
        raise ValueError(f"Missing features in null data: {missing_features}")

    # Check for NaN values
    if observed_df.isna().any().any():
        raise ValueError("NaN values found in observed data")
    if null_df.isna().any().any():
        raise ValueError("NaN values found in null data")

    # Check for unequal sample sizes and warn
    null_grouped = null_df.groupby(level=0)
    sample_counts = {name: len(group) for name, group in null_grouped}
    if len(set(sample_counts.values())) > 1:
        logger.warning("Unequal null sample counts per feature may affect results")

    # Convert to numpy arrays for speed
    observed_values = observed_df.values

    # Group null data and stack into 3D array
    null_grouped = null_df.groupby(level=0)

    # Get the maximum number of null samples per feature
    max_null_samples = max(len(group) for _, group in null_grouped)

    # Pre-allocate 3D array: [features, null_samples, attributes]
    null_array = np.full(
        (len(observed_df), max_null_samples, len(observed_df.columns)), np.nan
    )

    # Fill the null array and track actual sample counts
    actual_sample_counts = np.zeros(len(observed_df), dtype=int)

    for i, (feature, group) in enumerate(null_grouped):
        feature_idx = observed_df.index.get_loc(feature)
        null_array[feature_idx, : len(group)] = group.values
        actual_sample_counts[feature_idx] = len(group)

    # Midrank method - count values strictly less than observed
    less_than = np.nansum(null_array < observed_values[:, np.newaxis, :], axis=1)

    # Count values equal to observed
    equal_to = np.nansum(null_array == observed_values[:, np.newaxis, :], axis=1)

    # Check for cases where all values are identical (no variance)
    # This happens when equal_to equals the total sample count
    all_identical = equal_to == actual_sample_counts[:, np.newaxis]

    # Midrank formula: (less_than + equal_to/2) / total
    quantiles = (less_than + equal_to / 2) / actual_sample_counts[:, np.newaxis]

    # Set NaN where all values are identical (no meaningful quantile)
    quantiles[all_identical] = np.nan

    return pd.DataFrame(quantiles, index=observed_df.index, columns=observed_df.columns)
