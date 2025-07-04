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
    ultra-fast vectorized operations.

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
        Each value represents the proportion of null values <= observed value.
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

    # Fill the null array
    for i, (feature, group) in enumerate(null_grouped):
        feature_idx = observed_df.index.get_loc(feature)
        null_array[feature_idx, : len(group)] = group.values

    # Broadcast comparison: observed[features, 1, attributes] vs null[features, samples, attributes]
    # This creates a boolean array of shape [features, null_samples, attributes]
    # Less than or equal to is used to calculate the quantile consistent with the R quantile function
    comparisons = null_array <= observed_values[:, np.newaxis, :]

    # Calculate quantiles by taking mean along the null_samples axis
    # Use nanmean to handle padded NaN values
    quantiles = np.nanmean(comparisons, axis=1)

    return pd.DataFrame(quantiles, index=observed_df.index, columns=observed_df.columns)
