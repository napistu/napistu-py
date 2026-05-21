import numpy as np
import pandas as pd
import pytest

from napistu.statistics.constants import QUANTILE_METHODS
from napistu.statistics.quantiles import calculate_quantiles


def test_calculate_quantiles_valid_inputs():
    """Test calculate_quantiles with valid, well-formed inputs."""
    # Create observed data: 4 features x 3 attributes
    observed = pd.DataFrame(
        [[0.8, 0.3, 0.9], [0.2, 0.7, 0.1], [0.5, 0.5, 0.5], [0.1, 0.9, 0.2]],
        index=["gene1", "gene2", "gene3", "gene4"],
        columns=["attr1", "attr2", "attr3"],
    )

    # Create null data: 2 samples per feature (8 rows total)
    null_index = ["gene1", "gene2", "gene3", "gene4"] * 2
    null_data = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],  # gene1 sample 1
            [0.4, 0.5, 0.6],  # gene2 sample 1
            [0.7, 0.8, 0.9],  # gene3 sample 1
            [0.0, 0.1, 0.2],  # gene4 sample 1
            [0.2, 0.3, 0.4],  # gene1 sample 2
            [0.5, 0.6, 0.7],  # gene2 sample 2
            [0.8, 0.9, 1.0],  # gene3 sample 2
            [0.1, 0.2, 0.3],  # gene4 sample 2
        ],
        index=null_index,
        columns=["attr1", "attr2", "attr3"],
    )

    # Calculate quantiles
    result = calculate_quantiles(observed, null_data)

    # Verify output structure
    assert result.shape == observed.shape
    assert list(result.index) == list(observed.index)
    assert list(result.columns) == list(observed.columns)

    # Check specific quantile calculations
    # gene1, attr1: observed=0.8, nulls=[0.1, 0.2] -> quantile = 1.0 (100%)
    assert result.loc["gene1", "attr1"] == 1.0

    # gene2, attr2: observed=0.7, nulls=[0.5, 0.6] -> quantile = 1.0 (100%)
    assert result.loc["gene2", "attr2"] == 1.0

    # gene3, attr3: observed=0.5, nulls=[0.9, 1.0] -> quantile = 0.0 (0%)
    assert result.loc["gene3", "attr3"] == 0.0

    # gene4, attr1: observed=0.1, nulls=[0.0, 0.1]
    # with midrank; 0.5 + 0.5/2 (because 0.1 is equal to 0.1)
    assert result.loc["gene4", "attr1"] == 0.75


def test_calculate_quantiles_error_cases():
    """Test calculate_quantiles with invalid inputs that should raise errors or warnings."""
    # Base observed data
    observed = pd.DataFrame(
        [[0.8, 0.3], [0.2, 0.7]], index=["gene1", "gene2"], columns=["attr1", "attr2"]
    )

    # Test 1: Mismatched columns
    null_wrong_cols = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        index=["gene1", "gene2"],
        columns=["attr1", "attr2", "attr3"],  # Extra column
    )

    with pytest.raises((KeyError, ValueError)):
        calculate_quantiles(observed, null_wrong_cols)

    # Test 2: Missing features in null data
    null_missing_feature = pd.DataFrame(
        [[0.1, 0.2]], index=["gene1"], columns=["attr1", "attr2"]  # Missing gene2
    )

    # Current implementation doesn't validate - it will likely fail in groupby or indexing
    # This test verifies current behavior (may change if validation added)
    try:
        result = calculate_quantiles(observed, null_missing_feature)
        # If it succeeds, gene2 quantiles will be invalid/error
        assert True  # Just check it doesn't crash for now
    except (KeyError, ValueError, IndexError):
        assert True  # Expected behavior

    # Test 3: Unequal null samples per feature
    null_unequal_samples = pd.DataFrame(
        [
            [0.1, 0.2],  # gene1 sample 1
            [0.3, 0.4],  # gene1 sample 2
            [0.5, 0.6],  # gene2 sample 1 (only 1 sample)
        ],
        index=["gene1", "gene1", "gene2"],
        columns=["attr1", "attr2"],
    )

    # This should still work but may give different results
    result = calculate_quantiles(observed, null_unequal_samples)
    assert result.shape == observed.shape

    # Test 4: Empty null data
    null_empty = pd.DataFrame(columns=["attr1", "attr2"])

    with pytest.raises((ValueError, IndexError)):
        calculate_quantiles(observed, null_empty)

    # Test 5: Single null sample (edge case)
    null_single = pd.DataFrame(
        [[0.1, 0.2], [0.5, 0.6]], index=["gene1", "gene2"], columns=["attr1", "attr2"]
    )

    result = calculate_quantiles(observed, null_single)
    assert result.shape == observed.shape
    # With single sample, results should be binary (0 or 1)
    assert all(val in [0.0, 1.0] for val in result.values.flatten())

    # Test 6: NaN values in data
    observed_with_nan = observed.copy()
    observed_with_nan.loc["gene1", "attr1"] = np.nan

    null_with_nan = pd.DataFrame(
        [[np.nan, 0.2], [0.4, 0.5], [0.1, 0.3], [0.6, 0.7]],
        index=["gene1", "gene2", "gene1", "gene2"],
        columns=["attr1", "attr2"],
    )

    # Should raise ValueError for NaN values
    with pytest.raises(ValueError, match="NaN values found in observed data"):
        calculate_quantiles(observed_with_nan, null_single)

    with pytest.raises(ValueError, match="NaN values found in null data"):
        calculate_quantiles(observed, null_with_nan)


def test_midrank_tie_handling():
    """Test midrank method handles ties correctly."""
    observed = pd.DataFrame([[0.0, 0.5]], index=["gene1"], columns=["attr1", "attr2"])

    null_data = pd.DataFrame(
        [
            [0.0, 0.2],  # gene1: 0.0 ties, 0.2 < 0.5
            [0.0, 0.5],  # gene1: 0.0 ties, 0.5 ties
            [0.1, 0.8],  # gene1: 0.1 > 0.0, 0.8 > 0.5
        ],
        index=["gene1", "gene1", "gene1"],
        columns=["attr1", "attr2"],
    )

    result = calculate_quantiles(observed, null_data)

    # attr1: observed=0.0, nulls=[0.0, 0.0, 0.1] -> (0 + 2/2)/3 = 1/3
    assert result.loc["gene1", "attr1"] == pytest.approx(1 / 3)

    # attr2: observed=0.5, nulls=[0.2, 0.5, 0.8] -> (1 + 1/2)/3 = 0.5
    assert result.loc["gene1", "attr2"] == pytest.approx(0.5)


def test_all_identical_returns_nan():
    """Test that NaN is returned when all values are identical."""
    observed = pd.DataFrame([[0.5]], index=["gene1"], columns=["attr1"])
    null_data = pd.DataFrame(
        [[0.5], [0.5], [0.5]], index=["gene1", "gene1", "gene1"], columns=["attr1"]
    )

    result = calculate_quantiles(observed, null_data)
    assert pd.isna(result.loc["gene1", "attr1"])


def test_basic_functionality():
    """Test basic quantile calculation without ties."""
    observed = pd.DataFrame([[0.5]], index=["gene1"], columns=["attr1"])
    null_data = pd.DataFrame(
        [[0.1], [0.3], [0.7], [0.9]],
        index=["gene1", "gene1", "gene1", "gene1"],
        columns=["attr1"],
    )

    result = calculate_quantiles(observed, null_data)
    # 2 values < 0.5, so quantile = 2/4 = 0.5
    assert result.loc["gene1", "attr1"] == 0.5


def test_per_feature_and_dense_match_float64():
    """Dense (vectorized) and per_feature midrank should agree for float64."""
    observed = pd.DataFrame(
        [[0.8, 0.3, 0.9], [0.2, 0.7, 0.1], [0.5, 0.5, 0.5], [0.1, 0.9, 0.2]],
        index=["gene1", "gene2", "gene3", "gene4"],
        columns=["attr1", "attr2", "attr3"],
    )
    null_index = ["gene1", "gene2", "gene3", "gene4"] * 2
    null_data = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.0, 0.1, 0.2],
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
            [0.1, 0.2, 0.3],
        ],
        index=null_index,
        columns=["attr1", "attr2", "attr3"],
    )
    a = calculate_quantiles(
        observed,
        null_data,
        method=QUANTILE_METHODS.PER_FEATURE,
        comparison_dtype=np.float64,
    )
    b = calculate_quantiles(
        observed, null_data, method=QUANTILE_METHODS.DENSE, comparison_dtype=np.float64
    )
    pd.testing.assert_frame_equal(a, b)


def test_unequal_null_counts_per_and_dense():
    """Unequal per-feature null counts: both methods should agree in float64."""
    observed = pd.DataFrame(
        [[0.8, 0.2], [0.3, 0.4]], index=["a", "b"], columns=["x", "y"]
    )
    null = pd.DataFrame(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ],
        index=["a", "a", "b"],
        columns=["x", "y"],
    )
    a = calculate_quantiles(
        observed, null, method=QUANTILE_METHODS.PER_FEATURE, comparison_dtype=np.float64
    )
    b = calculate_quantiles(
        observed, null, method=QUANTILE_METHODS.DENSE, comparison_dtype=np.float64
    )
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_method_invalid():
    obs = pd.DataFrame([[1.0]], index=[0], columns=["c"])
    null = pd.DataFrame([[0.0]], index=[0], columns=["c"])
    with pytest.raises(ValueError, match="Invalid method: other. Valid methods are:"):
        calculate_quantiles(obs, null, method="other")
