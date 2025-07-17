from typing import Union

import numpy as np
from scipy.stats import norm


def fisher_exact_vectorized(
    observed_members: Union[list[int], np.ndarray],
    missing_members: Union[list[int], np.ndarray],
    observed_nonmembers: Union[list[int], np.ndarray],
    nonobserved_nonmembers: Union[list[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized one-tailed Fisher exact test using normal approximation.

    Parameters:
    -----------
    observed_members, missing_members, observed_nonmembers, nonobserved_nonmembers : array-like
        The four cells of the 2x2 contingency tables (must be non-negative)

    Returns:
    --------
    odds_ratios : numpy array
        Odds ratios for each test
    p_values : numpy array
        One-tailed p-values (tests for enrichment)
    """
    # Convert to numpy arrays
    a = np.array(observed_members, dtype=float)
    b = np.array(missing_members, dtype=float)
    c = np.array(observed_nonmembers, dtype=float)
    d = np.array(nonobserved_nonmembers, dtype=float)

    # Check for negative values and raise error
    if np.any((a < 0) | (b < 0) | (c < 0) | (d < 0)):
        raise ValueError("All contingency table values must be non-negative")

    # Calculate odds ratios
    odds_ratios = np.divide(
        a * d, b * c, out=np.full_like(a, np.inf, dtype=float), where=(b * c) != 0
    )

    # Normal approximation to hypergeometric distribution
    n = a + b + c + d

    # Avoid division by zero in expected value calculation
    expected_a = np.divide(
        (a + b) * (a + c), n, out=np.zeros_like(n, dtype=float), where=n != 0
    )

    # Variance calculation with protection against division by zero
    var_a = np.divide(
        (a + b) * (c + d) * (a + c) * (b + d),
        n * n * (n - 1),
        out=np.ones_like(n, dtype=float),  # Default to 1 to avoid sqrt(0)
        where=(n > 1),
    )
    var_a = np.maximum(var_a, 1e-10)  # Ensure positive variance

    # Continuity correction and z-score
    z = (a - expected_a - 0.5) / np.sqrt(var_a)

    # One-tailed p-value (upper tail for enrichment)
    p_values = norm.sf(z)  # 1 - norm.cdf(z)

    return odds_ratios, p_values
