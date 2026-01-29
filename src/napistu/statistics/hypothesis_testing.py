"""
Hypothesis tests.

Public Functions
----------------
fisher_exact_vectorized(observed_members, missing_members, observed_nonmembers, nonobserved_nonmembers)
    Fast vectorized one-tailed Fisher exact test using normal approximation.
neat_edge_enrichment_test(observed_edges, out_degrees_a, in_degrees_b, total_edges_universe, total_edges_observed)
    NEAT degree-corrected edge enrichment test.
"""

from typing import Dict, Union

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


def neat_edge_enrichment_test(
    observed_edges: int,
    out_degrees_a: np.ndarray,
    in_degrees_b: np.ndarray,
    total_edges_universe: int,
    total_edges_observed: int,
) -> Dict[str, float]:
    """
    NEAT degree-corrected edge enrichment test.

    Returns
    -------
    dict
        Statistical test results with keys:
        - observed: int
        - expected: float
        - variance: float
        - z_score: float
        - p_value: float
        - n_genes_a: int (number of genes in set A)
        - n_genes_b: int (number of genes in set B)
        - sum_out_deg_a: float (sum of out-degrees in set A)
        - sum_in_deg_b: float (sum of in-degrees in set B)
        - total_edges_universe: int
        - total_edges_observed: int
    """
    # Expected edges between sets A and B in the FULL universe
    sum_a = np.sum(out_degrees_a)
    sum_b = np.sum(in_degrees_b)
    expected_universe = (sum_a * sum_b) / total_edges_universe

    # Variance in the FULL universe
    sum_a_sq = np.sum(out_degrees_a**2)
    sum_b_sq = np.sum(in_degrees_b**2)
    variance_universe = expected_universe - (sum_a_sq * sum_b_sq) / (
        total_edges_universe**2
    )

    # Scale to the observed sample size
    sampling_fraction = total_edges_observed / total_edges_universe
    expected = expected_universe * sampling_fraction
    variance = variance_universe * sampling_fraction

    # Ensure variance is non-negative (numerical stability)
    variance = max(0, variance)

    # Z-score and p-value
    if variance == 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = (observed_edges - expected) / np.sqrt(variance)
        from scipy.stats import norm

        p_value = norm.sf(z_score)

    return {
        "observed": observed_edges,
        "expected": float(expected),
        "variance": float(variance),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n_genes_a": len(out_degrees_a),
        "n_genes_b": len(in_degrees_b),
        "sum_out_deg_a": float(sum_a),
        "sum_in_deg_b": float(sum_b),
        "total_edges_universe": total_edges_universe,
        "total_edges_observed": total_edges_observed,
    }
