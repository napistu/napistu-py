"""
Hypothesis tests.

Public Functions
----------------
fisher_exact_vectorized(observed_members, missing_members, observed_nonmembers, nonobserved_nonmembers)
    Fast vectorized one-tailed Fisher exact test using normal approximation.
neat_edge_enrichment_test(observed_edges, out_degrees_a, in_degrees_b, total_edges, same_set)
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
    total_edges: int,
    same_set: bool = False,
) -> Dict[str, float]:
    """
    NEAT degree-corrected edge enrichment test.

    Works for both directed and undirected graphs without branching.
    For undirected: out_degrees_a == in_degrees_a and in_degrees_b == out_degrees_b

    Parameters
    ----------
    observed_edges : int
        Number of edges observed between gene sets A and B
    out_degrees_a : np.ndarray
        Out-degrees of nodes in gene set A (or total degrees if undirected)
    in_degrees_b : np.ndarray
        In-degrees of nodes in gene set B (or total degrees if undirected)
    total_edges : int
        Total number of edges in the universe
    same_set : bool
        Whether A and B are the same gene set (affects self-loop correction)

    Returns
    -------
    dict
        Statistical test results with keys:
        - observed: int
        - expected: float
        - variance: float
        - z_score: float
        - p_value: float
    """
    # Expected edges: sum of (out_deg_i * in_deg_j) / total_edges
    degree_products = np.outer(out_degrees_a, in_degrees_b)
    expected = np.sum(degree_products) / total_edges

    # Self-loop correction for same set
    if same_set:
        # For same set, degrees should be identical (out_degrees_a == in_degrees_b)
        # Remove self-loops: sum of (deg_i^2) / total_edges
        expected -= np.sum(out_degrees_a**2) / total_edges

    # Exact variance: sum of p_ij(1 - p_ij)
    p_ij = degree_products / total_edges
    variance = np.sum(p_ij * (1 - p_ij))

    # Self-loop variance correction
    if same_set:
        self_probs = (out_degrees_a**2) / total_edges
        variance -= np.sum(self_probs * (1 - self_probs))

    # Z-score and p-value
    if variance == 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = (observed_edges - expected) / np.sqrt(variance)
        p_value = norm.sf(z_score)  # one-tailed upper tail

    return {
        "observed": observed_edges,
        "expected": float(expected),
        "variance": float(variance),
        "z_score": float(z_score),
        "p_value": float(p_value),
    }
