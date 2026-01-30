"""
Hypothesis tests.

Public Functions
----------------
binomial_test_vectorized(sample_successes, sample_total, population_successes, population_total)
    Fast vectorized one-tailed binomial test using normal approximation.
fisher_exact_vectorized(observed_members, missing_members, observed_nonmembers, nonobserved_nonmembers)
    Fast vectorized one-tailed Fisher exact test using normal approximation.
proportion_test_vectorized(sample_successes, sample_total, population_successes, population_total)
    Fast vectorized one-tailed proportion test using normal approximation.
"""

from typing import Union

import numpy as np
from scipy.stats import binom, norm


def binomial_test_vectorized(
    sample_successes, sample_total, population_successes, population_total
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binomial test for enrichment in sampled edges.

    H0: Sample edges are drawn proportionally from universe
    H1: This pathway pair is enriched in sample

    Parameters
    ----------
    sample_successes : array
        Observed edges for each pathway pair
    sample_total : int
        Total edges in sample (e.g., 10K)
    population_successes : array
        Universe edges for each pathway pair
    population_total : int
        Total edges in universe (e.g., 8M)

    Returns
    -------
    expected : array
        Expected edges under null
    p_values : array
        One-tailed p-values (upper tail)
    """

    # Probability under null hypothesis
    p_null = population_successes / population_total

    # Expected value
    expected = sample_total * p_null

    # P(X >= k) where X ~ Binomial(n=sample_total, p=p_null)
    # Using survival function: P(X >= k) = 1 - P(X <= k-1)
    p_values = binom.sf(sample_successes - 1, sample_total, p_null)

    # Handle edge cases (p_null = 0)
    p_values = np.where(p_null == 0, 1.0, p_values)

    return expected, p_values


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


def proportion_test_vectorized(
    sample_successes: Union[list[int], np.ndarray],
    sample_total: int,
    population_successes: Union[list[int], np.ndarray],
    population_total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized one-tailed proportion test using normal approximation.

    Tests whether the proportion of successes in a sample differs from the
    proportion in a reference population.

    Parameters
    ----------
    sample_successes : array-like
        Number of successes in the sample (must be non-negative)
    sample_total : int
        Total number of observations in the sample (must be positive)
    population_successes : array-like
        Number of successes in the population (must be non-negative)
    population_total : int
        Total number of observations in the population (must be positive)

    Returns
    -------
    expected_successes : numpy array
        Expected number of successes in sample under null hypothesis
    odds_ratios : numpy array
        Odds ratios for each test
    p_values : numpy array
        One-tailed p-values (tests for enrichment, upper tail)
    """

    # Convert to numpy arrays
    sample_succ = np.array(sample_successes, dtype=float)
    pop_succ = np.array(population_successes, dtype=float)

    # Check for negative values
    if np.any(sample_succ < 0) or np.any(pop_succ < 0):
        raise ValueError("All count values must be non-negative")
    if population_total <= 0 or sample_total <= 0:
        raise ValueError("Total counts must be positive")

    # Proportions
    p_sample = sample_succ / sample_total
    p_population = pop_succ / population_total

    # Expected under null hypothesis (sample has same proportion as population)
    expected = sample_total * p_population

    # Standard error for difference in proportions
    # SE = sqrt(p_pop * (1 - p_pop) * (1/n_sample + 1/n_pop))
    # For large population relative to sample, approximate as:
    # SE ≈ sqrt(p_pop * (1 - p_pop) / n_sample)
    se = np.sqrt(p_population * (1 - p_population) / sample_total)
    se = np.maximum(se, 1e-10)  # Ensure positive

    # Z-score: (p_sample - p_population) / SE
    z = (p_sample - p_population) / se

    # One-tailed p-value (upper tail for enrichment)
    p_values = norm.sf(z)

    # Handle edge cases
    p_values = np.where((sample_succ == 0) | (pop_succ == 0), 1.0, p_values)

    # Calculate odds ratios
    # OR = (sample_succ/sample_fail) / (pop_succ/pop_fail)
    sample_ratio = np.divide(
        sample_succ,
        sample_total - sample_succ,
        out=np.full_like(sample_succ, np.inf, dtype=float),
        where=(sample_succ < sample_total),
    )
    pop_ratio = np.divide(
        pop_succ,
        population_total - pop_succ,
        out=np.full_like(pop_succ, np.inf, dtype=float),
        where=(pop_succ < population_total),
    )
    odds_ratios = np.divide(
        sample_ratio,
        pop_ratio,
        out=np.full_like(sample_ratio, np.inf, dtype=float),
        where=(pop_ratio > 0),
    )

    return expected, odds_ratios, p_values
