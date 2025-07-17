import numpy as np
from scipy.stats import fisher_exact

from napistu.statistics import hypothesis_testing


def test_fisher_exact_vectorized_basic_and_vectorized():

    # Classic Fisher's test example: [[1, 9], [11, 3]]
    # a=1, b=9, c=11, d=3
    odds, p = hypothesis_testing.fisher_exact_vectorized([1], [9], [11], [3])
    # Odds ratio: (1*3)/(9*11) = 3/99 = 0.0303...
    assert np.allclose(odds, [3 / 99])
    assert p.shape == (1,)
    assert (p >= 0).all() and (p <= 1).all()

    # Vectorized: two tables
    odds, p = hypothesis_testing.fisher_exact_vectorized(
        [1, 2], [9, 8], [11, 10], [3, 4]
    )
    assert odds.shape == (2,)
    assert p.shape == (2,)
    # Check that odds ratios are correct
    expected_odds = np.array([(1 * 3) / (9 * 11), (2 * 4) / (8 * 10)])
    assert np.allclose(odds, expected_odds)
    # P-values should be between 0 and 1
    assert (p >= 0).all() and (p <= 1).all()


def test_fisher_exact_vectorized_vs_scipy():

    # Define several 2x2 tables
    tables = [
        ([1], [9], [11], [3]),
        ([5], [2], [8], [7]),
        ([10], [10], [10], [10]),
        ([0], [5], [5], [10]),
        ([3], [7], [2], [8]),
    ]
    for a, b, c, d in tables:
        odds_vec, p_vec = hypothesis_testing.fisher_exact_vectorized(a, b, c, d)
        # Build the table for scipy
        table = np.array([[a[0], b[0]], [c[0], d[0]]])
        odds_scipy, p_scipy = fisher_exact(table, alternative="greater")
        # Odds ratios should be nearly identical
        assert np.allclose(odds_vec, [odds_scipy], rtol=1e-6, atol=1e-8)
        # P-values should be close (normal approx vs exact)
        assert np.allclose(
            p_vec, [p_scipy], rtol=0.15, atol=1e-3
        )  # allow some tolerance

    # Also test vectorized input
    a = [1, 5, 10, 0, 3]
    b = [9, 2, 10, 5, 7]
    c = [11, 8, 10, 5, 2]
    d = [3, 7, 10, 10, 8]
    odds_vec, p_vec = hypothesis_testing.fisher_exact_vectorized(a, b, c, d)
    for i in range(len(a)):
        table = np.array([[a[i], b[i]], [c[i], d[i]]])
        odds_scipy, p_scipy = fisher_exact(table, alternative="greater")
        assert np.allclose(odds_vec[i], odds_scipy, rtol=1e-6, atol=1e-8)
        assert np.allclose(p_vec[i], p_scipy, rtol=0.15, atol=1e-3)
