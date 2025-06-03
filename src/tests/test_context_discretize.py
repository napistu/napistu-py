import numpy as np

from napistu.context import discretize


def test_peak_selector():

    # Test Case 0: No peaks (flat/monotonic)
    x0 = np.linspace(-5, 5, 100)
    y0 = np.exp(-x0)  # Monotonic decreasing
    peaks0 = discretize.PeakSelector().find_peaks(y0, x0)

    assert peaks0.major == -5
    assert peaks0.minor is None
    assert peaks0.other is None

    # Test Case 2: Single peak (Gaussian)
    x1 = np.linspace(-5, 5, 100)
    y1 = np.exp(-0.5 * x1**2)  # Single Gaussian peak
    peaks1 = discretize.PeakSelector().find_peaks(y1, x1)

    assert abs(peaks1.major - -0.05) < 0.01
    assert peaks1.minor is None
    assert peaks1.other is None

    # Test Case 3: Two peaks (bimodal)
    x2 = np.linspace(-5, 5, 100)
    y2 = 0.6 * np.exp(-0.5 * (x2 + 1.5) ** 2) + 0.4 * np.exp(-0.5 * (x2 - 1.5) ** 2)
    peaks2 = discretize.PeakSelector().find_peaks(y2, x2)

    assert abs(peaks2.major - 1.46) < 0.01
    assert abs(peaks2.minor - -1.46) < 0.01
    assert peaks2.other is None

    # Test Case 4: Three peaks (trimodal)
    x3 = np.linspace(-8, 8, 100)
    y3 = (
        0.4 * np.exp(-0.5 * (x3 + 4) ** 2)
        + 0.5 * np.exp(-0.5 * x3**2)
        + 0.3 * np.exp(-0.5 * (x3 - 4) ** 2)
    )
    peaks3 = discretize.PeakSelector().find_peaks(y3, x3)

    assert abs(peaks3.major - 3.95) < 0.01
    assert abs(peaks3.minor - -0.08) < 0.01
    assert abs(peaks3.other[0] - -3.95) < 0.01


def test_zfpkm():

    fpkm_df = discretize.generate_simple_test_data()
    zfpkm_df = discretize.zfpkm(fpkm_df)

    assert zfpkm_df.shape == fpkm_df.shape
    assert zfpkm_df.index.equals(fpkm_df.index)
    assert zfpkm_df.columns.equals(fpkm_df.columns)
