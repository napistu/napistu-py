from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, NamedTuple
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def discretize_expression_data(
    expression_data: pd.DataFrame,
    metadata_attributes: list[str] = None,
    min_row_sum: int = 50,
    zfpm_threshold: float = -3,
    min_peakheight: float = 0.02,
    min_peakdistance: int = 1,
    prominence: float = 0.05,
    verbose: bool = False,
):
    """
    Discretize the GTEx data

    Parameters
    ----------
    expression_data: pandas DataFrame
        The expression data to discretize
    metadata_attributes: list[str], optional
        Non-numeric and other metadata attributes which should be included in the output but ignored when discretizing expression data
    min_row_sum: int, optional
        The minimum row sum to use for filtering constituatively un-expressed genes
    zfpm_threshold: float, optional
        The zFPKM threshold to use for discretization. Samples with zFPKM values below this threshold are considered as unexpressed (0) in the sample/condition.
    min_peakheight: float, optional
        The minimum peak height to use for peak detection
    min_peakdistance: int, optional
        The minimum peak distance to use for peak detection
    prominence: float, optional
        The prominence to use for peak detection
    verbose: bool, optional
        Whether to print verbose output

    Returns
    -------
    tuple of pandas DataFrames
        A tuple of two pandas DataFrames. The first DataFrame contains the zFPKM-transformed expression data with the metadata attributes merged on the left. The second DataFrame contains the expression data with binary values (0 for unexpressed, 1 for expressed) merged on the left.

    """

    expression_data_types = expression_data.dtypes
    if metadata_attributes is None:
        metadata_attributes = [
            col
            for col in expression_data_types.index
            if expression_data_types[col] == "object"
        ]

    expression_numpy_df = expression_data.drop(columns=metadata_attributes)
    # ensure that all variables are numeric
    invalid_variables = [
        x
        for x, y in zip(expression_numpy_df.columns, expression_numpy_df.dtypes)
        if y not in ["int64", "float64"]
    ]

    if len(invalid_variables) > 0:
        raise ValueError(
            f"The following variables are not numeric: {invalid_variables}. Either include these in metadata_attributes, convert them to numeric, or remove them from the expression data."
        )

    # calculate rowsums
    expression_numpy_df = expression_numpy_df.loc[
        expression_numpy_df.sum(axis=1) > min_row_sum, :
    ]

    n_unexpressed = expression_data.shape[0] - expression_numpy_df.shape[0]
    if n_unexpressed > 0:
        logger.info(
            f"Removed {n_unexpressed} genes whose expression across all samples was below {min_row_sum}."
        )

    logger.info("Discretizing expression data...")
    zfpkm_df = zfpkm(
        expression_numpy_df,
        min_peakheight=min_peakheight,
        min_peakdistance=min_peakdistance,
        prominence=prominence,
        verbose=verbose,
    )

    is_expressed = (zfpkm_df > zfpm_threshold).astype(int)
    n_expressed = sum(is_expressed.values.flatten())
    expression_fraction = round(n_expressed / zfpkm_df.size, 3)
    logger.info(f"Expression fraction: {expression_fraction}")

    if expression_fraction < 0.01:
        logger.warning(
            "Less than 1% of the data was expressed. This is likely due to the zFPKM threshold being too high."
        )

    return (
        pd.DataFrame(expression_data[metadata_attributes]).merge(
            zfpkm_df, left_index=True, right_index=True
        ),
        pd.DataFrame(expression_data[metadata_attributes]).merge(
            is_expressed, left_index=True, right_index=True
        ),
    )


def zfpkm(
    fpkm_df: pd.DataFrame,
    min_peakheight: float = 0.02,
    min_peakdistance: int = 1,
    prominence: float = 0.05,
    verbose: bool = False,
) -> pd.DataFrame:
    """Transform entire DataFrame using zFPKM.

    Parameters
    ----------
    fpkm_df : pd.DataFrame
        DataFrame containing raw FPKM values.
        Rows = genes/transcripts, Columns = samples
    min_peakheight : float, optional
        Minimum height for peak detection, by default 0.02
    min_peakdistance : int, optional
        Minimum distance between peaks, by default 1
    prominence : float, optional
        Minimum prominence for peak detection, by default 0.05
    verbose : bool, optional
        Whether to log detailed information, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with zFPKM transformed values
    """
    # Remove problematic rows
    fpkm_df = _remove_nan_inf_rows(fpkm_df)

    zfpkm_df = pd.DataFrame(index=fpkm_df.index)

    for col in fpkm_df.columns:
        z_fpkm = _zfpkm_calc(
            fpkm_df[col], min_peakheight, min_peakdistance, prominence, verbose
        )
        zfpkm_df[col] = z_fpkm

    return zfpkm_df


class PeakIndices(NamedTuple):
    """Container for peak indices classified by importance.

    Parameters
    ----------
    major : float
        Position of the rightmost/highest peak
    minor : Optional[float]
        Position of the second most significant peak, if it exists
    other : Optional[np.ndarray]
        Positions of any remaining peaks
    """

    major: float
    minor: Optional[float]
    other: Optional[np.ndarray]


class PeakSelector:
    """Class to handle peak detection and classification in density data.

    Parameters
    ----------
    min_peakheight : float, optional
        Minimum height for peak detection, by default 0.02
    min_peakdistance : int, optional
        Minimum distance between peaks, by default 1
    prominence : float, optional
        Minimum prominence for peak detection, by default 0.05
    verbose : bool, optional
        Whether to log detailed information, by default True
    """

    def __init__(
        self,
        min_peakheight: float = 0.02,
        min_peakdistance: int = 1,
        prominence: float = 0.05,
        verbose: bool = False,
    ):
        self.min_peakheight = min_peakheight
        self.min_peakdistance = min_peakdistance
        self.prominence = prominence
        self.verbose = verbose

    def find_peaks(self, density_y: np.ndarray, x_eval: np.ndarray) -> PeakIndices:
        """Find and classify peaks in density data.

        Parameters
        ----------
        density_y : np.ndarray
            Y-values of the density estimation
        x_eval : np.ndarray
            X-values corresponding to density_y

        Returns
        -------
        PeakIndices
            Named tuple containing classified peak positions
        """
        peaks, _ = find_peaks(
            density_y,
            height=self.min_peakheight,
            distance=self.min_peakdistance,
            prominence=self.prominence,
        )

        logger.debug("Found %d peaks", len(peaks))

        if len(peaks) == 0:
            # If no peaks found, use the maximum density point
            peak_idx = np.argmax(density_y)
            return PeakIndices(major=x_eval[peak_idx], minor=None, other=None)

        # Get peak positions and sort by x-value
        peak_positions = x_eval[peaks]
        sorted_indices = np.argsort(peak_positions)
        sorted_positions = peak_positions[sorted_indices]

        # Always use rightmost peak as major
        major = sorted_positions[-1]

        # If we have more peaks, classify them
        minor = sorted_positions[-2] if len(sorted_positions) > 1 else None
        other = sorted_positions[:-2] if len(sorted_positions) > 2 else None

        if self.verbose:
            logger.info(
                "Major peak at %.3f, minor peak at %.3f",
                major,
                minor if minor is not None else float("nan"),
            )
            if other is not None:
                logger.debug("Additional peaks at: %s", other)

        return PeakIndices(major=major, minor=minor, other=other)


def generate_simple_test_data(n_genes: int = 200, n_samples: int = 100) -> pd.DataFrame:
    """Generate a simple test dataset for basic validation.

    Parameters
    ----------
    n_genes : int, optional
        Number of genes to generate, by default 200
    n_samples : int, optional
        Number of samples to generate, by default 50

    Returns
    -------
    pd.DataFrame
        DataFrame with simulated FPKM values.
        Rows = genes, Columns = samples
    """
    np.random.seed(42)

    fpkm_data = np.zeros((n_genes, n_samples))

    for gene_idx in range(n_genes):
        active_fraction = np.random.uniform(0.2, 0.8)  # 20-80% of samples active
        expression_delta = np.random.gamma(shape=2, scale=0.5)
        noise_level = np.random.gamma(shape=2, scale=0.5)

        n_active_samples = int(active_fraction * n_samples)
        base_log_expr = np.random.normal(-1, 0.5)  # Slightly lower baseline

        log_expr = np.random.normal(base_log_expr, noise_level, n_samples)

        if n_active_samples > 0:
            active_samples = np.random.choice(
                n_samples, n_active_samples, replace=False
            )
            log_expr[active_samples] += expression_delta

        fpkm_data[gene_idx, :] = np.power(2, log_expr)

    gene_names = [f"ENSG{i:05d}" for i in range(n_genes)]
    sample_names = [f"Sample_{i+1:02d}" for i in range(n_samples)]

    return pd.DataFrame(fpkm_data, index=gene_names, columns=sample_names)


def _remove_nan_inf_rows(fpkm_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing all NaN or infinite values.

    Parameters
    ----------
    fpkm_df : pd.DataFrame
        Input DataFrame with FPKM values

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing all NaN or infinite values removed

    Notes
    -----
    Logs a warning if any rows are filtered out.
    """
    initial_rows = len(fpkm_df)
    clean_df = fpkm_df[
        ~fpkm_df.apply(lambda row: row.isna().all() or np.isinf(row).all(), axis=1)
    ]
    filtered_rows = initial_rows - len(clean_df)

    if filtered_rows > 0:
        logger.warning(
            "Filtered out %d rows containing all NaN or infinite values (from %d total rows)",
            filtered_rows,
            initial_rows,
        )

    return clean_df


def _zfpkm_calc(
    fpkm: Union[np.ndarray, pd.Series],
    min_peakheight: float = 0.02,
    min_peakdistance: int = 1,
    prominence: float = 0.05,
    verbose: bool = False,
) -> np.ndarray:
    """Perform zFPKM transform on a single sample of FPKM data.

    The zFPKM algorithm fits a kernel density estimate to the log2(FPKM)
    distribution of ALL GENES within a single sample. This requires:
    - Input: A vector of FPKM values for all genes in ONE sample
    - Many genes (typically 1000+) for meaningful density estimation
    - The algorithm identifies the rightmost peak as "active" gene expression

    Parameters
    ----------
    fpkm : Union[np.ndarray, pd.Series]
        Raw FPKM values for all genes in ONE sample (NOT log2 transformed)
    min_peakheight : float, optional
        Minimum height for peak detection, by default 0.02
    min_peakdistance : int, optional
        Minimum distance between peaks, by default 1
    prominence : float, optional
        Minimum prominence for peak detection, by default 0.05
    verbose : bool, optional
        Whether to log debug information, by default False

    Returns
    -------
    np.ndarray
        Array of zFPKM values

    Raises
    ------
    ValueError
        If no valid FPKM values are found after filtering
    """

    # Convert to numpy array and handle non-numeric values
    fpkm = np.array(fpkm, dtype=float)
    initial_len = len(fpkm)
    fpkm = fpkm[~(np.isnan(fpkm) | np.isinf(fpkm))]

    if len(fpkm) < initial_len:
        logger.warning(
            "Filtered out %d NaN/infinite values from input vector of length %d",
            initial_len - len(fpkm),
            initial_len,
        )

    if len(fpkm) == 0:
        raise ValueError("No valid FPKM values found")

    # Log2 transform - CRITICAL: Don't add small value, use 0 for exact zeros
    fpkm_log2 = np.log2(np.maximum(fpkm, 1e-10))

    # Compute kernel density estimate using R-like bandwidth
    kde = gaussian_kde(fpkm_log2)

    # Create evaluation points matching R's approach
    x_min, x_max = fpkm_log2.min(), fpkm_log2.max()
    x_range = x_max - x_min
    x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 512)
    density_y = kde(x_eval)

    # Find peaks using PeakSelector
    peak_selector = PeakSelector(
        min_peakheight=min_peakheight,
        min_peakdistance=min_peakdistance,
        prominence=prominence,
        verbose=verbose,
    )
    peaks = peak_selector.find_peaks(density_y, x_eval)
    mu = peaks.major

    # Estimate standard deviation using method from Hart et al.
    active_samples = fpkm_log2[fpkm_log2 > mu]
    n_active = len(active_samples)
    n_total = len(fpkm_log2)
    active_fraction = n_active / n_total

    if verbose:
        logger.info(
            "Active samples: %d/%d (%.2f%%)", n_active, n_total, 100 * active_fraction
        )

    # Use Hart et al. method only if we have enough active samples
    min_active_samples = max(5, int(0.1 * n_total))  # At least 5 or 10% of samples

    if n_active >= min_active_samples and active_fraction >= 0.05:  # At least 5% active
        U = np.mean(active_samples)
        center = mu
        stdev = (U - mu) * np.sqrt(np.pi / 2)
        if verbose:
            logger.info(
                "Using Hart et al. method: mu=%.3f, U=%.3f, stdev=%.3f", mu, U, stdev
            )
    else:
        # Fall back to sample standard deviation
        stdev = np.std(fpkm_log2, ddof=1)  # Use sample std dev (N-1)
        # Use minor peak if it exists, otherwise use mean
        center = peaks.minor if peaks.minor is not None else peaks.major
        method_used = "sample_std"
        if verbose:
            logger.info(
                "Falling back to sample std dev: stdev=%.3f (too few active: %d)",
                stdev,
                n_active,
            )

    # Handle edge case where stdev might still be 0 or negative
    if stdev <= 0:
        stdev = 0.1  # Minimal fallback
        method_used = "minimal_fallback"
        center = np.mean(fpkm_log2)
        logger.warning(
            "Standard deviation calculation resulted in non-positive value. Using minimal fallback: %.3f",
            stdev,
        )

    # Compute zFPKM transform
    z_fpkm = (fpkm_log2 - center) / stdev

    if verbose:
        logger.info(
            "Method: %s, zFPKM range: %.3f to %.3f",
            method_used,
            z_fpkm.min(),
            z_fpkm.max(),
        )

    return z_fpkm
