
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import warnings

class ZFPKMResult:
    """Class to store zFPKM results and related metrics"""
    def __init__(self, z, density_x, density_y, mu, stdev, max_fpkm):
        self.z = z
        self.density_x = density_x
        self.density_y = density_y
        self.mu = mu
        self.stdev = stdev
        self.max_fpkm = max_fpkm

def remove_nan_inf_rows(fpkm_df):
    """Remove rows containing all NaN or infinite values"""
    return fpkm_df[~fpkm_df.apply(lambda row: row.isna().all() or np.isinf(row).all(), axis=1)]

def zfpkm_calc(fpkm, peak_parameters=None):
    """
    Perform zFPKM transform on a single SAMPLE (column) of FPKM data
    
    The zFPKM algorithm fits a kernel density estimate to the log2(FPKM) 
    distribution of ALL GENES within a single sample. This requires:
    - Input: A vector of FPKM values for all genes in ONE sample
    - Many genes (typically 1000+) for meaningful density estimation
    - The algorithm identifies the rightmost peak as "active" gene expression
    
    Parameters:
    fpkm: array-like of raw FPKM values for all genes in ONE sample (NOT log2 transformed)
    peak_parameters: dict with 'minpeakheight' and 'minpeakdistance' keys
    
    Returns:
    ZFPKMResult object containing transformed data and metrics
    """
    if peak_parameters is None:
        peak_parameters = {'minpeakheight': 0.02, 'minpeakdistance': 1}
    
    # Convert to numpy array and handle non-numeric values
    fpkm = np.array(fpkm, dtype=float)
    fpkm = fpkm[~(np.isnan(fpkm) | np.isinf(fpkm))]
    
    if len(fpkm) == 0:
        raise ValueError("No valid FPKM values found")
    
    # Log2 transform - CRITICAL: Don't add small value, use 0 for exact zeros
    fpkm_log2 = np.log2(np.maximum(fpkm, 1e-10))
    
    # Compute kernel density estimate using R-like bandwidth
    # R uses bandwidth estimation similar to this
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(fpkm_log2)
    
    # Create evaluation points matching R's approach
    x_min, x_max = fpkm_log2.min(), fpkm_log2.max()
    x_range = x_max - x_min
    x_eval = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 512)
    density_y = kde(x_eval)
    
    # Find peaks - be more permissive like R
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(
        density_y, 
        height=peak_parameters['minpeakheight'],
        distance=peak_parameters['minpeakdistance']
    )
    
    if len(peaks) == 0:
        # If no peaks found, use the maximum density point
        peak_idx = np.argmax(density_y)
        mu = x_eval[peak_idx]
        print(f"Warning: No peaks found, using max density at mu={mu:.3f}")
    else:
        # Get the peak with the highest x-value (rightmost peak)
        peak_positions = x_eval[peaks]
        max_x_index = np.argmax(peak_positions)
        mu = peak_positions[max_x_index]
        peak_idx = peaks[max_x_index]
        print(f"Found {len(peaks)} peaks, using rightmost at mu={mu:.3f}")
    
    max_fpkm = density_y[peak_idx]
    
    # Estimate standard deviation using method from Hart et al.
    # CRITICAL: This is the main calculation
    active_samples = fpkm_log2[fpkm_log2 > mu]
    n_active = len(active_samples)
    n_total = len(fpkm_log2)
    active_fraction = n_active / n_total
    
    print(f"Active samples: {n_active}/{n_total} ({active_fraction:.2%})")
    
    # Use Hart et al. method only if we have enough active samples
    # Otherwise fall back to sample standard deviation
    min_active_samples = max(5, int(0.1 * n_total))  # At least 5 or 10% of samples
    
    if n_active >= min_active_samples and active_fraction >= 0.05:  # At least 5% active
        U = np.mean(active_samples)
        stdev = (U - mu) * np.sqrt(np.pi / 2)
        method_used = "Hart_et_al"
        print(f"Using Hart et al. method: mu={mu:.3f}, U={U:.3f}, stdev={stdev:.3f}")
    else:
        # Fall back to sample standard deviation
        stdev = np.std(fpkm_log2, ddof=1)  # Use sample std dev (N-1)
        method_used = "sample_std"
        print(f"Falling back to sample std dev: stdev={stdev:.3f} (too few active: {n_active})")
    
    # Handle edge case where stdev might still be 0 or negative
    if stdev <= 0:
        stdev = 0.1  # Minimal fallback
        method_used = "minimal_fallback"
        warnings.warn(f"Standard deviation calculation resulted in non-positive value. Using minimal fallback: {stdev}")
    
    # Compute zFPKM transform
    z_fpkm = (fpkm_log2 - mu) / stdev
    
    print(f"Method: {method_used}, zFPKM range: {z_fpkm.min():.3f} to {z_fpkm.max():.3f}")
    
    return ZFPKMResult(z_fpkm, x_eval, density_y, mu, stdev, max_fpkm)

def zfpkm_transform(fpkm_df, peak_parameters=None):
    """
    Helper function to transform entire DataFrame
    
    Parameters:
    fpkm_df: pandas DataFrame with genes as rows, samples as columns
    peak_parameters: dict with peak detection parameters
    
    Returns:
    tuple: (results_dict, zfpkm_dataframe)
    """
    if peak_parameters is None:
        peak_parameters = {'minpeakheight': 0.02, 'minpeakdistance': 1}
    
    # Remove problematic rows
    fpkm_df = remove_nan_inf_rows(fpkm_df)
    
    zfpkm_df = pd.DataFrame(index=fpkm_df.index)
    results = {}
    
    for col in fpkm_df.columns:
        result = zfpkm_calc(fpkm_df[col], peak_parameters)
        zfpkm_df[col] = result.z
        results[col] = result
    
    return results, zfpkm_df

def zfpkm(fpkm_df, peak_parameters=None):
    """
    Main zFPKM transformation function
    
    Parameters:
    fpkm_df: pandas DataFrame containing raw FPKM values
             Rows = genes/transcripts, Columns = samples
    peak_parameters: dict with 'minpeakheight' and 'minpeakdistance' keys
                    Default: {'minpeakheight': 0.02, 'minpeakdistance': 1}
    
    Returns:
    pandas DataFrame with zFPKM transformed values
    """
    if peak_parameters is None:
        peak_parameters = {'minpeakheight': 0.02, 'minpeakdistance': 1}
    
    print(f"Peak identification parameters: minpeakheight = {peak_parameters['minpeakheight']}, "
          f"minpeakdistance = {peak_parameters['minpeakdistance']}")
    
    _, zfpkm_df = zfpkm_transform(fpkm_df, peak_parameters)
    return zfpkm_df

def generate_simple_test_data(n_genes: int = 200, n_samples: int = 50) -> pd.DataFrame:
    
    """Generate a single simple test dataset for basic validation"""
    np.random.seed(42)
    
    fpkm_data = np.zeros((n_genes, n_samples))
    
    for gene_idx in range(n_genes):
        active_fraction = np.random.uniform(0.1, 0.9)  # 10-90% of samples active
        expression_delta = np.random.gamma(shape=2, scale=0.5)
        noise_level = np.random.gamma(shape=2, scale=0.5)
        
        n_active_samples = int(active_fraction * n_samples)
        base_log_expr = np.random.normal(-1, 0.5)  # Slightly lower baseline
        
        log_expr = np.random.normal(base_log_expr, noise_level, n_samples)
        
        if n_active_samples > 0:
            active_samples = np.random.choice(n_samples, n_active_samples, replace=False)
            log_expr[active_samples] += expression_delta
        
        fpkm_data[gene_idx, :] = np.power(2, log_expr)
    
    gene_names = [f"ENSG{i:05d}" for i in range(n_genes)]
    sample_names = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    
    return pd.DataFrame(fpkm_data, index=gene_names, columns=sample_names)