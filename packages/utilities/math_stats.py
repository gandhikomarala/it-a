"""Statistical and mathematical functions for data validation, drift, and metrics."""
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

def compute_percentiles(values: List[float], percentiles: Optional[List[float]] = None) -> Dict[str, float]:
    """Calculate accurate quantile values for a list of numerical floats."""
    if not values:
        return {}
    if percentiles is None:
        percentiles = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
    
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {}
        
    results = {}
    for p in percentiles:
        results[f"p{int(p)}"] = float(np.percentile(arr, p))
    return results

def compute_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Compute Population Stability Index (PSI) between baseline and current distributions."""
    b = baseline[~np.isnan(baseline)]
    c = current[~np.isnan(current)]
    
    if len(b) == 0 or len(c) == 0:
        return 0.0

    # Determine quantile bins based on baseline
    percentile_cutoffs = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(b, percentile_cutoffs)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate bucket counts
    b_counts, _ = np.histogram(b, bins=bins)
    c_counts, _ = np.histogram(c, bins=bins)

    # Convert to fractions with Laplace epsilon smoothing
    eps = 1e-4
    b_fractions = (b_counts + eps) / (len(b) + eps * num_buckets)
    c_fractions = (c_counts + eps) / (len(c) + eps * num_buckets)

    psi_value = np.sum((c_fractions - b_fractions) * np.log(c_fractions / b_fractions))
    return float(np.maximum(0.0, psi_value))

def compute_ks_test(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Compute Two-Sample Kolmogorov-Smirnov test (statistic, p-value)."""
    from scipy.stats import ks_2samp
    b = baseline[~np.isnan(baseline)]
    c = current[~np.isnan(current)]
    if len(b) == 0 or len(c) == 0:
        return 0.0, 1.0
    res = ks_2samp(b, c)
    return float(res.statistic), float(res.pvalue)

def compute_entropy(labels: List[Any]) -> float:
    """Compute Shannon entropy for categorical distribution."""
    if not labels:
        return 0.0
    counts = {}
    for item in labels:
        counts[item] = counts.get(item, 0) + 1
    total = len(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score: mean squared error of predicted probabilities."""
    return float(np.mean((y_prob - y_true) ** 2))
