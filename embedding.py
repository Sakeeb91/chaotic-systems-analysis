import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def time_delay_embedding(time_series, embedding_dim, delay):
    """
    Perform time-delay embedding on a scalar time series.
    
    Parameters:
        time_series (ndarray): The input scalar time series
        embedding_dim (int): The embedding dimension
        delay (int): The time delay in indices
        
    Returns:
        ndarray: Embedded vectors, shape (n_points, embedding_dim)
    """
    if embedding_dim < 1 or delay < 1:
        raise ValueError("Embedding dimension and delay must be positive integers")
    
    n = len(time_series)
    max_index = n - (embedding_dim - 1) * delay
    
    if max_index <= 0:
        raise ValueError("Time series too short for the given embedding parameters")
    
    embedded = np.zeros((max_index, embedding_dim))
    
    for i in range(max_index):
        for j in range(embedding_dim):
            embedded[i, j] = time_series[i + j * delay]
            
    return embedded

def mutual_information(x, y, bins=10):
    """
    Calculate the mutual information between two variables.
    
    Parameters:
        x, y (ndarray): Input arrays
        bins (int): Number of bins for histogram
        
    Returns:
        float: Mutual information value
    """
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hist_2d = hist_2d / np.sum(hist_2d)
    
    # Calculate marginal distributions
    p_x = np.sum(hist_2d, axis=1)
    p_y = np.sum(hist_2d, axis=0)
    
    # Construct the product distribution
    p_xy = np.outer(p_x, p_y)
    
    # Calculate mutual information (avoiding log(0))
    mask = hist_2d > 0
    mi = np.sum(hist_2d[mask] * np.log(hist_2d[mask] / p_xy[mask]))
    
    return mi

def find_optimal_delay(time_series, max_delay=100):
    """
    Find the optimal embedding delay using the mutual information method.
    
    Parameters:
        time_series (ndarray): Input scalar time series
        max_delay (int): Maximum delay to consider
        
    Returns:
        int: Optimal delay
    """
    mi_values = np.zeros(max_delay)
    
    for delay in range(1, max_delay + 1):
        # Take two copies of the time series with increasing delay
        x = time_series[:-delay] if delay > 0 else time_series
        y = time_series[delay:] if delay > 0 else time_series
        
        mi_values[delay-1] = mutual_information(x, y)
    
    # Find the first minimum of mutual information
    # (Skipping the first few points to avoid noise)
    skip = 5
    for i in range(skip, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i + 1
    
    # If no clear minimum, return a reasonable default
    return max(1, max_delay // 10)

def correlation_sum(embedded_data, epsilons):
    """
    Calculate the correlation sum for various epsilon values.
    
    Parameters:
        embedded_data (ndarray): The embedded data, shape (n_points, embedding_dim)
        epsilons (ndarray): Array of epsilon values to test
        
    Returns:
        ndarray: Correlation sum values for each epsilon
    """
    n_points = embedded_data.shape[0]
    
    # Calculate pairwise distances
    distances = pdist(embedded_data)
    
    # Calculate correlation sum for each epsilon
    correlation_sums = np.zeros_like(epsilons)
    
    for i, epsilon in enumerate(epsilons):
        # Count pairs with distance < epsilon
        count = np.sum(distances < epsilon)
        
        # Normalize by the total number of pairs
        correlation_sums[i] = 2.0 * count / (n_points * (n_points - 1))
    
    return correlation_sums

def estimate_correlation_dimension(embedded_data, min_scale=None, max_scale=None, 
                                  num_scales=20, scaling_region=None):
    """
    Estimate the correlation dimension of an attractor.
    
    Parameters:
        embedded_data (ndarray): The embedded data, shape (n_points, embedding_dim)
        min_scale (float): Minimum epsilon value (if None, estimated from data)
        max_scale (float): Maximum epsilon value (if None, estimated from data)
        num_scales (int): Number of epsilon values to test
        scaling_region (tuple): Log-scale range to use for linear fit (if None, estimated)
        
    Returns:
        dict: Results including dimension estimate, log-log data, and scaling region
    """
    # Estimate reasonable epsilon range if not provided
    if min_scale is None or max_scale is None:
        distances = pdist(embedded_data)
        if min_scale is None:
            min_scale = np.percentile(distances, 1)  # 1st percentile
            # Ensure min_scale is not too small to avoid log(0) issues
            min_scale = max(min_scale, 1e-10)
        if max_scale is None:
            max_scale = np.percentile(distances, 90)  # 90th percentile
    
    # Generate logarithmically spaced epsilon values
    epsilons = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    # Calculate correlation sum
    c_values = correlation_sum(embedded_data, epsilons)
    
    # Convert to log scale, avoiding log(0)
    log_eps = np.log10(epsilons)
    # Add a small value to avoid log(0)
    c_values_safe = np.maximum(c_values, 1e-10)
    log_c = np.log10(c_values_safe)
    
    # Identify scaling region for linear fit if not provided
    if scaling_region is None:
        # Use middle 60% of the range as default scaling region
        lower_idx = int(0.2 * len(log_eps))
        upper_idx = int(0.8 * len(log_eps))
        scaling_region = (lower_idx, upper_idx)
    else:
        lower_idx, upper_idx = scaling_region
    
    # Linear fit to estimate dimension
    scaling_log_eps = log_eps[lower_idx:upper_idx]
    scaling_log_c = log_c[lower_idx:upper_idx]
    
    # Perform linear regression
    slope, intercept = np.polyfit(scaling_log_eps, scaling_log_c, 1)
    
    return {
        'dimension': slope,
        'log_eps': log_eps,
        'log_c': log_c,
        'scaling_region': (lower_idx, upper_idx),
        'scaling_log_eps': scaling_log_eps,
        'scaling_log_c': scaling_log_c,
        'fit_intercept': intercept,
        'fit_slope': slope
    } 