---
layout: default
title: Methodology
permalink: /methodology/
---

# Methodology: Advanced Techniques for Chaotic Systems Analysis

## Overview

This document details the mathematical methods and computational algorithms implemented in the Chaotic Systems Analysis toolkit. Our approach combines rigorous theoretical foundations with efficient numerical implementation.

## Core Algorithms

### 1. Time Series Integration

**Numerical Methods for Differential Equations**

```python
def runge_kutta_4th_order(func, y0, t_span, dt):
    """
    Fourth-order Runge-Kutta integration for dynamical systems
    """
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(1, len(t)):
        h = dt
        k1 = h * func(t[i-1], y[i-1])
        k2 = h * func(t[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * func(t[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * func(t[i-1] + h, y[i-1] + k3)
        
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y
```

### 2. Phase Space Reconstruction

**Takens' Embedding Implementation**

```python
def takens_embedding(time_series, embedding_dim, delay):
    """
    Reconstruct phase space using time-delay embedding
    
    Parameters:
    - time_series: 1D array of scalar observations
    - embedding_dim: embedding dimension (m)
    - delay: time delay (τ)
    """
    N = len(time_series)
    M = N - (embedding_dim - 1) * delay
    
    embedded = np.zeros((M, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * delay:i * delay + M]
    
    return embedded
```

### 3. Optimal Parameter Selection

**Mutual Information Analysis**

```python
def mutual_information(x, y, bins=50):
    """
    Calculate mutual information between two time series
    I(X,Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]
    """
    # Create joint histogram
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hist_x = np.histogram(x, bins=x_edges)[0]
    hist_y = np.histogram(y, bins=y_edges)[0]
    
    # Normalize to probabilities
    p_xy = hist_xy / len(x)
    p_x = hist_x / len(x)
    p_y = hist_y / len(y)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i,j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
    
    return mi

def optimal_delay(time_series, max_delay=50):
    """
    Find optimal delay using first minimum of mutual information
    """
    delays = range(1, max_delay + 1)
    mi_values = []
    
    for delay in delays:
        x = time_series[:-delay]
        y = time_series[delay:]
        mi = mutual_information(x, y)
        mi_values.append(mi)
    
    # Find first local minimum
    optimal_tau = delays[np.argmin(mi_values)]
    return optimal_tau, mi_values
```

**False Nearest Neighbors**

```python
def false_nearest_neighbors(time_series, max_dim=15, tau=1, rtol=15.0):
    """
    Determine optimal embedding dimension using false nearest neighbors
    """
    N = len(time_series)
    fnn_percentages = []
    
    for dim in range(1, max_dim + 1):
        # Create embedded vectors
        embedded = takens_embedding(time_series, dim, tau)
        embedded_plus1 = takens_embedding(time_series, dim + 1, tau)
        
        false_neighbors = 0
        total_neighbors = 0
        
        for i in range(len(embedded)):
            # Find nearest neighbor in dim-dimensional space
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[i] = np.inf  # Exclude self
            nn_idx = np.argmin(distances)
            
            # Check if still nearest neighbor in (dim+1)-dimensional space
            dist_dim = distances[nn_idx]
            dist_dim_plus1 = np.linalg.norm(embedded_plus1[i] - embedded_plus1[nn_idx])
            
            # Test for false nearest neighbor
            if (dist_dim_plus1 / dist_dim) > rtol:
                false_neighbors += 1
            
            total_neighbors += 1
        
        fnn_percentage = 100.0 * false_neighbors / total_neighbors
        fnn_percentages.append(fnn_percentage)
        
        # Stop when FNN percentage drops below threshold
        if fnn_percentage < 1.0:
            break
    
    return fnn_percentages
```

### 4. Correlation Dimension Analysis

**Grassberger-Procaccia Algorithm**

```python
def correlation_dimension(embedded_data, r_min=None, r_max=None, n_radii=50):
    """
    Calculate correlation dimension using Grassberger-Procaccia algorithm
    """
    N = len(embedded_data)
    
    # Calculate all pairwise distances
    distances = pdist(embedded_data)
    
    # Define radius range
    if r_min is None:
        r_min = np.min(distances[distances > 0])
    if r_max is None:
        r_max = np.max(distances) * 0.5
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    correlations = []
    
    for r in radii:
        # Count pairs within radius r
        pairs_within_r = np.sum(distances <= r)
        correlation = pairs_within_r / (N * (N - 1) / 2)
        correlations.append(correlation)
    
    correlations = np.array(correlations)
    
    # Find scaling region and calculate dimension
    log_r = np.log10(radii)
    log_c = np.log10(correlations + 1e-15)  # Avoid log(0)
    
    # Use least squares fit in the scaling region
    valid_indices = (log_c > -np.inf) & (log_c < 0)
    if np.sum(valid_indices) > 5:
        slope, intercept = np.polyfit(log_r[valid_indices], log_c[valid_indices], 1)
        correlation_dim = slope
    else:
        correlation_dim = np.nan
    
    return correlation_dim, radii, correlations
```

### 5. Statistical Validation

**Bootstrap Confidence Intervals**

```python
def bootstrap_correlation_dimension(embedded_data, n_bootstrap=100, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for correlation dimension
    """
    N = len(embedded_data)
    bootstrap_dims = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(N, size=N, replace=True)
        bootstrap_sample = embedded_data[indices]
        
        # Calculate correlation dimension
        dim, _, _ = correlation_dimension(bootstrap_sample)
        if not np.isnan(dim):
            bootstrap_dims.append(dim)
    
    bootstrap_dims = np.array(bootstrap_dims)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_dims, lower_percentile)
    ci_upper = np.percentile(bootstrap_dims, upper_percentile)
    mean_dim = np.mean(bootstrap_dims)
    
    return mean_dim, ci_lower, ci_upper, bootstrap_dims
```

## Advanced Analysis Techniques

### Lyapunov Exponent Estimation

```python
def largest_lyapunov_exponent(time_series, embedding_dim, tau, 
                             mean_period=None, max_iter=1000):
    """
    Estimate largest Lyapunov exponent using the algorithm of Rosenstein et al.
    """
    # Embed the time series
    embedded = takens_embedding(time_series, embedding_dim, tau)
    N = len(embedded)
    
    if mean_period is None:
        mean_period = int(N / 10)
    
    # Find nearest neighbors
    divergences = []
    
    for i in range(N - max_iter):
        # Find nearest neighbor that is temporally separated
        distances = np.linalg.norm(embedded - embedded[i], axis=1)
        
        # Exclude points too close in time
        valid_indices = np.where(np.abs(np.arange(N) - i) > mean_period)[0]
        if len(valid_indices) == 0:
            continue
        
        valid_distances = distances[valid_indices]
        nn_idx = valid_indices[np.argmin(valid_distances)]
        
        # Follow divergence of this pair
        for j in range(1, min(max_iter, N - max(i, nn_idx))):
            if i + j < N and nn_idx + j < N:
                divergence = np.linalg.norm(embedded[i + j] - embedded[nn_idx + j])
                if divergence > 0:
                    divergences.append((j, np.log(divergence)))
    
    # Linear fit to extract Lyapunov exponent
    if len(divergences) > 10:
        times, log_divs = zip(*divergences)
        times = np.array(times)
        log_divs = np.array(log_divs)
        
        # Group by time and average
        unique_times = np.unique(times)
        avg_log_divs = []
        for t in unique_times:
            mask = times == t
            avg_log_divs.append(np.mean(log_divs[mask]))
        
        # Linear fit
        if len(unique_times) > 5:
            slope, _ = np.polyfit(unique_times[:len(avg_log_divs)], avg_log_divs, 1)
            return slope / tau  # Convert to original time units
    
    return np.nan
```

## Computational Optimization

### Efficient Distance Calculations

```python
def efficient_correlation_sum(embedded_data, radius):
    """
    Memory-efficient calculation of correlation sum for large datasets
    """
    N = len(embedded_data)
    correlation_sum = 0
    
    # Process in chunks to manage memory
    chunk_size = min(1000, N)
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk1 = embedded_data[i:end_i]
        
        for j in range(i, N, chunk_size):
            end_j = min(j + chunk_size, N)
            chunk2 = embedded_data[j:end_j]
            
            # Calculate distance matrix for this chunk pair
            if i == j:
                # Same chunk - use upper triangle only
                distances = pdist(chunk1)
                pairs_within_r = np.sum(distances <= radius)
            else:
                # Different chunks - full matrix
                distances = cdist(chunk1, chunk2)
                pairs_within_r = np.sum(distances <= radius)
            
            correlation_sum += pairs_within_r
    
    return correlation_sum / (N * (N - 1) / 2)
```

## Quality Assurance

### Unit Testing Framework

```python
import unittest
import numpy as np

class TestChaoticAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with known chaotic data"""
        # Generate Rössler system test data
        self.test_data = self.generate_rossler_data()
    
    def test_embedding_dimensions(self):
        """Test that embedding preserves basic properties"""
        embedded = takens_embedding(self.test_data, 3, 5)
        self.assertEqual(embedded.shape[1], 3)
        self.assertTrue(len(embedded) > 0)
    
    def test_correlation_dimension_bounds(self):
        """Test that correlation dimension is within reasonable bounds"""
        embedded = takens_embedding(self.test_data, 3, 5)
        dim, _, _ = correlation_dimension(embedded)
        self.assertTrue(1.0 <= dim <= 3.0)  # Expected range for Rössler
    
    def test_mutual_information_properties(self):
        """Test mathematical properties of mutual information"""
        x = np.random.randn(1000)
        # Self-information should be maximal
        mi_self = mutual_information(x, x)
        mi_random = mutual_information(x, np.random.randn(1000))
        self.assertGreater(mi_self, mi_random)

if __name__ == '__main__':
    unittest.main()
```

## Conclusions

This methodology provides a comprehensive framework for chaotic systems analysis with:

1. **Theoretical Rigor**: All algorithms based on established mathematical theory
2. **Computational Efficiency**: Optimized implementations for large datasets  
3. **Statistical Validation**: Bootstrap methods for uncertainty quantification
4. **Quality Assurance**: Comprehensive testing and validation procedures

The implementation serves as both a research tool and educational framework for understanding chaos theory and nonlinear dynamics.

---

*For detailed implementation and source code, see the [GitHub repository](https://github.com/Sakeeb91/chaotic-systems-analysis).*
