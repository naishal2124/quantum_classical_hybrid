"""
Data analysis utilities for quantum-classical simulations.
Implements statistical analysis, error estimation, and data processing functions.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from scipy import stats
from dataclasses import dataclass

@dataclass
class StatisticalResult:
    """Container for statistical analysis results"""
    mean: float
    std: float
    stderr: float
    confidence_interval: Tuple[float, float]

class QuantumObservable:
    """Analysis utilities for quantum observables"""
    
    @staticmethod
    def expectation_stats(measurements: np.ndarray, 
                         confidence: float = 0.95) -> StatisticalResult:
        """
        Calculate statistical properties of quantum measurements
        
        Args:
            measurements: Array of measurement results
            confidence: Confidence level for intervals (default: 0.95)
            
        Returns:
            StatisticalResult with mean, std, stderr, and confidence interval
        """
        mean = np.mean(measurements)
        std = np.std(measurements, ddof=1)  # Use N-1 for unbiased estimation
        n = len(measurements)
        stderr = std / np.sqrt(n)
        
        # Calculate confidence interval
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=stderr)
        
        return StatisticalResult(mean, std, stderr, ci)
    
    @staticmethod
    def correlation_time(time_series: np.ndarray) -> float:
        """
        Calculate autocorrelation time of a measurement time series
        
        Args:
            time_series: Array of sequential measurements
            
        Returns:
            Autocorrelation time τ
        """
        mean = np.mean(time_series)
        fluctuations = time_series - mean
        
        # Compute autocorrelation function
        acf = np.correlate(fluctuations, fluctuations, mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        
        # Find first crossing of e^(-1)
        tau = np.where(acf < np.exp(-1))[0]
        if len(tau) > 0:
            return float(tau[0])
        return 1.0  # Default if no crossing found

class ErrorAnalysis:
    """Error estimation and uncertainty propagation"""
    
    @staticmethod
    def bootstrap_error(data: np.ndarray, 
                       statistic: callable,
                       n_resamples: int = 1000,
                       confidence: float = 0.95) -> StatisticalResult:
        """
        Bootstrap error estimation for arbitrary statistics
        
        Args:
            data: Input data array
            statistic: Function to compute on resampled data
            n_resamples: Number of bootstrap resamples
            confidence: Confidence level for intervals
            
        Returns:
            StatisticalResult for the bootstrapped statistic
        """
        n = len(data)
        resamples = np.random.choice(data, size=(n_resamples, n), replace=True)
        bootstrap_stats = np.array([statistic(resample) for resample in resamples])
        
        mean = np.mean(bootstrap_stats)
        std = np.std(bootstrap_stats, ddof=1)
        stderr = std / np.sqrt(n_resamples)
        
        # Calculate percentile confidence interval
        alpha = (1 - confidence) / 2
        ci = np.percentile(bootstrap_stats, [100*alpha, 100*(1-alpha)])
        
        return StatisticalResult(mean, std, stderr, tuple(ci))
    
    @staticmethod
    def jackknife_error(data: np.ndarray, 
                       statistic: callable) -> StatisticalResult:
        """
        Jackknife error estimation
        
        Args:
            data: Input data array
            statistic: Function to compute on resampled data
            
        Returns:
            StatisticalResult for the jackknife estimate
        """
        n = len(data)
        jackknife_stats = np.array([
            statistic(np.delete(data, i)) for i in range(n)
        ])
        
        # Jackknife estimate and error
        mean = np.mean(jackknife_stats)
        stderr = np.sqrt((n-1) * np.var(jackknife_stats, ddof=1))
        std = stderr * np.sqrt(n)
        
        # Approximate confidence interval
        ci = (mean - 2*stderr, mean + 2*stderr)  # ~95% CI
        
        return StatisticalResult(mean, std, stderr, ci)

class DataProcessing:
    """Data processing and transformation utilities"""
    
    @staticmethod
    def remove_burnin(data: np.ndarray, 
                     fraction: float = 0.1) -> np.ndarray:
        """
        Remove initial burn-in period from time series
        
        Args:
            data: Input time series
            fraction: Fraction of data to remove (default: 0.1)
            
        Returns:
            Array with burn-in period removed
        """
        n = len(data)
        cutoff = int(n * fraction)
        return data[cutoff:]
    
    @staticmethod
    def blocking_analysis(data: np.ndarray, 
                         max_blocks: Optional[int] = None) -> Dict[int, float]:
        """
        Perform blocking analysis to estimate statistical errors
        
        Args:
            data: Input time series
            max_blocks: Maximum number of blocking transformations
            
        Returns:
            Dictionary mapping block sizes to error estimates
        """
        if max_blocks is None:
            max_blocks = int(np.log2(len(data)))
        
        results = {}
        x = data.copy()
        
        for k in range(max_blocks):
            n = len(x) // 2
            block_size = 2**k
            
            if n < 2:  # Need at least 2 blocks
                break
                
            # Calculate error for this block size
            results[block_size] = np.std(x[:2*n].reshape(n, 2).mean(axis=1)) / np.sqrt(n-1)
            
            # Create blocked data for next iteration
            x = x[:2*n].reshape(n, 2).mean(axis=1)
            
        return results