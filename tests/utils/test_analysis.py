"""
Tests for data analysis utilities.
Validates statistical analysis, error estimation, and data processing functions.
"""

import numpy as np
import pytest
from src.utils.analysis import (
    QuantumObservable,
    ErrorAnalysis,
    DataProcessing,
    StatisticalResult
)

def test_quantum_expectation_stats():
    """Test statistical analysis of quantum measurements"""
    # Generate fake measurement data
    np.random.seed(42)
    measurements = np.random.normal(loc=1.0, scale=0.1, size=1000)
    
    # Calculate statistics
    stats = QuantumObservable.expectation_stats(measurements)
    
    # Test mean is close to true value
    assert np.abs(stats.mean - 1.0) < 0.01
    
    # Test standard error
    assert stats.stderr == pytest.approx(stats.std / np.sqrt(len(measurements)))
    
    # Test confidence interval contains true mean
    assert stats.confidence_interval[0] < 1.0 < stats.confidence_interval[1]

def test_correlation_time():
    """Test autocorrelation time calculation"""
    # Generate correlated data
    np.random.seed(42)
    n = 1000
    true_tau = 10
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i-1] + 0.1 * np.random.randn()
    
    tau = QuantumObservable.correlation_time(x)
    
    # Test correlation time is reasonable
    assert 5 < tau < 20  # Should be around true_tau=10

def test_bootstrap_error():
    """Test bootstrap error estimation"""
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(loc=2.0, scale=0.5, size=100)
    
    # Define simple statistic
    def mean_statistic(x):
        return np.mean(x)
    
    # Calculate bootstrap error
    result = ErrorAnalysis.bootstrap_error(data, mean_statistic)
    
    # Test mean is close to true value
    assert np.abs(result.mean - 2.0) < 0.1
    
    # Test confidence interval
    assert result.confidence_interval[0] < 2.0 < result.confidence_interval[1]

def test_jackknife_error():
    """Test jackknife error estimation"""
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(loc=2.0, scale=0.5, size=100)
    
    # Define simple statistic
    def mean_statistic(x):
        return np.mean(x)
    
    # Calculate jackknife error
    result = ErrorAnalysis.jackknife_error(data, mean_statistic)
    
    # Test mean is close to true value
    assert np.abs(result.mean - 2.0) < 0.1
    
    # Test error estimation
    assert result.stderr > 0
    assert result.confidence_interval[0] < 2.0 < result.confidence_interval[1]

def test_remove_burnin():
    """Test burn-in removal"""
    # Create data with clear burn-in
    n = 1000
    data = np.ones(n)
    data[:100] = np.linspace(0, 1, 100)  # Burn-in period
    
    # Remove burn-in
    cleaned = DataProcessing.remove_burnin(data, fraction=0.15)
    
    # Test length
    assert len(cleaned) == int(n * 0.85)
    
    # Test all values are equilibrated
    assert np.allclose(cleaned, 1.0)

def test_blocking_analysis():
    """Test blocking analysis for error estimation"""
    # Generate correlated data
    np.random.seed(42)
    n = 1024
    data = np.zeros(n)
    for i in range(1, n):
        data[i] = 0.9 * data[i-1] + 0.1 * np.random.randn()
    
    # Perform blocking analysis
    results = DataProcessing.blocking_analysis(data)
    
    # Test results
    assert isinstance(results, dict)
    assert len(results) > 0
    assert all(isinstance(k, int) for k in results.keys())
    assert all(isinstance(v, float) for v in results.values())
    
    # Test block sizes are powers of 2
    assert all(k & (k-1) == 0 for k in results.keys())  # Test if power of 2
    
    # Test error estimates are positive and finite
    assert all(v > 0 and np.isfinite(v) for v in results.values())

if __name__ == '__main__':
    pytest.main([__file__])