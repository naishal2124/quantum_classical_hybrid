"""
Tests for visualization utilities.
Validates plotting functions for quantum and statistical visualizations.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from src.utils.visualization import (
    QuantumVisualizer,
    StatisticalVisualizer,
    PlotStyle,
    configure_plotting
)
from src.quantum.wavefunctions import GaussianWavepacket
from src.quantum.potentials import HarmonicOscillator
from src.utils.analysis import StatisticalResult

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for each test"""
    plt.style.use('default')  # Reset to default style
    yield
    plt.close('all')  # Close all figures after each test

@pytest.fixture
def quantum_visualizer():
    """Fixture for quantum visualization tests"""
    style = PlotStyle(figsize=(8, 6), dpi=100)
    return QuantumVisualizer(style)

@pytest.fixture
def statistical_visualizer():
    """Fixture for statistical visualization tests"""
    style = PlotStyle(figsize=(8, 6), dpi=100)
    return StatisticalVisualizer(style)

def test_wavefunction_plotting(quantum_visualizer):
    """Test wavefunction plotting functionality"""
    # Create test wavefunction
    N = 1000
    dx = 0.01
    psi = GaussianWavepacket(N, dx)
    
    # Create test potential
    potential = HarmonicOscillator()
    
    # Test plotting with various options
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    
    # Test probability plotting
    ax1 = quantum_visualizer.plot_wavefunction(psi, potential=potential,
                                             show_probability=True, ax=axes[0])
    assert ax1 is axes[0]
    assert len(ax1.lines) >= 2  # At least probability and potential
    
    # Test wavefunction plotting
    ax2 = quantum_visualizer.plot_wavefunction(psi, potential=potential,
                                             show_probability=False, ax=axes[1])
    assert ax2 is axes[1]
    assert len(ax2.lines) >= 3  # Real, imaginary, and potential

def test_density_matrix_plotting(quantum_visualizer):
    """Test density matrix visualization"""
    # Create test density matrix
    N = 50
    x = np.linspace(-5, 5, N)
    rho = np.outer(np.exp(-x**2), np.exp(-x**2))  # Simple Gaussian density
    
    # Test plotting
    fig, ax = plt.subplots()
    ax = quantum_visualizer.plot_density_matrix(rho, x, ax=ax)
    
    assert len(ax.images) == 1  # Should have one imshow plot
    assert ax.get_xlabel() == 'x'
    assert ax.get_ylabel() == "x'"

def test_correlation_function_plotting(statistical_visualizer):
    """Test correlation function plotting"""
    # Create test data
    times = np.linspace(0, 10, 100)
    correlation = np.exp(-times/2)  # Simple exponential decay
    
    # Test plotting
    fig, ax = plt.subplots()
    ax = statistical_visualizer.plot_correlation_function(times, correlation, ax=ax)
    
    assert len(ax.lines) == 2  # Correlation and e⁻¹ line
    assert ax.get_xlabel() == 'Time'
    assert ax.get_ylabel() == 'Correlation'

def test_blocking_analysis_plotting(statistical_visualizer):
    """Test blocking analysis visualization"""
    # Create test blocking data
    block_sizes = 2**np.arange(8)
    errors = 0.1 * np.sqrt(block_sizes)  # Simple scaling
    block_data = dict(zip(block_sizes, errors))
    
    # Test plotting
    fig, ax = plt.subplots()
    ax = statistical_visualizer.plot_blocking_analysis(block_data, ax=ax)
    
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == 'Block Size'
    assert ax.get_ylabel() == 'Statistical Error'

def test_error_distribution_plotting(statistical_visualizer):
    """Test error distribution visualization"""
    # Create test data
    np.random.seed(42)
    data = np.random.normal(loc=1.0, scale=0.1, size=1000)
    
    # Create statistical result
    mean = np.mean(data)
    std = np.std(data)
    stderr = std / np.sqrt(len(data))
    ci = (mean - 2*stderr, mean + 2*stderr)
    stat_result = StatisticalResult(mean, std, stderr, ci)
    
    # Test plotting
    fig, ax = plt.subplots()
    ax = statistical_visualizer.plot_error_distribution(stat_result, data, ax=ax)
    
    assert len(ax.lines) >= 3  # Normal fit and CI lines
    assert ax.get_xlabel() == 'Value'
    assert ax.get_ylabel() == 'Density'

def test_plot_style_configuration():
    """Test plot style configuration"""
    # Test default style
    style = PlotStyle()
    assert style.figsize == (10, 6)
    assert style.dpi == 100
    assert style.cmap == 'viridis'
    
    # Test custom style
    custom_style = PlotStyle(figsize=(12, 8), dpi=150, cmap='plasma')
    assert custom_style.figsize == (12, 8)
    assert custom_style.dpi == 150
    assert custom_style.cmap == 'plasma'
    
    # Test global configuration
    configure_plotting(style='default', context='paper', figsize=(8, 6))
    expected_figsize = [8.0, 6.0]  # matplotlib stores as list of floats
    assert pytest.approx(plt.rcParams['figure.figsize']) == expected_figsize

if __name__ == '__main__':
    pytest.main([__file__])