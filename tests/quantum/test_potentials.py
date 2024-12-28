"""
Tests for potential energy functions.
Validates potential energies, gradients, and laplacians against known values.
"""

import numpy as np
import pytest
from src.quantum.potentials import HarmonicOscillator, PotentialParams

def test_harmonic_oscillator():
    """Test harmonic oscillator potential with known values"""
    # Setup
    params = {"mass": 1.0, "omega": 1.0}
    ho = HarmonicOscillator(params)
    x = np.array([-1.0, 0.0, 1.0])
    
    # Test potential
    expected_v = np.array([0.5, 0.0, 0.5])
    np.testing.assert_allclose(ho(x), expected_v)
    
    # Test gradient
    expected_grad = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(ho.gradient(x), expected_grad)
    
    # Test laplacian
    expected_lap = np.ones_like(x)
    np.testing.assert_allclose(ho.laplacian(x), expected_lap)

def test_parameter_validation():
    """Test parameter validation and defaults"""
    ho = HarmonicOscillator()  # Should use defaults
    assert ho.params.mass == 1.0
    assert ho.params.omega == 1.0

def test_array_input():
    """Test handling of different array shapes"""
    ho = HarmonicOscillator()
    
    # Test 1D array
    x1 = np.linspace(-1, 1, 10)
    assert ho(x1).shape == x1.shape
    
    # Test 2D array
    x2 = np.array([[-1, 0, 1], [1, 0, -1]])
    assert ho(x2).shape == x2.shape