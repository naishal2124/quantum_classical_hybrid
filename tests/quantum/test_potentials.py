"""
Tests for potential energy functions.
Validates potential energies, gradients, and laplacians against known values.
"""

import numpy as np
import pytest
from src.quantum.potentials import (
    HarmonicOscillator, 
    AnharmonicOscillator, 
    DoubleWell, 
    PotentialParams
)

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

def test_anharmonic_oscillator():
    """Test anharmonic oscillator potential"""
    params = {
        "mass": 1.0, 
        "omega": 1.0, 
        "lambda_quartic": 0.1
    }
    ao = AnharmonicOscillator(params)
    x = np.array([-1.0, 0.0, 1.0])
    
    # Test potential
    expected_v = 0.5 * x**2 + 0.1 * x**4
    np.testing.assert_allclose(ao(x), expected_v)
    
    # Test gradient
    expected_grad = x + 0.4 * x**3
    np.testing.assert_allclose(ao.gradient(x), expected_grad)
    
    # Test laplacian
    expected_lap = 1 + 1.2 * x**2
    np.testing.assert_allclose(ao.laplacian(x), expected_lap)

def test_double_well():
    """Test double well potential"""
    params = {
        "well_depth": 1.0,
        "barrier_height": 0.25
    }
    dw = DoubleWell(params)
    x = np.array([-2.0, 0.0, 2.0])
    
    # Test potential
    expected_v = -params["well_depth"] * x**2 + params["barrier_height"] * x**4
    np.testing.assert_allclose(dw(x), expected_v)
    
    # Test gradient
    expected_grad = -2 * params["well_depth"] * x + 4 * params["barrier_height"] * x**3
    np.testing.assert_allclose(dw.gradient(x), expected_grad)
    
    # Test physical properties
    x_test = np.linspace(-3, 3, 100)
    v_test = dw(x_test)
    
    # Check that potential has two minima
    assert len(np.where(np.diff(np.sign(dw.gradient(x_test))))[0]) == 3

def test_potential_symmetry():
    """Test symmetry properties of potentials"""
    x = np.linspace(-5, 5, 100)
    
    # Test harmonic oscillator symmetry
    ho = HarmonicOscillator()
    np.testing.assert_allclose(ho(x), ho(-x))
    
    # Test anharmonic oscillator symmetry
    ao = AnharmonicOscillator({"lambda_quartic": 0.1})
    np.testing.assert_allclose(ao(x), ao(-x))
    
    # Test double well symmetry
    dw = DoubleWell({"well_depth": 1.0, "barrier_height": 0.25})
    np.testing.assert_allclose(dw(x), dw(-x))