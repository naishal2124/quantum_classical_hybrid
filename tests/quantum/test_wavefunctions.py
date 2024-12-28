"""
Tests for quantum mechanical wavefunctions.
"""

import numpy as np
import pytest
from src.quantum.wavefunctions import (
    GaussianWavePacket, HarmonicOscillator, DoubleWell
)

def test_gaussian_wavepacket():
    """Test Gaussian wave packet properties"""
    x = np.linspace(-5, 5, 1000)
    psi = GaussianWavePacket(x, x0=0, k0=1, sigma=1.0)
    
    # Test normalization
    prob = psi.probability_density()
    assert abs(np.sum(prob) * (x[1]-x[0]) - 1.0) < 1e-10
    
    # Test expectation values
    x_expect = np.sum(x * prob) * (x[1]-x[0])
    assert abs(x_expect) < 1e-10  # Centered at x=0

def test_harmonic_oscillator():
    """Test harmonic oscillator eigenstates"""
    x = np.linspace(-5, 5, 1000)
    
    # Ground state
    psi0 = HarmonicOscillator(x, n=0)
    E0 = np.sum(x**2 * psi0.probability_density()) * (x[1]-x[0]) / 2
    assert abs(E0 - 0.5) < 1e-2  # Ground state energy = 1/2
    
    # First excited state
    psi1 = HarmonicOscillator(x, n=1)
    # Test orthogonality
    overlap = np.sum(np.conj(psi0()) * psi1()) * (x[1]-x[0])
    assert abs(overlap) < 1e-10

def test_double_well():
    """Test double well eigenstates"""
    x = np.linspace(-3, 3, 1000)
    
    # Ground state (symmetric)
    psi0 = DoubleWell(x, n=0)
    prob0 = psi0.probability_density()
    assert abs(prob0[len(x)//2-1] - prob0[len(x)//2+1]) < 1e-10
    
    # First excited state (antisymmetric)
    psi1 = DoubleWell(x, n=1)
    prob1 = psi1.probability_density()
    assert abs(prob1[len(x)//2]) < 1e-10  # Node at x=0

if __name__ == '__main__':
    pytest.main([__file__])