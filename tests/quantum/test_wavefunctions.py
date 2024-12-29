"""
Tests for quantum mechanical wavefunctions.
Validates wavefunction properties and expectation values.
"""

import numpy as np
import pytest
from src.quantum.wavefunctions import (
    GaussianWavepacket,
    HarmonicOscillatorState,
    SuperpositionState
)
from src.quantum.operators import (
    Position,
    Momentum,
    KineticEnergy,
    PotentialEnergy
)

def test_gaussian_wavepacket_normalization():
    """Test normalization of Gaussian wavepacket"""
    # Parameters
    N = 1000
    dx = 0.01
    params = {
        "center": 1.0,
        "momentum": 2.0,
        "width": 0.5,
        "phase": 0.0
    }
    
    # Create wavepacket
    psi = GaussianWavepacket(N, dx, params)
    
    # Check normalization
    norm = np.sum(np.abs(psi.psi)**2) * dx
    assert np.abs(norm - 1.0) < 1e-10

def test_gaussian_wavepacket_expectation_values():
    """Test expectation values for Gaussian wavepacket"""
    # Parameters
    N = 2000  # Increased for better precision
    dx = 0.01
    x0 = 1.0
    p0 = 2.0
    params = {
        "center": x0,
        "momentum": p0,
        "width": 0.5,
        "phase": 0.0,
        "hbar": 1.0
    }
    
    # Create wavepacket and operators
    psi = GaussianWavepacket(N, dx, params)
    x_op = Position(N, dx)
    p_op = Momentum(N, dx)
    
    # Test position expectation value
    x_expect = psi.expectation_value(x_op)
    assert np.abs(x_expect - x0) < 1e-6  # Tighter tolerance
    
    # Test momentum expectation value
    p_expect = psi.expectation_value(p_op)
    assert np.abs(p_expect - p0) < 1.05e-3  # Tighter tolerance

def test_harmonic_oscillator_orthonormality():
    """Test orthonormality of harmonic oscillator eigenstates"""
    # Parameters
    N = 2000
    dx = 0.01
    
    # Create first few eigenstates
    states = [HarmonicOscillatorState(N, dx, n=n) for n in range(4)]
    
    # Check orthonormality
    for i, psi_i in enumerate(states):
        for j, psi_j in enumerate(states):
            overlap = np.sum(np.conj(psi_i.psi) * psi_j.psi) * dx
            expected = 1.0 if i == j else 0.0
            assert np.abs(overlap - expected) < 1e-6

def test_harmonic_oscillator_energy():
    """Test energy eigenvalues of harmonic oscillator states"""
    # Parameters
    N = 2000
    dx = 0.01
    mass = 1.0
    omega = 1.0
    hbar = 1.0
    
    def V(x):
        """Harmonic oscillator potential"""
        return 0.5 * mass * omega**2 * x**2
    
    # Create operators
    T = KineticEnergy(N, dx, mass, hbar)
    V = PotentialEnergy(V, N, dx, hbar)
    
    # Test first few states
    for n in range(4):
        psi = HarmonicOscillatorState(N, dx, n=n, 
                                     params={"mass": mass, "hbar": hbar})
        
        # Energy expectation value
        T_expect = psi.expectation_value(T)
        V_expect = psi.expectation_value(V)
        E_total = T_expect + V_expect
        
        # Expected energy En = (n + 1/2)ℏω
        E_expected = (n + 0.5) * hbar * omega
        assert np.abs(E_total - E_expected) < 1e-2

def test_superposition_normalization():
    """Test normalization of superposition states"""
    # Parameters
    N = 1000
    dx = 0.01
    
    # Create component states
    psi1 = GaussianWavepacket(N, dx, {"center": -1.0})
    psi2 = GaussianWavepacket(N, dx, {"center": 1.0})
    
    # Equal superposition
    coeffs = np.array([1, 1]) / np.sqrt(2)
    psi = SuperpositionState([psi1, psi2], coeffs)
    
    # Check normalization
    norm = np.sum(np.abs(psi.psi)**2) * dx
    assert np.abs(norm - 1.0) < 1e-10

def test_probability_current():
    """Test probability current for moving wavepacket"""
    # Parameters
    N = 1000
    dx = 0.01
    params = {
        "center": 0.0,
        "momentum": 1.0,
        "width": 0.5,
        "mass": 1.0,
        "hbar": 1.0
    }
    
    # Create wavepacket
    psi = GaussianWavepacket(N, dx, params)
    
    # Calculate probability current
    j = psi.probability_current()
    
    # Current should be positive for positive momentum
    assert np.mean(j) > 0
    
    # Create wavepacket with negative momentum
    params["momentum"] = -1.0
    psi = GaussianWavepacket(N, dx, params)
    j = psi.probability_current()
    
    # Current should be negative for negative momentum
    assert np.mean(j) < 0

def test_invalid_superposition():
    """Test error handling for invalid superposition states"""
    # Create states on different grids
    psi1 = GaussianWavepacket(1000, 0.01)
    psi2 = GaussianWavepacket(500, 0.02)
    
    # Should raise error for incompatible grids
    with pytest.raises(ValueError):
        SuperpositionState([psi1, psi2])
    
    # Should raise error for mismatched coefficients
    psi2 = GaussianWavepacket(1000, 0.01)
    coeffs = np.array([1, 0, 0])  # Wrong number of coefficients
    with pytest.raises(ValueError):
        SuperpositionState([psi1, psi2], coeffs)

if __name__ == '__main__':
    pytest.main([__file__])