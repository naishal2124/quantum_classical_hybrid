"""
Tests for quantum mechanical operators.
"""

import numpy as np
import pytest
from scipy import sparse
from src.quantum.operators import (
    Position, Momentum, KineticEnergy, PotentialEnergy, Hamiltonian
)

def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    """Normalize a wavefunction"""
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

def gaussian_wavepacket(x: np.ndarray, x0: float = 0.0, 
                       k0: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Create a normalized Gaussian wavepacket"""
    dx = x[1] - x[0]
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2) + 1j * k0 * x)
    return normalize(psi, dx)

def harmonic_oscillator_groundstate(x: np.ndarray, mass: float = 1.0, 
                                  omega: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """Create normalized harmonic oscillator ground state"""
    dx = x[1] - x[0]
    alpha = np.sqrt(mass * omega / hbar)
    psi = np.exp(-alpha * x**2 / 2) / np.power(np.pi / alpha, 0.25)
    return normalize(psi, dx)

def test_position_operator():
    """Test position operator properties"""
    # Parameters
    N = 2000  # Increased for better accuracy
    dx = 0.01
    x_op = Position(N, dx)
    
    # Test matrix is Hermitian
    matrix = x_op.to_matrix()
    assert sparse.linalg.norm(matrix - matrix.conj().T) < 1e-10
    
    # Test expectation value with Gaussian centered at x0=1.0
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    sigma = 0.5
    psi = gaussian_wavepacket(x, x0=1.0, sigma=sigma)
    
    # Calculate expectation value
    expect_x = x_op.expectation_value(psi)
    assert abs(expect_x - 1.0) < 1e-3

def test_momentum_operator():
    """Test momentum operator properties"""
    # Parameters
    N = 2000
    dx = 0.01
    p_op = Momentum(N, dx)
    hbar = 1.0
    
    # Create a plane wave with known momentum
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    k0 = 1.0  # Wavevector (momentum = ℏk₀)
    sigma = 0.5  # Width of Gaussian envelope
    psi = gaussian_wavepacket(x, k0=k0, sigma=sigma)
    
    # Calculate momentum expectation value
    expect_p = p_op.expectation_value(psi)
    assert abs(expect_p - hbar*k0) < 1e-2

def test_kinetic_energy():
    """Test kinetic energy operator"""
    # Parameters
    N = 2000
    dx = 0.01
    mass = 1.0
    hbar = 1.0
    T_op = KineticEnergy(N, dx, mass, hbar)
    
    # Test matrix is Hermitian
    matrix = T_op.to_matrix()
    assert sparse.linalg.norm(matrix - matrix.conj().T) < 1e-10
    
    # Test kinetic energy of Gaussian wavepacket
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    k0 = 1.0  # Non-zero momentum
    sigma = 1.0  # Width parameter
    psi = gaussian_wavepacket(x, k0=k0, sigma=sigma)
    
    # Calculate kinetic energy
    energy = T_op.expectation_value(psi)
    
    # Expected energy includes both momentum and spread contributions
    # E = ℏ²k₀²/2m + ℏ²/8mσ² (zero-point energy)
    expected = 0.5 * (hbar*k0)**2 / mass + hbar**2 / (8 * mass * sigma**2)
    assert abs(energy - expected) < 1e-2

def test_harmonic_oscillator():
    """Test complete harmonic oscillator Hamiltonian"""
    # Parameters
    N = 2000
    dx = 0.01
    mass = 1.0
    hbar = 1.0
    omega = 1.0
    
    def V(x):
        """Harmonic oscillator potential"""
        return 0.5 * mass * omega**2 * x**2
    
    # Create operators
    T = KineticEnergy(N, dx, mass, hbar)
    V = PotentialEnergy(V, N, dx, hbar)
    H = Hamiltonian(T, V)
    
    # Create ground state wavefunction
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    psi = harmonic_oscillator_groundstate(x, mass, omega, hbar)
    
    # Test ground state energy
    energy = H.expectation_value(psi)
    expected = 0.5 * hbar * omega  # Ground state energy
    assert abs(energy - expected) < 1e-2

if __name__ == '__main__':
    pytest.main([__file__])