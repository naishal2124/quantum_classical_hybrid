"""
Tests for quantum mechanical operators.
"""

import numpy as np
import pytest
from scipy import sparse
from src.quantum.operators import (
    Position, Momentum, KineticEnergy, PotentialEnergy, Hamiltonian
)

def test_position_operator():
    """Test position operator properties"""
    N = 100
    dx = 0.1
    x_op = Position(N, dx)
    
    # Test matrix is Hermitian
    matrix = x_op.to_matrix()
    assert sparse.linalg.norm(matrix - matrix.conj().T) < 1e-10
    
    # Test expectation value
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    psi = np.exp(-(x**2)/2) / np.pi**0.25  # Ground state of harmonic oscillator
    assert abs(x_op.expectation_value(psi)) < 1e-10  # Should be zero

def test_momentum_operator():
    """Test momentum operator properties"""
    N = 100
    dx = 0.1
    p_op = Momentum(N, dx)
    
    # Test matrix is Hermitian
    matrix = p_op.to_matrix()
    assert sparse.linalg.norm(matrix - matrix.conj().T) < 1e-10
    
    # Test expectation value
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    psi = np.exp(-(x**2)/2) / np.pi**0.25
    assert abs(p_op.expectation_value(psi)) < 1e-10

def test_kinetic_energy():
    """Test kinetic energy operator"""
    N = 100
    dx = 0.1
    T_op = KineticEnergy(N, dx)
    
    # Test matrix is Hermitian
    matrix = T_op.to_matrix()
    assert sparse.linalg.norm(matrix - matrix.conj().T) < 1e-10
    
    # Test ground state energy of free particle
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    k = 0  # Ground state
    psi = np.exp(1j * k * x) / np.sqrt(N * dx)
    energy = T_op.expectation_value(psi)
    expected = k**2 / 2  # E = ℏ²k²/2m with ℏ=m=1
    assert abs(energy - expected) < 1e-10

def test_harmonic_oscillator():
    """Test complete harmonic oscillator Hamiltonian"""
    N = 100
    dx = 0.1
    
    # Define potential
    def V(x):
        return 0.5 * x**2  # Harmonic oscillator potential
    
    # Create operators
    T = KineticEnergy(N, dx)
    V = PotentialEnergy(V, N, dx)
    H = Hamiltonian(T, V)
    
    # Test ground state energy
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    psi = np.exp(-(x**2)/2) / np.pi**0.25
    energy = H.expectation_value(psi)
    assert abs(energy - 0.5) < 1e-2  # Ground state energy = ℏω/2 = 0.5

if __name__ == '__main__':
    pytest.main([__file__])