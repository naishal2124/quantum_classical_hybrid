"""
Tests for quantum-classical correspondence implementation.
Validates Ehrenfest dynamics, Wigner distributions, and fidelity measures.
"""

import numpy as np
import pytest
from src.classical.correspondence import (
    EhrenfestDynamics,
    WignerDistribution,
    quantum_classical_fidelity,
    ExpectationValues
)
from src.quantum.potentials import HarmonicOscillator
from src.quantum.wavefunctions import GaussianWavepacket
from src.classical.dynamics import HamiltonianDynamics, PhaseSpaceState

@pytest.fixture
def harmonic_system():
    """Fixture for harmonic oscillator"""
    potential = HarmonicOscillator()
    return HamiltonianDynamics(potential)

@pytest.fixture
def gaussian_state():
    """Fixture for Gaussian wavepacket"""
    N = 1000
    dx = 0.01
    params = {
        "center": 1.0,
        "momentum": 0.0,
        "width": 0.5,
        "mass": 1.0,
        "hbar": 1.0
    }
    return GaussianWavepacket(N, dx, params)

def test_ehrenfest_expectation_values(harmonic_system, gaussian_state):
    """Test computation of quantum expectation values"""
    ehrenfest = EhrenfestDynamics(harmonic_system)
    expect = ehrenfest.compute_expectation_values(gaussian_state)
    
    # Test position expectation
    assert np.abs(expect.position - gaussian_state.params.center) < 1e-3
    
    # Test momentum expectation
    assert np.abs(expect.momentum - gaussian_state.params.momentum) < 1e-3
    
    # Test uncertainty relation
    assert expect.uncertainty_product >= 0.5  # ℏ/2

def test_classical_trajectory(harmonic_system, gaussian_state):
    """Test classical trajectory from quantum initial state"""
    ehrenfest = EhrenfestDynamics(harmonic_system)
    
    # Generate trajectory
    n_steps = 100
    trajectory = ehrenfest.classical_trajectory_from_quantum(
        gaussian_state, n_steps
    )
    
    # Check trajectory properties
    assert 'positions' in trajectory
    assert 'momenta' in trajectory
    assert 'energies' in trajectory
    assert 'times' in trajectory
    
    # Check energy conservation
    energies = trajectory['energies']
    assert np.all(np.abs(energies - energies[0]) < 1e-4)

def test_wigner_distribution(gaussian_state):
    """Test Wigner distribution calculation"""
    wigner = WignerDistribution(n_grid=50)  # Use smaller grid for testing
    
    # Compute distribution
    X, P, W = wigner.compute(gaussian_state)
    
    # Check grid dimensions
    assert X.shape == (50, 50)
    assert P.shape == (50, 50)
    assert W.shape == (50, 50)
    
    # Check normalization
    dx = X[0, 1] - X[0, 0]
    dp = P[1, 0] - P[0, 0]
    norm = np.sum(W) * dx * dp
    assert np.abs(norm - 1.0) < 1e-6
    
    # Check that distribution is real-valued
    assert np.all(np.isreal(W))
    
    # Find maximum and its location
    i_max, j_max = np.unravel_index(np.argmax(W), W.shape)
    x_max = X[i_max, j_max]
    p_max = P[i_max, j_max]
    
    # For Gaussian state centered at x=1.0, p=0.0
    assert np.abs(x_max - gaussian_state.params.center) < 0.3
    assert np.abs(p_max - gaussian_state.params.momentum) < 0.3

def test_classical_limit(gaussian_state):
    """Test classical limit of Wigner distribution"""
    wigner = WignerDistribution(n_grid=50)
    
    # Compute distributions for different ℏ
    X, P, W1 = wigner.compute(gaussian_state)
    X, P, W2 = wigner.classical_limit(gaussian_state, hbar_scale=0.1)
    
    # Classical limit should be more localized
    assert np.max(W2) > np.max(W1)
    
    # Both distributions should be normalized
    dx = X[0, 1] - X[0, 0]
    dp = P[1, 0] - P[0, 0]
    norm1 = np.sum(W1) * dx * dp
    norm2 = np.sum(W2) * dx * dp
    assert np.abs(norm1 - 1.0) < 1e-6
    assert np.abs(norm2 - 1.0) < 1e-6

def test_quantum_classical_fidelity(gaussian_state):
    """Test fidelity between quantum and classical states"""
    # Classical state at wavepacket center
    classical_state = PhaseSpaceState(
        position=np.array([gaussian_state.params.center]),
        momentum=np.array([gaussian_state.params.momentum])
    )
    
    # Compute fidelity
    sigma_x = gaussian_state.params.width
    sigma_p = gaussian_state.params.hbar / (2 * sigma_x)  # Minimum uncertainty
    
    fidelity = quantum_classical_fidelity(
        gaussian_state, classical_state, sigma_x, sigma_p
    )
    
    # Fidelity should be high for matching states
    assert fidelity > 0.99

if __name__ == '__main__':
    pytest.main([__file__])