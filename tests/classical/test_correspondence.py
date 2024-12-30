"""
Tests for quantum-classical correspondence implementation.
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
from src.classical.dynamics import VelocityVerlet

@pytest.fixture
def harmonic_system():
    """Fixture for harmonic oscillator"""
    potential = HarmonicOscillator({"mass": 1.0, "omega": 1.0})
    # Use VelocityVerlet with small timestep
    integrator = VelocityVerlet(dt=0.005, mass=1.0)
    return HamiltonianDynamics(potential, integrator=integrator)

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
    
    assert np.abs(expect.position - gaussian_state.params.center) < 1e-6
    assert np.abs(expect.momentum - gaussian_state.params.momentum) < 1e-6
    assert expect.uncertainty_product >= 0.5 - 1e-6  # ℏ/2

def test_classical_trajectory(harmonic_system, gaussian_state):
    """Test classical trajectory from quantum initial state"""
    ehrenfest = EhrenfestDynamics(harmonic_system)
    n_steps = 100
    trajectory = ehrenfest.classical_trajectory_from_quantum(gaussian_state, n_steps)
    
    assert len(trajectory['positions']) == n_steps + 1
    assert len(trajectory['momenta']) == n_steps + 1
    
    # Test energy conservation
    energies = trajectory['energies']
    assert np.all(np.abs(energies - energies[0]) / energies[0] < 1e-5)

def test_wigner_distribution(gaussian_state):
    """Test Wigner distribution calculation"""
    wigner = WignerDistribution(n_grid=50)
    X, P, W = wigner.compute(gaussian_state)
    
    assert X.shape == (50, 50)
    assert P.shape == (50, 50)
    assert W.shape == (50, 50)
    
    # Test normalization
    dx = X[0, 1] - X[0, 0]
    dp = P[1, 0] - P[0, 0]
    norm = np.sum(W) * dx * dp
    assert abs(norm - 1.0) < 1e-5
    
    # Find distribution maximum
    i_max, j_max = np.unravel_index(np.argmax(W), W.shape)
    x_max = X[i_max, j_max]
    p_max = P[i_max, j_max]
    
    # Maximum should be near wavepacket center and momentum
    assert np.abs(x_max - gaussian_state.params.center) < 0.5
    assert np.abs(p_max - gaussian_state.params.momentum) < 0.5

def test_classical_limit(gaussian_state):
    """Test classical limit of Wigner distribution"""
    wigner = WignerDistribution(n_grid=50)
    X1, P1, W1 = wigner.compute(gaussian_state)
    X2, P2, W2 = wigner.classical_limit(gaussian_state, hbar_scale=0.1)
    
    # Calculate distribution widths
    x_width1 = np.sqrt(np.average((X1 - gaussian_state.params.center)**2, weights=np.abs(W1)))
    x_width2 = np.sqrt(np.average((X2 - gaussian_state.params.center)**2, weights=np.abs(W2)))
    
    # Width should scale with √ℏ
    width_ratio = x_width2 / x_width1
    assert abs(width_ratio - np.sqrt(0.1)) < 0.1

def test_quantum_classical_fidelity(gaussian_state):
    """Test fidelity between quantum and classical states"""
    classical_state = PhaseSpaceState(
        position=np.array([gaussian_state.params.center]),
        momentum=np.array([gaussian_state.params.momentum])
    )
    
    sigma_x = gaussian_state.params.width
    sigma_p = gaussian_state.params.hbar / (2 * sigma_x)
    
    fidelity = quantum_classical_fidelity(
        gaussian_state, classical_state, sigma_x, sigma_p
    )
    
    assert fidelity > 0.99

if __name__ == '__main__':
    pytest.main([__file__])