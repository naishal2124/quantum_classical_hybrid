"""
Tests for Euclidean action implementation.
Validates action calculations and thermodynamic estimators.
"""

import numpy as np
import pytest
from src.pimc.action import EuclideanAction, ActionParameters
from src.quantum.potentials import HarmonicOscillator

@pytest.fixture
def harmonic_action():
    """Fixture for harmonic oscillator action"""
    params = {
        "mass": 1.0,
        "beta": 10.0,
        "n_slices": 32,
        "hbar": 1.0
    }
    potential = HarmonicOscillator({"mass": 1.0, "omega": 1.0})
    return EuclideanAction(potential, params)

@pytest.fixture
def gaussian_path():
    """Generate test path from Gaussian distribution"""
    n_slices = 32
    np.random.seed(42)
    return np.random.normal(loc=0.0, scale=1.0, size=n_slices)

def test_action_parameters():
    """Test action parameter handling"""
    params = ActionParameters(mass=2.0, beta=5.0, n_slices=64)
    assert params.dtau == 5.0/64
    assert params.mass == 2.0
    assert params.hbar == 1.0  # default value

def test_kinetic_action(harmonic_action, gaussian_path):
    """Test kinetic part of action"""
    kinetic = harmonic_action.kinetic(gaussian_path)
    
    # Test shape
    assert kinetic.shape == gaussian_path.shape
    
    # Test positivity
    assert np.all(kinetic >= 0)
    
    # Test periodicity
    path_shifted = np.roll(gaussian_path, 1)
    kinetic_shifted = harmonic_action.kinetic(path_shifted)
    assert np.allclose(kinetic, kinetic_shifted)

def test_potential_action(harmonic_action, gaussian_path):
    """Test potential part of action"""
    potential = harmonic_action.potential_energy(gaussian_path)
    
    # Test shape and positivity
    assert potential.shape == gaussian_path.shape
    assert np.all(potential >= 0)
    
    # Test scaling with dtau
    action2 = EuclideanAction(
        harmonic_action.potential,
        {"beta": 20.0, "n_slices": 64}  # double beta
    )
    potential2 = action2.potential_energy(gaussian_path)
    assert np.allclose(potential2, potential * 2)

def test_total_action(harmonic_action, gaussian_path):
    """Test total action calculation"""
    total = harmonic_action.total(gaussian_path)
    
    # Should be scalar
    assert np.isscalar(total)
    
    # Should be sum of kinetic and potential
    kinetic_sum = np.sum(harmonic_action.kinetic(gaussian_path))
    potential_sum = np.sum(harmonic_action.potential_energy(gaussian_path))
    assert np.isclose(total, kinetic_sum + potential_sum)

def test_local_action(harmonic_action, gaussian_path):
    """Test local action computation"""
    slice_idx = 5
    local = harmonic_action.local_action(gaussian_path, slice_idx)
    
    # Should be scalar and positive for harmonic oscillator
    assert np.isscalar(local)
    assert local >= 0
    
    # Test periodic boundary handling
    local_0 = harmonic_action.local_action(gaussian_path, 0)
    local_n = harmonic_action.local_action(gaussian_path, len(gaussian_path)-1)
    assert np.isfinite(local_0) and np.isfinite(local_n)

def test_force(harmonic_action, gaussian_path):
    """Test force computation"""
    force = harmonic_action.force(gaussian_path)
    
    # Test shape and scaling for harmonic oscillator
    assert force.shape == gaussian_path.shape
    
    # Force should be zero at origin
    x0 = np.zeros_like(gaussian_path)
    f0 = harmonic_action.force(x0)
    assert np.allclose(f0, 0.0)
    
    # Test linear scaling
    force_2x = harmonic_action.force(2 * gaussian_path)
    assert np.allclose(force_2x, 2 * force)

def test_thermodynamic_energy(harmonic_action):
    """Test thermodynamic energy estimator"""
    # Generate multiple paths
    np.random.seed(42)
    n_paths = 1000
    paths = np.random.normal(0, 1, size=(n_paths, harmonic_action.params.n_slices))
    
    # Compute energy
    energy, error = harmonic_action.thermodynamic_energy(paths)
    
    # For harmonic oscillator at low temperature (large beta)
    # Energy should be close to ground state energy = ℏω/2
    assert abs(energy - 0.5) < 0.1
    
    # Error should be positive and finite
    assert error > 0
    assert np.isfinite(error)

def test_harmonic_oscillator_properties(harmonic_action):
    """Test specific properties of harmonic oscillator action"""
    # Test path with constant displacement
    x0 = 1.0
    path = np.full(harmonic_action.params.n_slices, x0)
    
    # Kinetic action should be zero
    kinetic = harmonic_action.kinetic(path)
    assert np.allclose(kinetic, 0.0)
    
    # Potential should be constant
    potential = harmonic_action.potential_energy(path)
    expected_potential = 0.5 * x0**2 * harmonic_action.dtau
    assert np.allclose(potential, expected_potential)

if __name__ == '__main__':
    pytest.main([__file__])