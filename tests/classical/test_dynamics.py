"""
Tests for classical dynamics implementation.
Validates Hamiltonian evolution, integrators, and analysis tools.
"""

import numpy as np
import pytest
from src.classical.dynamics import (
    PhaseSpaceState,
    VelocityVerlet,
    RungeKutta4,
    HamiltonianDynamics
)
from src.quantum.potentials import HarmonicOscillator

@pytest.fixture
def harmonic_system():
    """Fixture for harmonic oscillator system"""
    potential = HarmonicOscillator()
    return HamiltonianDynamics(potential)

def test_phase_space_state():
    """Test phase space state initialization and energy calculation"""
    x = np.array([1.0])
    p = np.array([0.0])
    state = PhaseSpaceState(x, p)
    
    # Test harmonic oscillator energy
    mass = 1.0
    V = lambda x: 0.5 * mass * x**2
    
    energy = state.total_energy(mass, V)
    expected = 0.5  # purely potential energy at x=1, p=0
    assert np.abs(energy - expected) < 1e-10

def test_velocity_verlet():
    """Test velocity Verlet integration"""
    # Setup harmonic oscillator
    dt = 0.1
    mass = 1.0
    integrator = VelocityVerlet(dt, mass)
    
    # Initial state
    x0 = np.array([1.0])
    p0 = np.array([0.0])
    state = PhaseSpaceState(x0, p0)
    
    # Force function for harmonic oscillator
    F = lambda x: -x
    
    # Evolve for one step
    new_state = integrator.step(state, F)
    
    # Energy should be conserved
    E0 = state.total_energy(mass, lambda x: 0.5*x**2)
    E1 = new_state.total_energy(mass, lambda x: 0.5*x**2)
    assert np.abs(E1 - E0) < 1.25e-5

def test_runge_kutta():
    """Test RK4 integration"""
    # Setup harmonic oscillator
    dt = 0.1
    mass = 1.0
    integrator = RungeKutta4(dt, mass)
    
    # Initial state
    x0 = np.array([1.0])
    p0 = np.array([0.0])
    state = PhaseSpaceState(x0, p0)
    
    # Force function for harmonic oscillator
    F = lambda x: -x
    
    # Evolve for one step
    new_state = integrator.step(state, F)
    
    # Energy should be conserved
    E0 = state.total_energy(mass, lambda x: 0.5*x**2)
    E1 = new_state.total_energy(mass, lambda x: 0.5*x**2)
    assert np.abs(E1 - E0) < 1e-6

def test_harmonic_evolution(harmonic_system):
    """Test time evolution of harmonic oscillator"""
    # Initial state
    x0 = np.array([1.0])
    p0 = np.array([0.0])
    state = PhaseSpaceState(x0, p0)
    
    # Evolve system
    n_steps = 100
    trajectory = harmonic_system.evolve(state, n_steps)
    
    # Check energy conservation
    energies = trajectory['energies']
    assert np.all(np.abs(energies - energies[0]) < 1e-4)
    
    # Check trajectory shape
    assert trajectory['positions'].shape == (n_steps + 1, 1)
    assert trajectory['momenta'].shape == (n_steps + 1, 1)
    assert trajectory['times'].shape == (n_steps + 1,)

def test_poincare_section(harmonic_system):
    """Test Poincaré section calculation"""
    # Initial state with non-zero momentum
    x0 = np.array([1.0])
    p0 = np.array([1.0])
    state = PhaseSpaceState(x0, p0)
    
    # Long evolution to get many crossings
    n_steps = 1000
    trajectory = harmonic_system.evolve(state, n_steps)
    
    # Calculate Poincaré section
    pos_section, mom_section = harmonic_system.poincare_section(
        trajectory, threshold=0.0
    )
    
    # Check that points lie close to section
    assert np.all(np.abs(pos_section[:, 0]) < 1e-10)
    
    # Check energy conservation on section
    E0 = 0.5 * (x0**2 + p0**2)  # Initial energy
    for x, p in zip(pos_section, mom_section):
        E = 0.5 * (x**2 + p**2)
        assert np.abs(E - E0) < 1e-4

if __name__ == '__main__':
    pytest.main([__file__])