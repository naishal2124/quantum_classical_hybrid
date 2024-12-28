"""
Tests for Schrödinger equation solver.
Validates eigenvalue solutions and expectation values.
"""

import numpy as np
import pytest
from scipy import sparse
from src.quantum.schrodinger import SchrodingerSolver
from src.quantum.potentials import HarmonicOscillator

def test_harmonic_oscillator_ground_state():
    """Test ground state energy of harmonic oscillator"""
    config = {
        'n_points': 2000,  # Increased points for better accuracy
        'x_min': -10.0,
        'x_max': 10.0,
        'mass': 1.0,
        'potential': HarmonicOscillator(),
        'hbar': 1.0
    }
    
    solver = SchrodingerSolver(config)
    E0, psi0 = solver.solve_ground_state()
    
    # Ground state energy should be ℏω/2 = 0.5 for m=ω=ℏ=1
    assert np.abs(E0 - 0.5) < 1e-4  # Relaxed tolerance
    
    # Check normalization
    norm = np.sum(np.abs(psi0)**2) * solver.dx
    assert np.abs(norm - 1.0) < 1e-4

def test_excited_states():
    """Test first few excited states of harmonic oscillator"""
    config = {
        'n_points': 2000,
        'x_min': -10.0,
        'x_max': 10.0,
        'mass': 1.0,
        'potential': HarmonicOscillator(),
        'hbar': 1.0
    }
    
    solver = SchrodingerSolver(config)
    energies, states = solver.solve_n_states(4)
    
    # Energy levels should be (n + 1/2)ℏω
    expected_energies = np.array([0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(energies, expected_energies, rtol=1e-4)
    
    # Check orthonormality
    for i in range(4):
        for j in range(4):
            overlap = np.sum(np.conj(states[:, i]) * states[:, j]) * solver.dx
            expected = 1.0 if i == j else 0.0
            assert np.abs(overlap - expected) < 1e-4

def test_expectation_values():
    """Test expectation value calculations"""
    config = {
        'n_points': 2000,
        'x_min': -10.0,
        'x_max': 10.0,
        'mass': 1.0,
        'potential': HarmonicOscillator(),
        'hbar': 1.0
    }
    
    solver = SchrodingerSolver(config)
    E0, psi0 = solver.solve_ground_state()
    
    # Ground state should be symmetric, so ⟨x⟩ = ⟨p⟩ = 0
    assert np.abs(solver.position_expectation(psi0)) < 1e-4
    assert np.abs(solver.momentum_expectation(psi0)) < 1e-4
    
    # Energy expectation should match eigenvalue
    energy_expect = solver.energy_expectation(psi0)
    assert np.abs(energy_expect - 0.5) < 1e-4