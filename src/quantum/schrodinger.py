"""
Time-independent and time-dependent Schrödinger equation solvers.
Implements numerical methods for quantum mechanical calculations.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Dict, Optional, Tuple, Union
from .potentials import Potential, HarmonicOscillator

class SchrodingerSolver:
    """
    Solves the time-independent Schrödinger equation: Hψ = Eψ
    Uses sparse matrices for efficient computation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the solver with configuration parameters.
        """
        self.config = config
        self.hbar = config.get('hbar', 1.0)
        self.mass = config.get('mass', 1.0)
        self.potential = config.get('potential', HarmonicOscillator())
        
        # Set up spatial grid
        self.n_points = config['n_points']
        self.x_min = config['x_min']
        self.x_max = config['x_max']
        self.setup_grid()
        
        # Initialize Hamiltonian
        self.setup_hamiltonian()
    
    def setup_grid(self):
        """Initialize spatial and momentum grids"""
        # Position space grid
        self.dx = (self.x_max - self.x_min) / (self.n_points - 1)
        self.x = np.linspace(self.x_min, self.x_max, self.n_points)
        
        # Momentum space grid for kinetic energy calculation
        self.k = 2 * np.pi * np.fft.fftfreq(self.n_points, self.dx)
    
    def setup_hamiltonian(self):
        """
        Construct the Hamiltonian matrix using kinetic and potential terms
        """
        # Kinetic energy matrix using finite difference
        diagonals = []
        offsets = []
        
        # Second derivative approximation
        dx2 = self.dx ** 2
        coeff = -self.hbar**2 / (2.0 * self.mass * dx2)
        
        # Main diagonal
        diagonals.append(np.ones(self.n_points) * (-2.0 * coeff))
        offsets.append(0)
        
        # Off diagonals
        diagonals.append(np.ones(self.n_points-1) * coeff)
        diagonals.append(np.ones(self.n_points-1) * coeff)
        offsets.extend([1, -1])
        
        # Create kinetic energy operator
        self.T = sparse.diags(diagonals, offsets)
        
        # Potential energy operator
        V = self.potential(self.x)
        self.V = sparse.diags(V)
        
        # Full Hamiltonian
        self.H = self.T + self.V
    
    def solve_ground_state(self) -> Tuple[float, np.ndarray]:
        """Solve for the ground state energy and wavefunction"""
        try:
            # Use higher tolerance for better convergence
            energy, state = eigsh(self.H, k=1, which='SA',
                                maxiter=10000, tol=1e-10)
            
            # Normalize the wavefunction
            state = state.flatten()
            norm = np.sqrt(np.sum(np.abs(state)**2) * self.dx)
            state /= norm
            
            return energy[0], state
            
        except Exception as e:
            print(f"Error in solve_ground_state: {e}")
            raise
    
    def solve_n_states(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for the first n eigenstates"""
        try:
            energy, states = eigsh(self.H, k=n, which='SA',
                                 maxiter=10000, tol=1e-10)
            
            # Normalize each state
            for i in range(n):
                norm = np.sqrt(np.sum(np.abs(states[:, i])**2) * self.dx)
                states[:, i] /= norm
            
            return energy, states
            
        except Exception as e:
            print(f"Error in solve_n_states: {e}")
            raise
    
    def position_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨x⟩"""
        return np.real(np.sum(self.x * np.abs(state)**2) * self.dx)
    
    def momentum_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨p⟩"""
        psi_k = np.fft.fft(state)
        k_expectation = np.sum(self.k * np.abs(psi_k)**2) / len(state)
        return self.hbar * k_expectation
    
    def energy_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨H⟩"""
        Hpsi = self.H @ state
        return np.real(np.vdot(state, Hpsi) * self.dx)