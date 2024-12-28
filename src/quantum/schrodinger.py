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
        
        Args:
            config: Dictionary containing:
                - n_points: Number of spatial grid points
                - x_min: Minimum position
                - x_max: Maximum position
                - mass: Particle mass
                - potential: Potential energy function instance
                - hbar: Reduced Planck constant (default=1.0)
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
        
        # Momentum space grid
        self.k = 2 * np.pi * np.fft.fftfreq(self.n_points, self.dx)
        
    def setup_hamiltonian(self):
        """
        Construct the Hamiltonian matrix using sparse matrices.
        H = -ℏ²/2m ∂²/∂x² + V(x)
        """
        # Kinetic energy in momentum space
        T = 0.5 * self.hbar**2 * self.k**2 / self.mass
        self.T = sparse.diags(T)
        
        # Potential energy in position space
        V = self.potential(self.x)
        self.V = sparse.diags(V)
        
        # Full Hamiltonian
        self.H = self.T + self.V
    
    def solve_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Solve for the ground state energy and wavefunction.
        
        Returns:
            Tuple of (ground state energy, ground state wavefunction)
        """
        # Solve for lowest eigenvalue/eigenvector
        energy, state = eigsh(self.H, k=1, which='SA')
        
        # Normalize the wavefunction
        state = state.flatten()
        norm = np.sqrt(np.sum(np.abs(state)**2) * self.dx)
        state /= norm
        
        return energy[0], state
    
    def solve_n_states(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the first n eigenstates.
        
        Args:
            n: Number of states to compute
            
        Returns:
            Tuple of (energies, states) where states is a matrix
            with eigenstates as columns
        """
        energies, states = eigsh(self.H, k=n, which='SA')
        
        # Normalize each state
        for i in range(n):
            norm = np.sqrt(np.sum(np.abs(states[:, i])**2) * self.dx)
            states[:, i] /= norm
        
        return energies, states
    
    def compute_expectation(self, operator: sparse.spmatrix, state: np.ndarray) -> float:
        """
        Compute expectation value ⟨ψ|A|ψ⟩ for operator A.
        
        Args:
            operator: Sparse matrix representing the operator
            state: Wavefunction to compute expectation value with
            
        Returns:
            Expectation value (real number)
        """
        return np.real(np.vdot(state, operator @ state) * self.dx)
    
    def position_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨x⟩"""
        return np.sum(self.x * np.abs(state)**2) * self.dx
    
    def momentum_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨p⟩"""
        return np.sum(self.k * np.abs(np.fft.fft(state))**2) * self.hbar / len(state)
    
    def energy_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨H⟩"""
        return self.compute_expectation(self.H, state)