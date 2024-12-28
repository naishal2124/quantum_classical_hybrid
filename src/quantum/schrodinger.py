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
        dk = 2 * np.pi / (self.x_max - self.x_min)
        self.k = np.fft.fftfreq(self.n_points, self.dx) * 2 * np.pi
    
    def kinetic_energy(self, wavefunction: np.ndarray) -> np.ndarray:
        """
        Compute kinetic energy using spectral method
        """
        psi_k = np.fft.fft(wavefunction)
        T_psi_k = -0.5 * self.hbar**2 * self.k**2 * psi_k / self.mass
        return np.fft.ifft(T_psi_k)
    
    def setup_hamiltonian(self):
        """
        Construct the Hamiltonian matrix using kinetic and potential terms
        """
        # Create identity matrix for constructing Hamiltonian
        N = self.n_points
        
        # Kinetic energy matrix using finite difference
        diag = np.ones(N - 1)
        diagonals = [-2 * np.ones(N), diag, diag]
        positions = [0, 1, -1]
        T = sparse.diags(diagonals, positions)
        self.T = -0.5 * self.hbar**2 / (self.mass * self.dx**2) * T
        
        # Potential energy matrix
        V = self.potential(self.x)
        self.V = sparse.diags(V)
        
        # Full Hamiltonian
        self.H = self.T + self.V
    
    def solve_ground_state(self) -> Tuple[float, np.ndarray]:
        """Solve for the ground state energy and wavefunction"""
        try:
            energy, state = eigsh(self.H, k=1, which='SA', 
                                maxiter=5000, tol=1e-8)
            
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
            energies, states = eigsh(self.H, k=n, which='SA',
                                   maxiter=5000, tol=1e-8)
            
            # Normalize each state
            for i in range(n):
                norm = np.sqrt(np.sum(np.abs(states[:, i])**2) * self.dx)
                states[:, i] /= norm
            
            return energies, states
            
        except Exception as e:
            print(f"Error in solve_n_states: {e}")
            raise
    
    def compute_expectation(self, operator: sparse.spmatrix, state: np.ndarray) -> float:
        """Compute expectation value ⟨ψ|A|ψ⟩"""
        return np.real(np.vdot(state, operator @ state) * self.dx)
    
    def position_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨x⟩"""
        return np.sum(self.x * np.abs(state)**2) * self.dx
    
    def momentum_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨p⟩"""
        psi_k = np.fft.fft(state)
        return np.real(np.sum(self.k * np.abs(psi_k)**2)) * self.hbar / self.n_points
    
    def energy_expectation(self, state: np.ndarray) -> float:
        """Compute ⟨H⟩"""
        return self.compute_expectation(self.H, state)