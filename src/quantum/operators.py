"""
Quantum mechanical operators and their matrix representations.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Union
import torch

class Operator:
    """Base class for quantum mechanical operators"""
    
    def __init__(self, grid_points: int, dx: float, hbar: float = 1.0):
        self.N = grid_points
        self.dx = dx
        self.hbar = hbar
        self.matrix = None
    
    def to_matrix(self) -> sparse.spmatrix:
        """Convert operator to sparse matrix representation"""
        if self.matrix is None:
            self._build_matrix()
        return self.matrix
    
    def _build_matrix(self):
        """Build the operator matrix"""
        raise NotImplementedError
    
    def expectation_value(self, wavefunction: np.ndarray) -> float:
        """Compute expectation value ⟨ψ|A|ψ⟩"""
        return np.real(np.vdot(wavefunction, self.to_matrix() @ wavefunction) * self.dx)

class Position(Operator):
    """Position operator x"""
    
    def _build_matrix(self):
        x = np.linspace(-self.N/2 * self.dx, self.N/2 * self.dx, self.N)
        self.matrix = sparse.diags(x)

class Momentum(Operator):
    """Momentum operator p = -iℏ∂/∂x"""
    
    def _build_matrix(self):
        # Use central difference for first derivative
        diagonals = []
        offsets = []
        
        # Off-diagonal elements
        diagonals.append(np.ones(self.N-1))  # Upper diagonal
        diagonals.append(-np.ones(self.N-1))  # Lower diagonal
        offsets.extend([1, -1])
        
        self.matrix = -1j * self.hbar / (2 * self.dx) * \
                     sparse.diags(diagonals, offsets)

class KineticEnergy(Operator):
    """Kinetic energy operator T = -ℏ²/2m ∂²/∂x²"""
    
    def __init__(self, grid_points: int, dx: float, mass: float = 1.0, hbar: float = 1.0):
        super().__init__(grid_points, dx, hbar)
        self.mass = mass
    
    def _build_matrix(self):
        # Use second-order central difference
        diagonals = []
        offsets = []
        
        # Main diagonal
        diagonals.append(-2 * np.ones(self.N))
        offsets.append(0)
        
        # Off-diagonals
        diagonals.extend([np.ones(self.N-1), np.ones(self.N-1)])
        offsets.extend([1, -1])
        
        self.matrix = -self.hbar**2 / (2 * self.mass * self.dx**2) * \
                     sparse.diags(diagonals, offsets)

class PotentialEnergy(Operator):
    """Potential energy operator V(x)"""
    
    def __init__(self, potential_func, grid_points: int, dx: float):
        super().__init__(grid_points, dx)
        self.potential_func = potential_func
        
    def _build_matrix(self):
        x = np.linspace(-self.N/2 * self.dx, self.N/2 * self.dx, self.N)
        V = self.potential_func(x)
        self.matrix = sparse.diags(V)

class Hamiltonian(Operator):
    """Hamiltonian operator H = T + V"""
    
    def __init__(self, T: KineticEnergy, V: PotentialEnergy):
        super().__init__(T.N, T.dx, T.hbar)
        self.T = T
        self.V = V
    
    def _build_matrix(self):
        self.matrix = self.T.to_matrix() + self.V.to_matrix()