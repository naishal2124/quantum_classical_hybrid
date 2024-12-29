"""
Quantum mechanical operators and their matrix representations.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Union

class Operator:
    """Base class for quantum mechanical operators"""
    
    def __init__(self, grid_points: int, dx: float, hbar: float = 1.0):
        self.N = grid_points
        self.dx = dx
        self.hbar = hbar
        self.matrix = None
        
        # Set up spatial grid
        self.grid_length = self.N * self.dx
        self.x = np.linspace(-self.grid_length/2, self.grid_length/2, self.N)
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
    
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
        if not isinstance(wavefunction, np.ndarray):
            wavefunction = np.array(wavefunction)
        matrix = self.to_matrix()
        return float(np.real(np.sum(np.conj(wavefunction) * (matrix @ wavefunction)) * self.dx))

class Position(Operator):
    """Position operator x"""
    
    def _build_matrix(self):
        """Build position operator matrix"""
        self.matrix = sparse.diags(self.x)
    
    def expectation_value(self, wavefunction: np.ndarray) -> float:
        """Compute position expectation value"""
        return float(np.real(np.sum(self.x * np.abs(wavefunction)**2) * self.dx))

class Momentum(Operator):
    """Momentum operator p = -iℏ∂/∂x"""
    
    def _build_matrix(self):
        """Build momentum operator using spectral method"""
        self.matrix = sparse.diags(self.hbar * self.k)
    
    def expectation_value(self, wavefunction: np.ndarray) -> float:
        """Compute momentum expectation value using spectral method"""
        psi_k = np.fft.fft(wavefunction) / np.sqrt(self.N)
        return float(np.real(np.sum(self.hbar * self.k * np.abs(psi_k)**2) * self.dx))

class KineticEnergy(Operator):
    """Kinetic energy operator T = -ℏ²/2m ∂²/∂x²"""
    
    def __init__(self, grid_points: int, dx: float, mass: float = 1.0, hbar: float = 1.0):
        super().__init__(grid_points, dx, hbar)
        self.mass = mass
    
    def _build_matrix(self):
        """Build kinetic energy operator using spectral method"""
        k = self.k
        T_k = 0.5 * self.hbar**2 * k**2 / self.mass
        self.matrix = sparse.diags(T_k)
    
    def expectation_value(self, wavefunction: np.ndarray) -> float:
        """Compute kinetic energy expectation value using spectral method"""
        psi_k = np.fft.fft(wavefunction) / np.sqrt(self.N)
        T_k = 0.5 * self.hbar**2 * self.k**2 / self.mass
        return float(np.real(np.sum(T_k * np.abs(psi_k)**2) * self.dx))

class PotentialEnergy(Operator):
    """Potential energy operator V(x)"""
    
    def __init__(self, potential_func, grid_points: int, dx: float, hbar: float = 1.0):
        super().__init__(grid_points, dx, hbar)
        self.potential_func = potential_func
    
    def _build_matrix(self):
        """Build potential energy operator"""
        V = self.potential_func(self.x)
        self.matrix = sparse.diags(V)
    
    def expectation_value(self, wavefunction: np.ndarray) -> float:
        """Compute potential energy expectation value"""
        V = self.potential_func(self.x)
        return float(np.real(np.sum(V * np.abs(wavefunction)**2) * self.dx))

class Hamiltonian(Operator):
    """Hamiltonian operator H = T + V"""
    
    def __init__(self, T: KineticEnergy, V: PotentialEnergy):
        super().__init__(T.N, T.dx, T.hbar)
        self.T = T
        self.V = V
    
    def _build_matrix(self):
        """Build Hamiltonian operator"""
        self.matrix = self.T.to_matrix() + self.V.to_matrix()
    
    def expectation_value(self, wavefunction: np.ndarray) -> float:
        """Compute total energy expectation value"""
        T = self.T.expectation_value(wavefunction)
        V = self.V.expectation_value(wavefunction)
        return T + V