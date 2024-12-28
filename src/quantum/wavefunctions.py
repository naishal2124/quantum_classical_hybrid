"""
Quantum mechanical wavefunctions and their properties.
"""

import numpy as np
from scipy.special import hermite
from typing import Optional, Tuple, Union

class WaveFunction:
    """Base class for quantum mechanical wavefunctions"""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.dx = grid[1] - grid[0]
        self.psi = None
    
    def __call__(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate wavefunction at points x"""
        if self.psi is None:
            self._build_wavefunction()
        if x is None:
            return self.psi
        return np.interp(x, self.grid, self.psi)
    
    def _build_wavefunction(self):
        """Build the wavefunction"""
        raise NotImplementedError
    
    def normalize(self):
        """Normalize the wavefunction"""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.psi /= norm
    
    def probability_density(self) -> np.ndarray:
        """Compute |ψ(x)|²"""
        return np.abs(self.__call__())**2

class GaussianWavePacket(WaveFunction):
    """Gaussian wave packet ψ(x) = exp(-(x-x₀)²/4σ² + ik₀x)"""
    
    def __init__(self, grid: np.ndarray, x0: float = 0.0, 
                 k0: float = 0.0, sigma: float = 1.0):
        super().__init__(grid)
        self.x0 = x0
        self.k0 = k0
        self.sigma = sigma
    
    def _build_wavefunction(self):
        x = self.grid
        self.psi = np.exp(-(x - self.x0)**2 / (4 * self.sigma**2) + 
                         1j * self.k0 * x)
        self.normalize()

class HarmonicOscillator(WaveFunction):
    """Harmonic oscillator eigenstate ψₙ(x)"""
    
    def __init__(self, grid: np.ndarray, n: int = 0, 
                 omega: float = 1.0, mass: float = 1.0, hbar: float = 1.0):
        super().__init__(grid)
        self.n = n
        self.omega = omega
        self.mass = mass
        self.hbar = hbar
    
    def _build_wavefunction(self):
        x = self.grid
        # Characteristic length
        alpha = np.sqrt(self.mass * self.omega / self.hbar)
        
        # Normalized Hermite polynomial
        Hn = hermite(self.n)
        prefactor = 1.0 / np.sqrt(2**self.n * np.math.factorial(self.n))
        prefactor *= (alpha / np.pi)**0.25
        
        self.psi = prefactor * Hn(alpha * x) * np.exp(-alpha * x**2 / 2)
        self.normalize()

class DoubleWell(WaveFunction):
    """Double well potential eigenstate"""
    
    def __init__(self, grid: np.ndarray, n: int = 0, 
                 a: float = 1.0, b: float = 0.25):
        super().__init__(grid)
        self.n = n
        self.a = a  # Well depth
        self.b = b  # Barrier height
    
    def _build_wavefunction(self):
        # For n=0,1, approximate as symmetric/antisymmetric combinations
        # of Gaussian states centered at potential minima
        x = self.grid
        x_min = np.sqrt(self.a / (2 * self.b))
        
        # Gaussian states centered at ±x_min
        psi_left = np.exp(-(x + x_min)**2 / 2)
        psi_right = np.exp(-(x - x_min)**2 / 2)
        
        # Symmetric (n=0) or antisymmetric (n=1) combination
        if self.n == 0:
            self.psi = psi_left + psi_right
        elif self.n == 1:
            self.psi = psi_left - psi_right
        else:
            raise NotImplementedError("Only n=0,1 implemented")
        
        self.normalize()