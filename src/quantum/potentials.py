"""
Potential energy functions for quantum systems.
Implements various potentials used in quantum mechanics with their derivatives
for both quantum and classical calculations.
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass

@dataclass
class PotentialParams:
    """Parameters for potential energy functions"""
    mass: float = 1.0
    omega: float = 1.0
    lambda_quartic: float = 0.0  # For anharmonic
    well_depth: float = 1.0      # For double well
    barrier_height: float = 1.0  # For double well

class Potential:
    """Base class for potential energy functions"""
    def __init__(self, params: Optional[Dict] = None):
        self.params = PotentialParams(**params) if params else PotentialParams()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate potential at position x"""
        raise NotImplementedError
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of potential"""
        raise NotImplementedError
    
    def laplacian(self, x: np.ndarray) -> np.ndarray:
        """Compute laplacian of potential"""
        raise NotImplementedError

class HarmonicOscillator(Potential):
    """
    Harmonic oscillator potential: V(x) = ½mω²x²
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        m, omega = self.params.mass, self.params.omega
        return 0.5 * m * omega**2 * x**2
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        m, omega = self.params.mass, self.params.omega
        return m * omega**2 * x
    
    def laplacian(self, x: np.ndarray) -> np.ndarray:
        m, omega = self.params.mass, self.params.omega
        return m * omega**2 * np.ones_like(x)

# Additional potentials will go here