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


class AnharmonicOscillator(Potential):
    """
    Anharmonic oscillator potential: V(x) = ½mω²x² + λx⁴
    Shows breakdown of perturbation theory and classical chaos
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        m, omega = self.params.mass, self.params.omega
        lambda_quartic = self.params.lambda_quartic
        return 0.5 * m * omega**2 * x**2 + lambda_quartic * x**4
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        m, omega = self.params.mass, self.params.omega
        lambda_quartic = self.params.lambda_quartic
        return m * omega**2 * x + 4 * lambda_quartic * x**3
    
    def laplacian(self, x: np.ndarray) -> np.ndarray:
        m, omega = self.params.mass, self.params.omega
        lambda_quartic = self.params.lambda_quartic
        return m * omega**2 + 12 * lambda_quartic * x**2

class DoubleWell(Potential):
    """
    Double well potential: V(x) = -ax² + bx⁴
    Exhibits tunneling behavior and ground state splitting
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        a = self.params.well_depth
        b = self.params.barrier_height
        return -a * x**2 + b * x**4
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        a = self.params.well_depth
        b = self.params.barrier_height
        return -2 * a * x + 4 * b * x**3
    
    def laplacian(self, x: np.ndarray) -> np.ndarray:
        a = self.params.well_depth
        b = self.params.barrier_height
        return -2 * a + 12 * b * x**2