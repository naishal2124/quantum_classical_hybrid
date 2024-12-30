"""
Euclidean action for path integral Monte Carlo.
Implements discretized action and its derivatives for quantum systems.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from ..quantum.potentials import Potential, HarmonicOscillator

@dataclass
class ActionParameters:
    """Parameters for Euclidean action"""
    mass: float = 1.0
    beta: float = 1.0  # Inverse temperature
    n_slices: int = 32  # Number of imaginary time slices
    hbar: float = 1.0
    
    @property
    def dtau(self) -> float:
        """Imaginary time step"""
        return self.beta / self.n_slices

class EuclideanAction:
    """
    Discretized Euclidean action for path integral Monte Carlo.
    S[x] = ∫dτ [m/2 (dx/dτ)² + V(x)]
    """
    
    def __init__(self, potential: Potential, params: Optional[dict] = None):
        """Initialize Euclidean action."""
        self.potential = potential
        self.params = ActionParameters(**params if params else {})
        
        # Precompute constants
        self.dtau = self.params.dtau
        self.mass = self.params.mass
        self.hbar = self.params.hbar
        
        # Spring constant for kinetic action
        self.k = self.mass / (2.0 * self.dtau * self.hbar**2)
    
    def _get_differences(self, path: np.ndarray, forward: bool = True) -> np.ndarray:
        """Helper to compute periodic differences."""
        if forward:
            return np.roll(path, -1) - path
        else:
            return path - np.roll(path, 1)
    
    def kinetic(self, path: np.ndarray) -> np.ndarray:
        """Compute kinetic part of action for each slice."""
        dx = self._get_differences(path, forward=True)
        return self.k * dx**2
    
    def potential_energy(self, path: np.ndarray) -> np.ndarray:
        """Compute potential part of action for each slice."""
        # Full imaginary time weighting for each slice
        return self.potential(path) * self.dtau
    
    def total(self, path: np.ndarray) -> float:
        """Compute total Euclidean action."""
        return np.sum(self.kinetic(path) + self.potential_energy(path))
    
    def local_action(self, path: np.ndarray, slice_idx: int) -> float:
        """Compute action change for single slice."""
        n = len(path)
        prev_idx = (slice_idx - 1) % n
        next_idx = (slice_idx + 1) % n
        
        # Kinetic contribution from both links
        dx_prev = path[slice_idx] - path[prev_idx]
        dx_next = path[next_idx] - path[slice_idx]
        kinetic = self.k * (dx_prev**2 + dx_next**2)
        
        # Potential contribution
        potential = self.potential(path[slice_idx]) * self.dtau
        
        return kinetic + potential
    
    def force(self, path: np.ndarray) -> np.ndarray:
        """Compute forces -∂S/∂x for all slices."""
        # Kinetic force from both neighbors
        dx_forward = self._get_differences(path, forward=True)
        dx_backward = self._get_differences(path, forward=False)
        kinetic_force = -2.0 * self.k * (dx_forward - dx_backward)
        
        # Potential force
        potential_force = -self.dtau * self.potential.gradient(path)
        
        return kinetic_force + potential_force
    
    def thermodynamic_energy(self, paths: np.ndarray) -> Tuple[float, float]:
        """
        Compute thermodynamic energy estimator.
        Uses primitive and virial estimators with proper weighting.
        """
        n_paths, n_slices = paths.shape
        energies = np.zeros(n_paths)
        
        for i in range(n_paths):
            path = paths[i]
            
            # Primitive kinetic energy estimator
            dx = self._get_differences(path, forward=True)
            ke_primitive = np.mean(dx**2) * self.mass / (2.0 * self.dtau * self.hbar**2)
            
            # Virial estimator terms
            x_avg = np.mean(path)
            v_grad = np.mean(self.potential.gradient(path) * (path - x_avg))
            ke_virial = v_grad / 2.0
            
            # Potential energy
            pe = np.mean(self.potential(path))
            
            # Combined estimator with temperature correction
            zero_point = 1.0 / (2.0 * self.params.beta)
            energies[i] = 0.5 * (ke_primitive + ke_virial) + pe - zero_point
        
        return np.mean(energies), np.std(energies) / np.sqrt(n_paths)