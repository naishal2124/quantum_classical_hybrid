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
        """
        Initialize Euclidean action.
        
        Args:
            potential: Quantum potential energy function
            params: Optional parameters for action
        """
        self.potential = potential
        self.params = ActionParameters(**params if params else {})
        
        # Precompute constants
        self.dtau = self.params.dtau
        self.mass = self.params.mass
        self.hbar = self.params.hbar
        
        # Spring constant for kinetic action
        self.k = self.mass / (2.0 * self.hbar * self.dtau)
    
    def kinetic(self, path: np.ndarray) -> np.ndarray:
        """
        Compute kinetic part of action for each slice.
        
        Args:
            path: Path configuration [n_slices]
            
        Returns:
            Kinetic action per slice [n_slices]
        """
        n = len(path)
        dx = np.zeros_like(path)
        # Compute differences with periodic boundary
        dx[:-1] = path[1:] - path[:-1]
        dx[-1] = path[0] - path[-1]
        return self.k * dx**2
    
    def potential_energy(self, path: np.ndarray) -> np.ndarray:
        """
        Compute potential part of action for each slice.
        
        Args:
            path: Path configuration [n_slices]
            
        Returns:
            Potential action per slice [n_slices]
        """
        V = self.potential(path)
        return V * self.dtau
    
    def total(self, path: np.ndarray) -> float:
        """
        Compute total Euclidean action.
        
        Args:
            path: Path configuration [n_slices]
            
        Returns:
            Total action (dimensionless)
        """
        return np.sum(self.kinetic(path) + self.potential_energy(path))
    
    def local_action(self, path: np.ndarray, slice_idx: int) -> float:
        """
        Compute action change for single slice.
        
        Args:
            path: Path configuration [n_slices]
            slice_idx: Index of time slice
            
        Returns:
            Local action contribution
        """
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
        """
        Compute forces -∂S/∂x for all slices.
        
        Args:
            path: Path configuration [n_slices]
            
        Returns:
            Forces for each slice [n_slices]
        """
        n = len(path)
        forces = np.zeros_like(path)
        
        # Kinetic force
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            dx_prev = path[i] - path[prev_idx]
            dx_next = path[next_idx] - path[i]
            forces[i] = -2 * self.k * (dx_next - dx_prev)
        
        # Potential force
        forces -= self.potential.gradient(path) * self.dtau
        
        return forces
    
    def thermodynamic_energy(self, paths: np.ndarray) -> Tuple[float, float]:
        """
        Compute thermodynamic energy estimator.
        E = -∂/∂β log(Z) = ⟨(1/2)m(dx/dτ)² + V(x) - (N/2β)⟩
        
        Args:
            paths: Multiple path configurations [n_paths, n_slices]
            
        Returns:
            Tuple of (energy, error estimate)
        """
        n_paths, n_slices = paths.shape
        energies = np.zeros(n_paths)
        
        for i in range(n_paths):
            path = paths[i]
            
            # Kinetic energy term
            dx = np.zeros_like(path)
            dx[:-1] = path[1:] - path[:-1]
            dx[-1] = path[0] - path[-1]
            kinetic = np.sum(dx**2) * self.mass / (2 * self.hbar**2 * self.dtau)
            
            # Potential energy term
            potential = np.sum(self.potential(path)) / n_slices
            
            # Zero point energy correction
            zero_point = n_slices / (2 * self.params.beta)
            
            energies[i] = (kinetic + potential - zero_point) / n_slices
        
        return np.mean(energies), np.std(energies) / np.sqrt(n_paths)