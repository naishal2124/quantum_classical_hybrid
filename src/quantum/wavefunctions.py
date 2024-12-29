"""
Quantum mechanical wavefunction representations and manipulations.
Implements various wavefunction classes and utilities for quantum calculations.
"""

import numpy as np
import math
from typing import Optional, Union, Tuple, Callable
from scipy.special import hermite
from dataclasses import dataclass
from .operators import Operator
from .potentials import Potential, HarmonicOscillator

@dataclass
class WavefunctionParams:
    """Parameters for wavefunction initialization"""
    center: float = 0.0  # Center of wavepacket
    momentum: float = 0.0  # Initial momentum
    width: float = 1.0  # Width parameter
    phase: float = 0.0  # Initial phase
    mass: float = 1.0  # Particle mass
    hbar: float = 1.0  # Reduced Planck's constant

class Wavefunction:
    """Base class for quantum mechanical wavefunctions"""
    
    def __init__(self, grid_points: int, dx: float, params: Optional[dict] = None):
        """
        Initialize wavefunction on spatial grid.
        
        Args:
            grid_points: Number of spatial grid points
            dx: Grid spacing
            params: Dictionary of wavefunction parameters
        """
        self.N = grid_points
        self.dx = dx
        self.params = WavefunctionParams(**params) if params else WavefunctionParams()
        
        # Set up spatial grid
        self.grid_length = self.N * self.dx
        self.x = np.linspace(-self.grid_length/2, self.grid_length/2, self.N)
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        
        # Initialize wavefunction array
        self.psi = np.zeros(self.N, dtype=np.complex128)
        self._initialize()
        self.normalize()
    
    def _initialize(self):
        """Initialize wavefunction values - to be implemented by subclasses"""
        raise NotImplementedError
    
    def normalize(self):
        """Normalize wavefunction to unit probability"""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        if norm > 0:
            self.psi /= norm
    
    def expectation_value(self, operator: Operator) -> float:
        """
        Compute expectation value of an operator.
        
        Args:
            operator: Quantum mechanical operator
            
        Returns:
            float: Expectation value ⟨ψ|A|ψ⟩
        """
        return operator.expectation_value(self.psi)
    
    def probability_density(self) -> np.ndarray:
        """Return probability density |ψ(x)|²"""
        return np.abs(self.psi)**2
    
    def probability_current(self) -> np.ndarray:
        """
        Compute probability current j(x) = ℏ/m Im(ψ*∂ψ/∂x)
        """
        # Compute derivative using spectral method
        psi_k = np.fft.fft(self.psi)
        dpsi_dx = np.fft.ifft(1j * self.k * psi_k)
        
        return (self.params.hbar/self.params.mass * 
                np.imag(np.conj(self.psi) * dpsi_dx))

class GaussianWavepacket(Wavefunction):
    """
    Gaussian wavepacket ψ(x) = exp(-(x-x₀)²/4σ² + ik₀x + iφ)
    """
    def _initialize(self):
        """Initialize Gaussian wavepacket"""
        x0 = self.params.center
        k0 = self.params.momentum / self.params.hbar  # Corrected wavevector
        sigma = self.params.width
        phi = self.params.phase
        
        # Ensure proper normalization
        norm_factor = (2 * np.pi * sigma**2)**(-0.25)
        self.psi = norm_factor * np.exp(-(self.x - x0)**2 / (4 * sigma**2) + 
                                      1j * k0 * self.x + 1j * phi)

class HarmonicOscillatorState(Wavefunction):
    """
    Harmonic oscillator eigenstate ψₙ(x)
    """
    def __init__(self, grid_points: int, dx: float, n: int = 0, 
                 params: Optional[dict] = None):
        """
        Initialize nth harmonic oscillator eigenstate.
        
        Args:
            grid_points: Number of spatial grid points
            dx: Grid spacing
            n: Quantum number (default: 0 for ground state)
            params: Dictionary of wavefunction parameters
        """
        self.n = n
        super().__init__(grid_points, dx, params)
    
    def _initialize(self):
        """Initialize harmonic oscillator eigenstate"""
        # Extract parameters
        m = self.params.mass
        hbar = self.params.hbar
        omega = 1.0  # Default frequency
        
        # Characteristic length
        alpha = np.sqrt(m * omega / hbar)
        
        # Hermite polynomial
        xi = alpha * self.x
        Hn = hermite(self.n)
        
        # Normalization
        N = 1.0 / np.sqrt(2**self.n * math.factorial(self.n)) * (alpha/np.pi)**0.25
        
        # Wavefunction
        self.psi = N * Hn(xi) * np.exp(-xi**2 / 2)

class SuperpositionState(Wavefunction):
    """
    Superposition of quantum states ψ = Σᵢcᵢψᵢ
    """
    def __init__(self, states: list[Wavefunction], 
                 coefficients: Optional[np.ndarray] = None):
        """
        Initialize superposition state.
        
        Args:
            states: List of component wavefunctions
            coefficients: Complex coefficients for superposition
        """
        # Verify all states have same grid
        if not all(state.N == states[0].N and 
                  np.allclose(state.x, states[0].x) for state in states):
            raise ValueError("All states must be defined on same grid")
        
        self.N = states[0].N
        self.dx = states[0].dx
        self.x = states[0].x
        self.k = states[0].k
        
        # Default to equal superposition if no coefficients given
        if coefficients is None:
            coefficients = np.ones(len(states)) / np.sqrt(len(states))
        
        if len(coefficients) != len(states):
            raise ValueError("Number of coefficients must match number of states")
            
        # Construct superposition
        self.psi = np.sum([c * state.psi for c, state in 
                          zip(coefficients, states)], axis=0)
        self.normalize()