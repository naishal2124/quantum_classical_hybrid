"""
Quantum-classical correspondence implementation.
Implements tools for analyzing quantum-classical transitions and relationships.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict
from ..quantum.wavefunctions import Wavefunction
from ..quantum.operators import Position, Momentum, Operator
from ..classical.dynamics import PhaseSpaceState, HamiltonianDynamics
from dataclasses import dataclass

@dataclass
class ExpectationValues:
    """Container for quantum expectation values"""
    position: float
    momentum: float
    position_variance: float
    momentum_variance: float
    uncertainty_product: float

class SquaredOperator(Operator):
    """Helper class for squared operator expectation values"""
    def __init__(self, base_operator: Operator):
        super().__init__(base_operator.N, base_operator.dx, base_operator.hbar)
        self.base = base_operator
        
    def to_matrix(self):
        base_matrix = self.base.to_matrix()
        return base_matrix @ base_matrix

class EhrenfestDynamics:
    """
    Implements Ehrenfest's theorem for quantum-classical correspondence
    Tracks evolution of expectation values according to classical equations
    """
    
    def __init__(self, hamiltonian: HamiltonianDynamics):
        self.hamiltonian = hamiltonian
        self.mass = hamiltonian.mass
    
    def compute_expectation_values(self, wavefunction: Wavefunction) -> ExpectationValues:
        """
        Compute relevant expectation values from wavefunction
        
        Args:
            wavefunction: Quantum state
            
        Returns:
            ExpectationValues container
        """
        # Create operators
        x_op = Position(wavefunction.N, wavefunction.dx)
        p_op = Momentum(wavefunction.N, wavefunction.dx)
        x2_op = SquaredOperator(x_op)
        p2_op = SquaredOperator(p_op)
        
        # Compute expectation values
        x_expect = wavefunction.expectation_value(x_op)
        p_expect = wavefunction.expectation_value(p_op)
        
        # Compute variances
        x2_expect = wavefunction.expectation_value(x2_op)
        p2_expect = wavefunction.expectation_value(p2_op)
        
        x_var = x2_expect - x_expect**2
        p_var = p2_expect - p_expect**2
        
        return ExpectationValues(
            position=x_expect,
            momentum=p_expect,
            position_variance=x_var,
            momentum_variance=p_var,
            uncertainty_product=np.sqrt(x_var * p_var)
        )
    
    def classical_trajectory_from_quantum(self,
                                       wavefunction: Wavefunction,
                                       n_steps: int) -> Dict[str, np.ndarray]:
        """
        Generate classical trajectory from quantum expectation values
        
        Args:
            wavefunction: Initial quantum state
            n_steps: Number of evolution steps
            
        Returns:
            Classical trajectory dictionary
        """
        # Get initial expectation values
        expect = self.compute_expectation_values(wavefunction)
        
        # Create classical initial state
        initial_state = PhaseSpaceState(
            position=np.array([expect.position]),
            momentum=np.array([expect.momentum])
        )
        
        # Evolve classically
        return self.hamiltonian.evolve(initial_state, n_steps)

class WignerDistribution:
    """
    Wigner quasi-probability distribution
    Provides phase space representation of quantum states
    """
    
    def __init__(self, n_grid: int = 100):
        self.n_grid = n_grid
    
    def compute(self, 
                wavefunction: Wavefunction,
                x_range: Optional[Tuple[float, float]] = None,
                p_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Wigner distribution for a wavefunction
        
        Args:
            wavefunction: Quantum state
            x_range: Optional position range (min, max)
            p_range: Optional momentum range (min, max)
            
        Returns:
            Tuple of (X grid, P grid, Wigner function W(x,p))
        """
        # Set up grids with appropriate scaling
        if x_range is None:
            x_center = wavefunction.params.center
            width = 4 * wavefunction.params.width
            x_range = (x_center - width, x_center + width)
            
        if p_range is None:
            p_center = wavefunction.params.momentum
            dp = wavefunction.params.hbar / wavefunction.params.width
            p_range = (p_center - 4*dp, p_center + 4*dp)
        
        x_grid = np.linspace(x_range[0], x_range[1], self.n_grid)
        p_grid = np.linspace(p_range[0], p_range[1], self.n_grid)
        X, P = np.meshgrid(x_grid, p_grid)
        dx_grid = x_grid[1] - x_grid[0]
        dp_grid = p_grid[1] - p_grid[0]
        
        # Create meshgrid for integration
        y = np.linspace(-width*2, width*2, self.n_grid)
        dy = y[1] - y[0]
        
        # Initialize Wigner function
        W = np.zeros_like(X)
        
        # Compute for each point in phase space
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                x_val = x_grid[i]
                p_val = p_grid[j]
                
                # Compute wavefunction at shifted points
                x_plus = x_val + y/2
                x_minus = x_val - y/2
                
                psi_plus = np.interp(x_plus, wavefunction.x, wavefunction.psi)
                psi_minus = np.interp(x_minus, wavefunction.x, wavefunction.psi)
                
                # Wigner transform integral
                integrand = np.conj(psi_plus) * psi_minus * np.exp(-1j * p_val * y / wavefunction.params.hbar)
                W[i,j] = np.real(np.sum(integrand) * dy) / (2 * np.pi * wavefunction.params.hbar)
        
        # Normalize the distribution
        norm = np.sum(W) * dx_grid * dp_grid
        if norm != 0:
            W /= np.abs(norm)
        
        return X, P, W

    def classical_limit(self, 
                       wavefunction: Wavefunction,
                       hbar_scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Wigner distribution in classical limit (ℏ → 0)
        
        Args:
            wavefunction: Quantum state
            hbar_scale: Scaling factor for ℏ
            
        Returns:
            Tuple of (X grid, P grid, Classical limit Wigner function)
        """
        # Create a copy of the wavefunction
        psi_copy = type(wavefunction)(
            wavefunction.N,
            wavefunction.dx,
            {**wavefunction.params.__dict__}
        )
        
        # Copy wavefunction data and scale ℏ
        psi_copy.psi = wavefunction.psi.copy()
        psi_copy.normalize()
        psi_copy.params.hbar *= hbar_scale
        
        # Compute with scaled ℏ
        X, P, W = self.compute(psi_copy)
        
        return X, P, W

def quantum_classical_fidelity(wavefunction: Wavefunction,
                             classical_state: PhaseSpaceState,
                             sigma_x: float,
                             sigma_p: float) -> float:
    """
    Compute fidelity between quantum state and classical phase space point
    
    Args:
        wavefunction: Quantum state
        classical_state: Classical phase space point
        sigma_x: Position uncertainty
        sigma_p: Momentum uncertainty
        
    Returns:
        Fidelity between states
    """
    # Create classical gaussian wavepacket
    x = wavefunction.x
    x0 = classical_state.position[0]
    p0 = classical_state.momentum[0]
    hbar = wavefunction.params.hbar
    
    # Gaussian wavepacket
    psi_classical = np.exp(-(x - x0)**2 / (4 * sigma_x**2) + 
                          1j * p0 * x / hbar)
    psi_classical /= np.sqrt(np.sum(np.abs(psi_classical)**2) * wavefunction.dx)
    
    # Compute overlap
    overlap = np.sum(np.conj(wavefunction.psi) * psi_classical) * wavefunction.dx
    return np.abs(overlap)**2