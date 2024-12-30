"""
Quantum-classical correspondence implementation.
Implements tools for analyzing quantum-classical transitions and relationships.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.integrate import solve_ivp
from dataclasses import dataclass

from ..quantum.wavefunctions import Wavefunction, GaussianWavepacket
from ..quantum.operators import Position, Momentum, Operator
from ..classical.dynamics import PhaseSpaceState, HamiltonianDynamics

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
    Implements Ehrenfest's theorem for quantum-classical correspondence.
    Tracks evolution of expectation values according to classical equations.
    """
    
    def __init__(self, hamiltonian: HamiltonianDynamics):
        """Initialize with classical Hamiltonian system"""
        self.hamiltonian = hamiltonian
        self.mass = hamiltonian.mass
        # Use small timestep for better energy conservation
        self.hamiltonian.integrator.dt = 0.005
    
    def compute_expectation_values(self, wavefunction: Wavefunction) -> ExpectationValues:
        """Compute quantum expectation values"""
        x_op = Position(wavefunction.N, wavefunction.dx, wavefunction.params.hbar)
        p_op = Momentum(wavefunction.N, wavefunction.dx, wavefunction.params.hbar)
        x2_op = SquaredOperator(x_op)
        p2_op = SquaredOperator(p_op)
        
        x_expect = wavefunction.expectation_value(x_op)
        p_expect = wavefunction.expectation_value(p_op)
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
        """Generate classical trajectory from quantum expectation values"""
        expect = self.compute_expectation_values(wavefunction)
        initial_state = PhaseSpaceState(
            position=np.array([expect.position]),
            momentum=np.array([expect.momentum])
        )
        
        # Evolve with double steps and subsample
        full_trajectory = self.hamiltonian.evolve(initial_state, n_steps * 2)
        
        # Subsample each array in the dictionary
        return {
            'positions': full_trajectory['positions'][::2],
            'momenta': full_trajectory['momenta'][::2],
            'energies': full_trajectory['energies'][::2],
            'times': full_trajectory['times'][::2]
        }
    
    def ehrenfest_evolution(self, 
                          wavefunction: Wavefunction,
                          times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evolve system using Ehrenfest dynamics
        
        Args:
            wavefunction: Initial quantum state
            times: Time points for evolution
            
        Returns:
            Dictionary containing evolution data
        """
        expect = self.compute_expectation_values(wavefunction)
        
        def ehrenfest_rhs(t: float, y: np.ndarray) -> np.ndarray:
            x, p = y
            dx_dt = p / self.mass
            dp_dt = -self.hamiltonian.force(np.array([x]))[0]
            return np.array([dx_dt, dp_dt])
        
        solution = solve_ivp(
            ehrenfest_rhs,
            (times[0], times[-1]),
            np.array([expect.position, expect.momentum]),
            t_eval=times,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        
        return {
            'times': solution.t,
            'position': solution.y[0],
            'momentum': solution.y[1],
            'potential': self.hamiltonian.potential(solution.y[0])
        }

class WignerDistribution:
    """Wigner quasi-probability distribution."""
    
    def __init__(self, n_grid: int = 100):
        self.n_grid = n_grid
    
    def compute(self,
               wavefunction: Wavefunction,
               x_range: Optional[Tuple[float, float]] = None,
               p_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Wigner distribution for a wavefunction"""
        if x_range is None:
            x_max = max(abs(wavefunction.x[0]), abs(wavefunction.x[-1]))
            x_range = (-x_max, x_max)
        
        if p_range is None:
            dp = wavefunction.params.hbar / (2 * wavefunction.params.width)
            p_max = 5 * dp
            p_range = (-p_max, p_max)
        
        x = np.linspace(x_range[0], x_range[1], self.n_grid)
        p = np.linspace(p_range[0], p_range[1], self.n_grid)
        X, P = np.meshgrid(x, p)
        
        dx = wavefunction.dx
        y = np.linspace(-x_max, x_max, self.n_grid)
        dy = y[1] - y[0]
        
        W = np.zeros((self.n_grid, self.n_grid), dtype=np.float64)
        hbar = wavefunction.params.hbar
        
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                x_val = x[i]
                p_val = p[j]
                
                x_plus = x_val + y/2
                x_minus = x_val - y/2
                
                psi_plus = np.interp(x_plus, wavefunction.x, wavefunction.psi,
                                   left=0, right=0)
                psi_minus = np.interp(x_minus, wavefunction.x, wavefunction.psi,
                                    left=0, right=0)
                
                integrand = np.conj(psi_plus) * psi_minus * np.exp(-1j * p_val * y / hbar)
                W[j,i] = np.real(np.sum(integrand)) * dy / (2 * np.pi * hbar)
        
        # Normalize
        W /= np.sum(W) * (x[1] - x[0]) * (p[1] - p[0])
        
        return X, P, W
    
    def classical_limit(self, 
                       wavefunction: Wavefunction,
                       hbar_scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Wigner distribution in classical limit (ℏ → 0)"""
        scaled_params = {
            "center": wavefunction.params.center,
            "momentum": wavefunction.params.momentum,
            "width": wavefunction.params.width * np.sqrt(hbar_scale),
            "mass": wavefunction.params.mass,
            "hbar": wavefunction.params.hbar * hbar_scale
        }
        
        scaled_wf = GaussianWavepacket(
            wavefunction.N,
            wavefunction.dx,
            scaled_params
        )
        
        return self.compute(scaled_wf)

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
    x = wavefunction.x
    x0 = classical_state.position[0]
    p0 = classical_state.momentum[0]
    hbar = wavefunction.params.hbar
    
    psi_classical = np.exp(-(x - x0)**2 / (4 * sigma_x**2) + 1j * p0 * x / hbar)
    norm = np.sqrt(np.sum(np.abs(psi_classical)**2) * wavefunction.dx)
    psi_classical /= norm
    
    overlap = np.sum(np.conj(wavefunction.psi) * psi_classical) * wavefunction.dx
    return np.abs(overlap)**2