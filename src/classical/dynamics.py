"""
Classical mechanics implementation for quantum-classical hybrid simulations.
Implements Hamiltonian dynamics, numerical integrators, and trajectory analysis.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict
from dataclasses import dataclass
from ..quantum.potentials import Potential

@dataclass
class PhaseSpaceState:
    """State in classical phase space"""
    position: np.ndarray
    momentum: np.ndarray
    time: float = 0.0

    def total_energy(self, mass: float, potential_func: Callable) -> float:
        """Calculate total energy H = T + V"""
        kinetic = np.sum(self.momentum**2) / (2 * mass)
        potential = np.sum(potential_func(self.position))
        return kinetic + potential

class Integrator:
    """Base class for numerical integrators"""
    
    def __init__(self, dt: float, mass: float = 1.0):
        self.dt = dt
        self.mass = mass
    
    def step(self, 
            state: PhaseSpaceState,
            force_func: Callable[[np.ndarray], np.ndarray]) -> PhaseSpaceState:
        """Evolve system by one timestep"""
        raise NotImplementedError

class VelocityVerlet(Integrator):
    """
    Velocity Verlet integrator for Hamiltonian dynamics
    Symplectic integrator that preserves phase space volume
    """
    
    def step(self, 
            state: PhaseSpaceState,
            force_func: Callable[[np.ndarray], np.ndarray]) -> PhaseSpaceState:
        """
        Perform one velocity Verlet step
        
        Args:
            state: Current phase space state
            force_func: Force function F(x)
            
        Returns:
            Updated phase space state
        """
        x = state.position
        p = state.momentum
        m = self.mass
        dt = self.dt
        
        # Half step in momentum
        p_half = p + 0.5 * dt * force_func(x)
        
        # Full step in position
        x_new = x + dt * p_half / m
        
        # Final half step in momentum
        p_new = p_half + 0.5 * dt * force_func(x_new)
        
        return PhaseSpaceState(x_new, p_new, state.time + dt)

class RungeKutta4(Integrator):
    """
    4th order Runge-Kutta integrator
    More accurate but not symplectic
    """
    
    def step(self,
            state: PhaseSpaceState,
            force_func: Callable[[np.ndarray], np.ndarray]) -> PhaseSpaceState:
        """
        Perform one RK4 step
        
        Args:
            state: Current phase space state
            force_func: Force function F(x)
            
        Returns:
            Updated phase space state
        """
        x = state.position
        p = state.momentum
        m = self.mass
        dt = self.dt
        
        # RK4 for position
        k1x = dt * p / m
        k1p = dt * force_func(x)
        
        k2x = dt * (p + 0.5*k1p) / m
        k2p = dt * force_func(x + 0.5*k1x)
        
        k3x = dt * (p + 0.5*k2p) / m
        k3p = dt * force_func(x + 0.5*k2x)
        
        k4x = dt * (p + k3p) / m
        k4p = dt * force_func(x + k3x)
        
        # Update position and momentum
        x_new = x + (k1x + 2*k2x + 2*k3x + k4x) / 6
        p_new = p + (k1p + 2*k2p + 2*k3p + k4p) / 6
        
        return PhaseSpaceState(x_new, p_new, state.time + dt)

class HamiltonianDynamics:
    """
    Classical Hamiltonian dynamics simulator
    """
    
    def __init__(self, 
                potential: Potential,
                integrator: Optional[Integrator] = None,
                mass: float = 1.0):
        """
        Initialize Hamiltonian dynamics
        
        Args:
            potential: Potential energy function
            integrator: Numerical integrator (default: VelocityVerlet)
            mass: Particle mass (default: 1.0)
        """
        self.potential = potential
        self.mass = mass
        
        if integrator is None:
            integrator = VelocityVerlet(dt=0.01, mass=mass)
        self.integrator = integrator
    
    def force(self, x: np.ndarray) -> np.ndarray:
        """Calculate force F = -∇V"""
        return -self.potential.gradient(x)
    
    def evolve(self, 
              initial_state: PhaseSpaceState,
              n_steps: int) -> Dict[str, np.ndarray]:
        """
        Evolve system forward in time
        
        Args:
            initial_state: Initial phase space state
            n_steps: Number of integration steps
            
        Returns:
            Dictionary containing trajectory data
        """
        # Initialize arrays to store trajectory
        positions = np.zeros((n_steps + 1, len(initial_state.position)))
        momenta = np.zeros_like(positions)
        energies = np.zeros(n_steps + 1)
        times = np.zeros(n_steps + 1)
        
        # Store initial state
        state = initial_state
        positions[0] = state.position
        momenta[0] = state.momentum
        energies[0] = state.total_energy(self.mass, self.potential)
        times[0] = state.time
        
        # Time evolution
        for i in range(n_steps):
            state = self.integrator.step(state, self.force)
            
            positions[i+1] = state.position
            momenta[i+1] = state.momentum
            energies[i+1] = state.total_energy(self.mass, self.potential)
            times[i+1] = state.time
        
        return {
            'positions': positions,
            'momenta': momenta,
            'energies': energies,
            'times': times
        }
    
    def poincare_section(self,
                        trajectory: Dict[str, np.ndarray],
                        index: int = 0,
                        threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Poincaré section of trajectory
        
        Args:
            trajectory: Trajectory data from evolve()
            index: Index for section coordinate (default: 0)
            threshold: Section threshold value (default: 0.0)
            
        Returns:
            Tuple of positions and momenta at section crossings
        """
        positions = trajectory['positions']
        momenta = trajectory['momenta']
        
        # Find crossings of threshold
        x = positions[:, index]
        crossings = np.where(np.diff(np.signbit(x - threshold)))[0]
        
        # Interpolate to exact crossing points
        section_positions = []
        section_momenta = []
        
        for i in crossings:
            t = (threshold - x[i]) / (x[i+1] - x[i])
            pos = positions[i] + t * (positions[i+1] - positions[i])
            mom = momenta[i] + t * (momenta[i+1] - momenta[i])
            
            section_positions.append(pos)
            section_momenta.append(mom)
        
        return (np.array(section_positions), np.array(section_momenta))