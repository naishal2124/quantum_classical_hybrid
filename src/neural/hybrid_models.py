"""
Hybrid Neural Models for Quantum-Classical Systems.
Implements neural networks that can learn quantum-classical transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

class QuantumClassicalNet(nn.Module):
    """Neural network that learns quantum-classical correspondence."""
    
    def __init__(self, 
                input_dim: int,
                hidden_dim: int = 64,
                n_layers: int = 3,
                temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.input_dim = input_dim
        
        # Backbone network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layers with proper initialization
        self.wavefunction = nn.Linear(hidden_dim, 2)
        self.classical_density = nn.Linear(hidden_dim, 1)
        
        # Initialize to ensure proper normalization
        nn.init.normal_(self.wavefunction.weight, std=0.01)
        nn.init.zeros_(self.wavefunction.bias)
        nn.init.normal_(self.classical_density.weight, std=0.01)
        nn.init.zeros_(self.classical_density.bias)
    
    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized quantum wavefunction."""
        features = self.backbone(x)
        psi = self.wavefunction(features)
        psi_real, psi_imag = psi.chunk(2, dim=-1)
        psi = torch.complex(psi_real, psi_imag)
        
        # Normalize wavefunction
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2, dim=-1, keepdim=True))
        return psi / (norm + 1e-8)
    
    def classical_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized classical probability density."""
        features = self.backbone(x)
        logits = self.classical_density(features)
        
        # Ensure proper normalization through softmax
        return F.softplus(logits)
    
    def forward(self, x: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        """Interpolate between quantum and classical behaviors."""
        if beta is None:
            beta = 1.0 / self.temperature
            
        # Get normalized probabilities
        psi = self.quantum_forward(x)
        quantum_prob = torch.abs(psi)**2
        
        classical_prob = self.classical_forward(x)
        
        # Temperature-dependent interpolation
        weight = 1.0 / (1.0 + torch.exp(-beta))
        return weight * quantum_prob + (1.0 - weight) * classical_prob

class VariationalHybridState(nn.Module):
    """Variational state for quantum-classical systems."""
    
    def __init__(self,
                state_dim: int,
                hidden_dim: int = 64,
                n_layers: int = 3):
        super().__init__()
        
        # State transformation network
        self.transform = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh()) 
              for _ in range(n_layers-1)],
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Phase network
        self.phase = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, compute_phase: bool = True) -> torch.Tensor:
        """Transform state with optional phase."""
        x_transformed = self.transform(x)
        
        if compute_phase:
            phase = 2 * np.pi * torch.tanh(self.phase(x))  # Bounded phase
            return torch.complex(
                x_transformed * torch.cos(phase),
                x_transformed * torch.sin(phase)
            )
        return x_transformed
    
    def probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized probability density."""
        state = self.forward(x, compute_phase=True)
        prob = torch.abs(state)**2
        return prob / (torch.sum(prob, dim=-1, keepdim=True) + 1e-8)
    
    def classical_density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized classical density."""
        state = self.forward(x, compute_phase=False)
        density = torch.exp(-torch.sum(state**2, dim=-1))
        return density / (torch.sum(density, dim=-1, keepdim=True) + 1e-8)

class HybridPropagator(nn.Module):
    """Symplectic neural propagator for hybrid systems."""
    
    def __init__(self,
                state_dim: int,
                hidden_dim: int = 64,
                n_layers: int = 3,
                dt: float = 0.1):
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        
        # Hamiltonian network
        self.hamiltonian = nn.Sequential(
            nn.Linear(state_dim*2, hidden_dim),
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh()) 
              for _ in range(n_layers-1)],
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize with small weights for better energy conservation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def compute_hamiltonian(self, state: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian value."""
        inputs = torch.cat([state, p], dim=-1)
        return self.hamiltonian(inputs)
    
    def forward(self, state: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symplectic time evolution."""
        # Compute gradients of Hamiltonian
        inputs = torch.cat([state, p], dim=-1)
        inputs.requires_grad_(True)
        H = self.compute_hamiltonian(state, p)
        
        grads = torch.autograd.grad(H.sum(), inputs)[0]
        dH_dq, dH_dp = grads.chunk(2, dim=-1)
        
        # Symplectic update
        new_p = p - self.dt * dH_dq
        new_state = state + self.dt * dH_dp
        
        return new_state, new_p