"""
Neural network architectures for quantum state representation.
Implements various network architectures suitable for quantum wavefunctions.
Enhanced with hybrid quantum-classical capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class QuantumClassicalConfig:
    """Configuration for quantum-classical hybrid networks"""
    n_qubits: int = 1
    n_classical: int = 1
    measurement_basis: str = 'computational'
    classical_scale: float = 1.0
    quantum_scale: float = 1.0

class WaveFunction(nn.Module):
    """Enhanced base class for neural network quantum states"""
    
    def __init__(self, config: Optional[QuantumClassicalConfig] = None):
        super().__init__()
        self.config = config or QuantumClassicalConfig()
        self._quantum_state = None
        self._classical_state = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(ψ(x))"""
        raise NotImplementedError
    
    def probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute |ψ(x)|²"""
        return torch.exp(2 * torch.real(self.forward(x)))
    
    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute phase of ψ(x)"""
        return torch.imag(self.forward(x))
    
    def to_classical(self, state: torch.Tensor) -> torch.Tensor:
        """Convert quantum state to classical representation"""
        if self.config.measurement_basis == 'computational':
            return self._computational_basis_measurement(state)
        elif self.config.measurement_basis == 'position':
            return self._position_basis_measurement(state)
        else:
            raise ValueError(f"Unknown measurement basis: {self.config.measurement_basis}")
    
    def from_classical(self, state: torch.Tensor) -> torch.Tensor:
        """Convert classical state to quantum representation"""
        if self.config.measurement_basis == 'computational':
            return self._computational_basis_encoding(state)
        elif self.config.measurement_basis == 'position':
            return self._position_basis_encoding(state)
        else:
            raise ValueError(f"Unknown measurement basis: {self.config.measurement_basis}")
    
    def _computational_basis_measurement(self, state: torch.Tensor) -> torch.Tensor:
        """Measure state in computational basis"""
        probs = self.probability(state)
        return probs * self.config.quantum_scale
    
    def _position_basis_measurement(self, state: torch.Tensor) -> torch.Tensor:
        """Measure state in position basis"""
        return torch.abs(state)**2 * self.config.quantum_scale
    
    def _computational_basis_encoding(self, state: torch.Tensor) -> torch.Tensor:
        """Encode classical state in computational basis"""
        return torch.sqrt(state / self.config.classical_scale)
    
    def _position_basis_encoding(self, state: torch.Tensor) -> torch.Tensor:
        """Encode classical state in position basis"""
        return torch.sqrt(state / self.config.classical_scale)

    def get_quantum_state(self) -> Optional[torch.Tensor]:
        """Get current quantum state"""
        return self._quantum_state
    
    def get_classical_state(self) -> Optional[torch.Tensor]:
        """Get current classical state"""
        return self._classical_state

class HybridRBM(WaveFunction):
    """
    Enhanced RBM for hybrid quantum-classical systems
    ψ(x) = exp(a·x + b·h + x·W·h)
    """
    
    def __init__(self, n_visible: int, n_hidden: int, complex: bool = True,
                 config: Optional[QuantumClassicalConfig] = None):
        super().__init__(config)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.complex = complex
        
        # Initialize parameters with hybrid support
        if complex:
            self.a_real = nn.Parameter(torch.randn(n_visible) * 0.01)
            self.a_imag = nn.Parameter(torch.randn(n_visible) * 0.01)
            self.b_real = nn.Parameter(torch.randn(n_hidden) * 0.01)
            self.b_imag = nn.Parameter(torch.randn(n_hidden) * 0.01)
            self.W_real = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
            self.W_imag = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        else:
            self.a = nn.Parameter(torch.randn(n_visible) * 0.01)
            self.b = nn.Parameter(torch.randn(n_hidden) * 0.01)
            self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        
        # Add classical interface layers
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_visible, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_visible),
            nn.Tanh()
        )
        self.classical_decoder = nn.Sequential(
            nn.Linear(n_visible, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_visible),
            nn.Softplus()
        )
    
    def _compute_free_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute free energy for gradient stability"""
        if self.complex:
            visible_term = torch.sum(x * (self.a_real + 1j * self.a_imag), dim=1)
            z = torch.matmul(x, (self.W_real + 1j * self.W_imag)) + \
                (self.b_real + 1j * self.b_imag).unsqueeze(0)
            hidden_term = torch.sum(F.softplus(z.abs()), dim=1)
            return visible_term + hidden_term
        else:
            visible_term = torch.sum(x * self.a, dim=1)
            z = torch.matmul(x, self.W) + self.b.unsqueeze(0)
            hidden_term = torch.sum(F.softplus(z), dim=1)
            return visible_term + hidden_term
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with hybrid support"""
        batch_size = x.shape[0]
        
        if self.complex:
            if not x.is_complex():
                x = x.to(torch.cfloat)
            
        # Compute quantum state using free energy formulation
        self._quantum_state = self._compute_free_energy(x)
        
        # Classical interface
        classical_input = x.abs() if self.complex else x
        self._classical_state = self.classical_decoder(classical_input)
        
        return self._quantum_state
    
    def encode_classical(self, classical_state: torch.Tensor) -> torch.Tensor:
        """Encode classical state into quantum representation"""
        return self.classical_encoder(classical_state)
    
    def decode_quantum(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Decode quantum state to classical representation"""
        if len(quantum_state.shape) == 2:  # [batch_size, n_visible]
            return self.classical_decoder(quantum_state.abs() if self.complex else quantum_state)
        else:  # [batch_size] - scalar quantum states
            batch_size = quantum_state.shape[0]
            quantum_features = (quantum_state.abs() if self.complex else quantum_state).unsqueeze(1).expand(-1, self.n_visible)
            return self.classical_decoder(quantum_features)


class HybridFFNN(WaveFunction):
    """
    Enhanced feed-forward neural network for hybrid quantum-classical systems
    """
    
    def __init__(self, 
                 n_input: int,
                 hidden_dims: List[int],
                 complex: bool = True,
                 activation: str = 'Tanh',
                 config: Optional[QuantumClassicalConfig] = None):
        super().__init__(config)
        self.complex = complex
        self.n_input = n_input
        
        # Build network layers
        layers = []
        prev_dim = n_input
        
        for dim in hidden_dims:
            if complex:
                layers.append(ComplexLinear(prev_dim, dim))
                layers.append(ComplexActivation(activation))
            else:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(getattr(nn, activation)())
            prev_dim = dim
        
        # Output layer
        if complex:
            layers.append(ComplexLinear(prev_dim, 1))
        else:
            layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Add hybrid interface layers
        if complex:
            self.quantum_interface = ComplexLinear(n_input, n_input)
        else:
            self.quantum_interface = nn.Linear(n_input, n_input)
            
        self.classical_interface = nn.Sequential(
            nn.Linear(n_input, n_input),
            nn.Tanh(),
            nn.Linear(n_input, n_input),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with hybrid support"""
        if self.complex and not x.is_complex():
            x = x.to(torch.cfloat)
        
        # Quantum transformation
        quantum_state = self.quantum_interface(x)
        self._quantum_state = self.network(quantum_state).squeeze(-1)
        
        # Classical interface
        classical_input = quantum_state.abs() if self.complex else quantum_state
        self._classical_state = self.classical_interface(classical_input)
        
        return self._quantum_state
    
    def quantum_expectation(self, operator: torch.Tensor) -> torch.Tensor:
        """Compute quantum expectation value"""
        if self._quantum_state is None:
            raise ValueError("No quantum state available. Run forward pass first.")
            
        if len(operator.shape) == 2:  # Matrix operator
            return torch.einsum('b,bc,c->b', 
                              torch.conj(self._quantum_state),
                              operator,
                              self._quantum_state)
        else:  # Vector operator
            return torch.einsum('b,b->b',
                              torch.conj(self._quantum_state),
                              operator * self._quantum_state)

class ComplexLinear(nn.Module):
    """Complex-valued linear layer with enhanced gradient handling"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)
        
        # Initialize with quantum-aware scaling
        bound = 1 / np.sqrt(in_features)
        nn.init.uniform_(self.real.weight, -bound, bound)
        nn.init.uniform_(self.imag.weight, -bound, bound)
        nn.init.zeros_(self.real.bias)
        nn.init.zeros_(self.imag.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            real = x.real
            imag = x.imag
        else:
            real = x
            imag = torch.zeros_like(x)
        
        return (self.real(real) - self.imag(imag)) + \
               1j * (self.real(imag) + self.imag(real))

class ComplexActivation(nn.Module):
    """Complex-valued activation with enhanced gradient stability"""
    
    def __init__(self, activation: str):
        super().__init__()
        activation_map = {
            'Tanh': nn.Tanh,
            'ReLU': nn.ReLU,
            'Sigmoid': nn.Sigmoid
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation_map[activation]()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return self.activation(x.real) + 1j * self.activation(x.imag)
        return self.activation(x)