"""
Neural network architectures for quantum state representation.
Implements various network architectures suitable for quantum wavefunctions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class WaveFunction(nn.Module):
    """Base class for neural network quantum states"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(ψ(x))"""
        raise NotImplementedError
    
    def probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute |ψ(x)|²"""
        return torch.exp(2 * torch.real(self.forward(x)))
    
    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute phase of ψ(x)"""
        return torch.imag(self.forward(x))

class SimpleRBM(WaveFunction):
    """
    Restricted Boltzmann Machine wavefunction
    ψ(x) = exp(a·x + b·h + x·W·h)
    """
    
    def __init__(self, n_visible: int, n_hidden: int, complex: bool = True):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.complex = complex
        
        # Visible bias
        if complex:
            self.a_real = nn.Parameter(torch.randn(n_visible) * 0.01)
            self.a_imag = nn.Parameter(torch.randn(n_visible) * 0.01)
        else:
            self.a = nn.Parameter(torch.randn(n_visible) * 0.01)
        
        # Hidden bias
        if complex:
            self.b_real = nn.Parameter(torch.randn(n_hidden) * 0.01)
            self.b_imag = nn.Parameter(torch.randn(n_hidden) * 0.01)
        else:
            self.b = nn.Parameter(torch.randn(n_hidden) * 0.01)
        
        # Weights
        if complex:
            self.W_real = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
            self.W_imag = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        else:
            self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log(ψ(x))
        Args:
            x: Input configurations (batch_size, n_visible)
        Returns:
            log(ψ(x)) (batch_size,)
        """
        if self.complex:
            # Convert input to complex if needed
            if not x.is_complex():
                x = x.to(torch.cfloat)
            
            # Compute visible term
            visible_term = torch.matmul(x, (self.a_real + 1j * self.a_imag))
            
            # Compute hidden term
            z = torch.matmul(x, (self.W_real + 1j * self.W_imag)) + \
                (self.b_real + 1j * self.b_imag)
            
            hidden_term = torch.sum(
                torch.log(1 + torch.exp(z)), dim=1
            )
            
            return visible_term + hidden_term
        else:
            visible_term = torch.matmul(x, self.a)
            hidden_term = torch.sum(
                torch.log(1 + torch.exp(torch.matmul(x, self.W) + self.b)), 
                dim=1
            )
            return visible_term + hidden_term

class FFNN(WaveFunction):
    """
    Feed-forward neural network wavefunction
    More flexible architecture for quantum states
    """
    
    def __init__(self, 
                 n_input: int,
                 hidden_dims: List[int],
                 complex: bool = True,
                 activation: str = 'Tanh'):
        super().__init__()
        self.complex = complex
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log(ψ(x))
        Args:
            x: Input configurations (batch_size, n_input)
        Returns:
            log(ψ(x)) (batch_size,)
        """
        if self.complex and not x.is_complex():
            x = x.to(torch.cfloat)
        return self.network(x).squeeze(-1)

class ComplexLinear(nn.Module):
    """Complex-valued linear layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)
    
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
    """Complex-valued activation function"""
    
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