"""
Tests for neural network architectures.
Validates wavefunction representations and their properties.
"""

import torch
import numpy as np
import pytest
from src.neural.architectures import SimpleRBM, FFNN

def test_rbm_wavefunction():
    """Test RBM wavefunction properties"""
    # Initialize RBM
    n_visible = 4
    n_hidden = 8
    rbm = SimpleRBM(n_visible, n_hidden, complex=True)
    
    # Test with batch of random inputs
    batch_size = 10
    x = torch.randn(batch_size, n_visible, dtype=torch.cfloat)
    
    # Test output shape
    output = rbm(x)
    assert output.shape == (batch_size,)
    assert output.dtype == torch.cfloat
    
    # Test probability is real and positive
    prob = rbm.probability(x)
    assert prob.dtype == torch.float32
    assert torch.all(prob >= 0)
    
    # Test real RBM
    rbm_real = SimpleRBM(n_visible, n_hidden, complex=False)
    x_real = torch.randn(batch_size, n_visible)
    output_real = rbm_real(x_real)
    assert output_real.dtype == torch.float32

def test_ffnn_wavefunction():
    """Test feed-forward neural network wavefunction"""
    # Initialize FFNN
    n_input = 4
    hidden_dims = [16, 16]
    ffnn = FFNN(n_input, hidden_dims, complex=True, activation='Tanh')
    
    # Test with batch of random inputs
    batch_size = 10
    x = torch.randn(batch_size, n_input, dtype=torch.cfloat)
    
    # Test output shape
    output = ffnn(x)
    assert output.shape == (batch_size,)
    assert torch.is_complex(output)
    
    # Test probability is real and positive
    prob = ffnn.probability(x)
    assert not torch.is_complex(prob)
    assert torch.all(prob >= 0)
    
    # Test real network
    ffnn_real = FFNN(n_input, hidden_dims, complex=False, activation='Tanh')
    output_real = ffnn_real(x.real)
    assert not torch.is_complex(output_real)

def test_complex_operations():
    """Test complex-valued neural network operations"""
    # Initialize complex FFNN
    n_input = 4
    hidden_dims = [16, 16]
    ffnn = FFNN(n_input, hidden_dims, complex=True, activation='Tanh')
    
    # Test with complex inputs
    batch_size = 10
    x = torch.randn(batch_size, n_input, dtype=torch.cfloat)
    
    # Forward pass should preserve complex values
    output = ffnn(x)
    assert torch.is_complex(output)
    
    # Test gradients
    loss = output.abs().sum()
    loss.backward()
    for param in ffnn.parameters():
        assert param.grad is not None
    
    # Test different activations
    ffnn_relu = FFNN(n_input, hidden_dims, complex=True, activation='ReLU')
    output_relu = ffnn_relu(x)
    assert torch.is_complex(output_relu)

if __name__ == '__main__':
    pytest.main([__file__])