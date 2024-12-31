"""
Tests for enhanced neural network architectures.
Validates hybrid quantum-classical functionality.
"""

import torch
import numpy as np
import pytest
from src.neural.architectures import (
    QuantumClassicalConfig,
    HybridRBM,
    HybridFFNN,
    ComplexLinear,
    ComplexActivation
)

@pytest.fixture
def quantum_classical_config():
    """Fixture for quantum-classical configuration"""
    return QuantumClassicalConfig(
        n_qubits=2,
        n_classical=2,
        measurement_basis='computational',
        classical_scale=1.0,
        quantum_scale=1.0
    )

def test_hybrid_rbm_initialization():
    """Test hybrid RBM initialization and basic properties"""
    n_visible = 4
    n_hidden = 8
    config = QuantumClassicalConfig()
    
    # Test complex RBM
    rbm = HybridRBM(n_visible, n_hidden, complex=True, config=config)
    assert hasattr(rbm, 'classical_encoder')
    assert hasattr(rbm, 'classical_decoder')
    assert rbm.complex
    
    # Test real RBM
    rbm_real = HybridRBM(n_visible, n_hidden, complex=False, config=config)
    assert hasattr(rbm_real, 'a')
    assert hasattr(rbm_real, 'b')
    assert hasattr(rbm_real, 'W')
    assert not rbm_real.complex

def test_hybrid_rbm_forward():
    """Test hybrid RBM forward pass"""
    n_visible = 4
    n_hidden = 8
    config = QuantumClassicalConfig()
    rbm = HybridRBM(n_visible, n_hidden, complex=True, config=config)
    
    # Test with batch of random inputs
    batch_size = 10
    x = torch.randn(batch_size, n_visible, dtype=torch.cfloat)
    
    # Forward pass
    output = rbm(x)
    assert output.shape == (batch_size,)
    assert output.dtype == torch.cfloat
    
    # Check states are stored
    assert rbm.get_quantum_state() is not None
    assert rbm.get_classical_state() is not None

def test_hybrid_ffnn_initialization():
    """Test hybrid FFNN initialization and basic properties"""
    n_input = 4
    hidden_dims = [16, 16]
    config = QuantumClassicalConfig()
    
    # Test complex FFNN
    ffnn = HybridFFNN(n_input, hidden_dims, complex=True, 
                      activation='Tanh', config=config)
    assert hasattr(ffnn, 'quantum_interface')
    assert hasattr(ffnn, 'classical_interface')
    assert ffnn.complex
    
    # Test real FFNN
    ffnn_real = HybridFFNN(n_input, hidden_dims, complex=False,
                          activation='Tanh', config=config)
    assert not ffnn_real.complex

def test_hybrid_ffnn_forward():
    """Test hybrid FFNN forward pass"""
    n_input = 4
    hidden_dims = [16, 16]
    config = QuantumClassicalConfig()
    ffnn = HybridFFNN(n_input, hidden_dims, complex=True, 
                      activation='Tanh', config=config)
    
    # Test with batch of random inputs
    batch_size = 10
    x = torch.randn(batch_size, n_input, dtype=torch.cfloat)
    
    # Forward pass
    output = ffnn(x)
    assert output.shape == (batch_size,)
    assert output.dtype == torch.cfloat
    
    # Test quantum expectation
    operator = torch.eye(batch_size, dtype=torch.cfloat)
    expectation = ffnn.quantum_expectation(operator)
    assert expectation.shape == (batch_size,)

def test_complex_linear():
    """Test complex linear layer"""
    in_features = 4
    out_features = 8
    layer = ComplexLinear(in_features, out_features)
    
    # Test with complex input
    batch_size = 10
    x = torch.randn(batch_size, in_features, dtype=torch.cfloat)
    output = layer(x)
    assert output.shape == (batch_size, out_features)
    assert output.dtype == torch.cfloat
    
    # Test with real input
    x_real = torch.randn(batch_size, in_features)
    output_real = layer(x_real)
    assert output_real.shape == (batch_size, out_features)
    assert output_real.dtype == torch.cfloat

def test_complex_activation():
    """Test complex activation functions"""
    activation = ComplexActivation('Tanh')
    
    # Test with complex input
    batch_size = 10
    features = 4
    x = torch.randn(batch_size, features, dtype=torch.cfloat)
    output = activation(x)
    assert output.shape == (batch_size, features)
    assert output.dtype == torch.cfloat
    
    # Test with real input
    x_real = torch.randn(batch_size, features)
    output_real = activation(x_real)
    assert output_real.shape == (batch_size, features)
    assert not torch.is_complex(output_real)

def test_quantum_classical_conversion():
    """Test quantum to classical state conversion"""
    n_visible = 4
    n_hidden = 8
    config = QuantumClassicalConfig()
    rbm = HybridRBM(n_visible, n_hidden, complex=True, config=config)
    
    # Test classical encoding
    classical_state = torch.randn(10, n_visible)
    quantum_state = rbm.encode_classical(classical_state)
    assert quantum_state.shape == classical_state.shape
    
    # Test quantum decoding
    quantum_state = torch.randn(10, n_visible, dtype=torch.cfloat)
    classical_state = rbm.decode_quantum(quantum_state)
    assert classical_state.shape == (10, n_visible)
    assert not torch.is_complex(classical_state)

def test_measurement_bases():
    """Test different measurement bases"""
    config = QuantumClassicalConfig(measurement_basis='position')
    n_visible = 4
    n_hidden = 8
    rbm = HybridRBM(n_visible, n_hidden, complex=True, config=config)
    
    # Test position basis measurement
    state = torch.randn(10, n_visible, dtype=torch.cfloat)
    measured = rbm.to_classical(state)
    assert measured.shape == (10, n_visible)
    assert not torch.is_complex(measured)
    assert torch.all(measured >= 0)  # Probabilities should be non-negative
    
    # Test invalid basis
    with pytest.raises(ValueError):
        config_invalid = QuantumClassicalConfig(measurement_basis='invalid')
        rbm = HybridRBM(n_visible, n_hidden, complex=True, config=config_invalid)
        rbm.to_classical(state)

def test_gradient_flow():
    """Test gradient flow through hybrid networks"""
    n_visible = 4
    n_hidden = 8
    config = QuantumClassicalConfig()
    rbm = HybridRBM(n_visible, n_hidden, complex=True, config=config)
    
    # Forward pass with gradient tracking
    x = torch.randn(10, n_visible, dtype=torch.cfloat, requires_grad=True)
    output = rbm(x)
    loss = output.abs().sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    for param in rbm.parameters():
        assert param.grad is not None

def test_state_consistency():
    """Test consistency between quantum and classical states"""
    n_input = 4
    hidden_dims = [16, 16]
    config = QuantumClassicalConfig()
    ffnn = HybridFFNN(n_input, hidden_dims, complex=True, 
                      activation='Tanh', config=config)
    
    # Initial state
    x = torch.randn(10, n_input, dtype=torch.cfloat)
    
    # Forward pass
    output = ffnn(x)
    
    # Check states
    quantum_state = ffnn.get_quantum_state()
    classical_state = ffnn.get_classical_state()
    
    assert quantum_state is not None
    assert classical_state is not None
    assert quantum_state.shape[0] == classical_state.shape[0]  # Batch size matches

if __name__ == '__main__':
    pytest.main([__file__])