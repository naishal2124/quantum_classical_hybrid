"""
Tests for hybrid neural models.
Validates quantum-classical transition behavior.
"""

import torch
import numpy as np
import pytest
from src.neural.hybrid_models import (
    QuantumClassicalNet,
    VariationalHybridState,
    HybridPropagator
)

@pytest.fixture
def quantum_classical_net():
    """Fixture for quantum-classical network."""
    return QuantumClassicalNet(
        input_dim=1,
        hidden_dim=32,
        n_layers=2,
        temperature=1.0
    )

@pytest.fixture
def variational_state():
    """Fixture for variational hybrid state."""
    return VariationalHybridState(
        state_dim=1,
        hidden_dim=32,
        n_layers=2
    )

@pytest.fixture
def hybrid_propagator():
    """Fixture for hybrid propagator."""
    return HybridPropagator(
        state_dim=1,
        hidden_dim=32,
        n_layers=2,
        dt=0.1
    )

def test_quantum_classical_shapes(quantum_classical_net):
    """Test output shapes of quantum-classical network."""
    batch_size = 10
    x = torch.randn(batch_size, 1)
    
    # Test quantum output
    psi = quantum_classical_net.quantum_forward(x)
    assert psi.shape == (batch_size, 1)
    assert torch.is_complex(psi)
    
    # Test classical output
    rho = quantum_classical_net.classical_forward(x)
    assert rho.shape == (batch_size, 1)
    assert not torch.is_complex(rho)
    
    # Test interpolated output
    p = quantum_classical_net(x)
    assert p.shape == (batch_size, 1)
    assert not torch.is_complex(p)

def test_temperature_limits(quantum_classical_net):
    """Test high and low temperature limits."""
    x = torch.randn(10, 1)
    
    # High temperature (classical) limit
    high_T = quantum_classical_net(x, beta=0.01)
    classical = quantum_classical_net.classical_forward(x)
    assert torch.allclose(high_T, classical, rtol=1e-2)
    
    # Low temperature (quantum) limit
    low_T = quantum_classical_net(x, beta=100.0)
    quantum = torch.abs(quantum_classical_net.quantum_forward(x))**2
    assert torch.allclose(low_T, quantum, rtol=1e-2)

def test_probability_normalization(quantum_classical_net):
    """Test probability normalization."""
    x = torch.linspace(-5, 5, 1000).reshape(-1, 1)
    
    # Test quantum probability
    psi = quantum_classical_net.quantum_forward(x)
    prob_q = torch.abs(psi)**2
    norm_q = torch.trapz(prob_q.squeeze(), x.squeeze())
    assert abs(norm_q - 1.0) < 0.1
    
    # Test classical probability
    prob_c = quantum_classical_net.classical_forward(x)
    norm_c = torch.trapz(prob_c.squeeze(), x.squeeze())
    assert abs(norm_c - 1.0) < 0.1

def test_variational_state_properties(variational_state):
    """Test properties of variational hybrid state."""
    x = torch.randn(10, 1)
    
    # Test quantum state
    psi = variational_state(x, compute_phase=True)
    assert torch.is_complex(psi)
    
    # Test classical state
    rho = variational_state(x, compute_phase=False)
    assert not torch.is_complex(rho)
    
    # Test probability positivity
    prob = variational_state.probability(x)
    assert torch.all(prob >= 0)

def test_hybrid_propagator_conservation(hybrid_propagator):
    """Test energy conservation properties."""
    state = torch.randn(10, 1)
    p = torch.randn(10, 1)
    
    # Propagate for several steps
    states = [state]
    momenta = [p]
    for _ in range(10):
        state, p = hybrid_propagator(state, p)
        states.append(state)
        momenta.append(p)
    
    # Check energy conservation
    states = torch.stack(states)
    momenta = torch.stack(momenta)
    kinetic = 0.5 * momenta**2
    potential = 0.5 * states**2  # Harmonic potential
    energy = kinetic + potential
    
    # Energy should be approximately conserved
    energy_std = torch.std(energy, dim=0)
    assert torch.all(energy_std < 0.1)

def test_propagator_reversibility(hybrid_propagator):
    """Test time reversibility of propagator."""
    state = torch.randn(10, 1)
    p = torch.randn(10, 1)
    
    # Forward propagation
    state_new, p_new = hybrid_propagator(state, p)
    
    # Backward propagation (negative time step)
    hybrid_propagator.dt *= -1
    state_rev, p_rev = hybrid_propagator(state_new, p_new)
    hybrid_propagator.dt *= -1  # Reset dt
    
    # Check reversibility
    assert torch.allclose(state, state_rev, rtol=1e-3)
    assert torch.allclose(p, p_rev, rtol=1e-3)

if __name__ == '__main__':
    pytest.main([__file__]) 
