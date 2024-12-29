"""
Tests for physical constants and unit conversions.
Validates constant values and conversion functions.
"""

import numpy as np
import pytest
import dataclasses
from src.utils.constants import PhysicalConstants, UnitConverter

def test_physical_constants():
    """Test physical constants values and immutability"""
    constants = PhysicalConstants()
    
    # Test atomic units
    assert constants.hbar == 1.0
    assert constants.e_charge == 1.0
    assert constants.m_electron == 1.0
    
    # Test immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        constants.hbar = 2.0

def test_unit_converter_reversibility():
    """Test that unit conversions are reversible"""
    converter = UnitConverter()
    
    # Test values
    length = 1.5  # Angstrom
    energy = 2.0  # eV
    temp = 300.0  # Kelvin
    
    # Test length conversion reversibility
    assert np.isclose(
        converter.bohr_to_angstrom(converter.angstrom_to_bohr(length)),
        length
    )
    
    # Test energy conversion reversibility
    assert np.isclose(
        converter.hartree_to_ev(converter.ev_to_hartree(energy)),
        energy
    )
    
    # Test temperature conversion reversibility
    assert np.isclose(
        converter.au_to_kelvin(converter.kelvin_to_au(temp)),
        temp
    )

def test_thermal_wavelength():
    """Test thermal de Broglie wavelength calculation"""
    converter = UnitConverter()
    
    # Test at room temperature (300K)
    T_au = converter.kelvin_to_au(300.0)
    wavelength = converter.get_thermal_wavelength(T_au)
    
    # Wavelength should be positive and finite
    assert wavelength > 0
    assert np.isfinite(wavelength)
    
    # Test temperature scaling (λ ∝ 1/√T)
    T2_au = converter.kelvin_to_au(1200.0)  # 4x temperature
    wavelength2 = converter.get_thermal_wavelength(T2_au)
    
    # Should be half the wavelength (within numerical precision)
    assert np.isclose(wavelength2 * 2, wavelength)

def test_constants_to_dict():
    """Test conversion of constants to dictionary"""
    constants = PhysicalConstants()
    constants_dict = constants.to_dict()
    
    # Check all constants are present
    assert 'hbar' in constants_dict
    assert 'e_charge' in constants_dict
    assert 'm_electron' in constants_dict
    assert 'alpha' in constants_dict
    assert 'k_B' in constants_dict
    
    # Check values match
    assert constants_dict['hbar'] == constants.hbar
    assert constants_dict['alpha'] == constants.alpha
    assert constants_dict['k_B'] == constants.k_B

def test_specific_conversions():
    """Test specific known conversion values"""
    converter = UnitConverter()
    
    # Test Angstrom to Bohr conversion (1 Å ≈ 1.89 a₀)
    assert np.isclose(converter.angstrom_to_bohr(1.0), 1.8897259886)
    
    # Test eV to Hartree conversion (1 eV ≈ 0.037 Eh)
    assert np.isclose(converter.ev_to_hartree(1.0), 0.0367493224)
    
    # Test room temperature conversion (300 K)
    T_au = converter.kelvin_to_au(300.0)
    assert np.isclose(T_au, 0.0009500434689)

if __name__ == '__main__':
    pytest.main([__file__])