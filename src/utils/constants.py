"""
Physical constants and unit conversions for quantum-classical simulations.
Implements standard physical constants in atomic units and common unit conversions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants in atomic units (ℏ = e = me = 1)"""
    # Fundamental constants
    hbar: float = 1.0
    e_charge: float = 1.0
    m_electron: float = 1.0
    
    # Derived constants
    alpha: float = 0.0072973525693  # Fine structure constant
    k_B: float = 3.166811563e-6     # Boltzmann constant in a.u.
    
    # Unit conversions
    angstrom_to_bohr: float = 1.8897259886
    ev_to_hartree: float = 0.0367493224
    kelvin_to_au: float = 3.166811563e-6
    
    def to_dict(self) -> Dict[str, float]:
        """Convert constants to dictionary"""
        return {
            'hbar': self.hbar,
            'e_charge': self.e_charge,
            'm_electron': self.m_electron,
            'alpha': self.alpha,
            'k_B': self.k_B,
            'angstrom_to_bohr': self.angstrom_to_bohr,
            'ev_to_hartree': self.ev_to_hartree,
            'kelvin_to_au': self.kelvin_to_au
        }

class UnitConverter:
    """Handles unit conversions between different systems"""
    
    def __init__(self, constants: Optional[PhysicalConstants] = None):
        self.constants = constants or PhysicalConstants()
    
    def angstrom_to_bohr(self, length: float) -> float:
        """Convert length from Angstroms to Bohr radii"""
        return length * self.constants.angstrom_to_bohr
    
    def bohr_to_angstrom(self, length: float) -> float:
        """Convert length from Bohr radii to Angstroms"""
        return length / self.constants.angstrom_to_bohr
    
    def ev_to_hartree(self, energy: float) -> float:
        """Convert energy from eV to Hartree"""
        return energy * self.constants.ev_to_hartree
    
    def hartree_to_ev(self, energy: float) -> float:
        """Convert energy from Hartree to eV"""
        return energy / self.constants.ev_to_hartree
    
    def kelvin_to_au(self, temperature: float) -> float:
        """Convert temperature from Kelvin to atomic units"""
        return temperature * self.constants.kelvin_to_au
    
    def au_to_kelvin(self, temperature: float) -> float:
        """Convert temperature from atomic units to Kelvin"""
        return temperature / self.constants.kelvin_to_au
    
    def get_thermal_wavelength(self, temperature: float, mass: float = 1.0) -> float:
        """
        Calculate thermal de Broglie wavelength λ = h/√(2πmkT)
        Args:
            temperature: Temperature in atomic units
            mass: Particle mass in atomic units (default: electron mass)
        Returns:
            Thermal wavelength in atomic units
        """
        return np.sqrt(2 * np.pi * self.constants.hbar**2 / 
                      (mass * self.constants.k_B * temperature))