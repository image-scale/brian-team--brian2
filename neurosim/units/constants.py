"""
Physical constants commonly used in neuroscience and physics.
"""

from .core import Quantity, get_or_create_dimension
from .baseunits import second, metre, kilogram, ampere, kelvin, mole, coulomb, joule

__all__ = [
    'e', 'q_e', 'electron_charge',
    'k_B', 'kB', 'boltzmann',
    'N_A', 'avogadro',
    'R', 'gas_constant',
    'F', 'faraday_constant',
    'epsilon_0', 'electric_constant',
    'c', 'speed_of_light',
    'h', 'planck',
]

# Elementary charge (charge of electron/proton)
e = Quantity(1.602176634e-19, dim=coulomb.dim)
q_e = e
electron_charge = e

# Boltzmann constant
_dim_energy_per_kelvin = get_or_create_dimension(mass=1, length=2, time=-2, temperature=-1)
k_B = Quantity(1.380649e-23, dim=_dim_energy_per_kelvin)
kB = k_B
boltzmann = k_B

# Avogadro constant
_dim_per_mole = get_or_create_dimension(substance=-1)
N_A = Quantity(6.02214076e23, dim=_dim_per_mole)
avogadro = N_A

# Gas constant R = N_A * k_B
_dim_energy_per_mole_kelvin = get_or_create_dimension(mass=1, length=2, time=-2, substance=-1, temperature=-1)
R = Quantity(8.314462618, dim=_dim_energy_per_mole_kelvin)
gas_constant = R

# Faraday constant F = N_A * e
_dim_charge_per_mole = get_or_create_dimension(current=1, time=1, substance=-1)
F = Quantity(96485.33212, dim=_dim_charge_per_mole)
faraday_constant = F

# Electric constant (vacuum permittivity)
_dim_permittivity = get_or_create_dimension(mass=-1, length=-3, time=4, current=2)
epsilon_0 = Quantity(8.8541878128e-12, dim=_dim_permittivity)
electric_constant = epsilon_0

# Speed of light
_dim_velocity = get_or_create_dimension(length=1, time=-1)
c = Quantity(299792458, dim=_dim_velocity)
speed_of_light = c

# Planck constant
_dim_action = get_or_create_dimension(mass=1, length=2, time=-1)
h = Quantity(6.62607015e-34, dim=_dim_action)
planck = h
