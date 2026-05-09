"""
SI base units and common derived units.
"""

from .core import Unit, get_or_create_dimension, DIMENSIONLESS

__all__ = [
    'second', 'metre', 'meter', 'kilogram', 'ampere', 'amp', 'kelvin', 'mole', 'candela',
    'volt', 'ohm', 'siemens', 'farad', 'coulomb', 'hertz', 'watt', 'joule', 'pascal', 'newton',
]

# Create dimensions for SI base units
_dim_length = get_or_create_dimension(length=1)
_dim_mass = get_or_create_dimension(mass=1)
_dim_time = get_or_create_dimension(time=1)
_dim_current = get_or_create_dimension(current=1)
_dim_temperature = get_or_create_dimension(temperature=1)
_dim_substance = get_or_create_dimension(substance=1)
_dim_luminosity = get_or_create_dimension(luminosity=1)

# SI base units
second = Unit.create(_dim_time, 'second', 's')
metre = Unit.create(_dim_length, 'metre', 'm')
meter = metre  # American spelling
kilogram = Unit.create(_dim_mass, 'kilogram', 'kg')
ampere = Unit.create(_dim_current, 'ampere', 'A')
amp = ampere  # Alias
kelvin = Unit.create(_dim_temperature, 'kelvin', 'K')
mole = Unit.create(_dim_substance, 'mole', 'mol')
candela = Unit.create(_dim_luminosity, 'candela', 'cd')

# Derived units with special names
# Voltage: V = kg*m^2/(A*s^3) = W/A = J/C
_dim_voltage = get_or_create_dimension(mass=1, length=2, time=-3, current=-1)
volt = Unit.create(_dim_voltage, 'volt', 'V')

# Resistance: ohm = V/A = kg*m^2/(A^2*s^3)
_dim_resistance = get_or_create_dimension(mass=1, length=2, time=-3, current=-2)
ohm = Unit.create(_dim_resistance, 'ohm', 'ohm')

# Conductance: S = A/V = A^2*s^3/(kg*m^2)
_dim_conductance = get_or_create_dimension(mass=-1, length=-2, time=3, current=2)
siemens = Unit.create(_dim_conductance, 'siemens', 'S')

# Capacitance: F = C/V = A^2*s^4/(kg*m^2)
_dim_capacitance = get_or_create_dimension(mass=-1, length=-2, time=4, current=2)
farad = Unit.create(_dim_capacitance, 'farad', 'F')

# Charge: C = A*s
_dim_charge = get_or_create_dimension(current=1, time=1)
coulomb = Unit.create(_dim_charge, 'coulomb', 'C')

# Frequency: Hz = 1/s
_dim_frequency = get_or_create_dimension(time=-1)
hertz = Unit.create(_dim_frequency, 'hertz', 'Hz')

# Power: W = V*A = kg*m^2/s^3
_dim_power = get_or_create_dimension(mass=1, length=2, time=-3)
watt = Unit.create(_dim_power, 'watt', 'W')

# Energy: J = W*s = kg*m^2/s^2
_dim_energy = get_or_create_dimension(mass=1, length=2, time=-2)
joule = Unit.create(_dim_energy, 'joule', 'J')

# Pressure: Pa = N/m^2 = kg/(m*s^2)
_dim_pressure = get_or_create_dimension(mass=1, length=-1, time=-2)
pascal = Unit.create(_dim_pressure, 'pascal', 'Pa')

# Force: N = kg*m/s^2
_dim_force = get_or_create_dimension(mass=1, length=1, time=-2)
newton = Unit.create(_dim_force, 'newton', 'N')
