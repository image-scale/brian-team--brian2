"""
Physical units system for neural simulations.
"""

from .core import (
    Dimension,
    DimensionMismatchError,
    Quantity,
    Unit,
    DIMENSIONLESS,
    get_or_create_dimension,
    get_dimensions,
    is_dimensionless,
    have_same_dimensions,
    check_units,
)

from .baseunits import (
    second, metre, meter, kilogram, ampere, amp, kelvin, mole, candela,
    volt, ohm, siemens, farad, coulomb, hertz, watt, joule, pascal, newton,
)

from .stdunits import (
    # Time
    psecond, nsecond, usecond, msecond, ksecond,
    ps, ns, us, ms,
    # Length
    pmetre, nmetre, umetre, mmetre, cmetre, kmetre,
    pmeter, nmeter, umeter, mmeter, cmeter, kmeter,
    pm, nm, um, mm, cm, km,
    # Voltage
    pvolt, nvolt, uvolt, mvolt, kvolt, Mvolt,
    pV, nV, uV, mV, kV,
    # Current
    pamp, namp, uamp, mamp, kamp,
    pampere, nampere, uampere, mampere, kampere,
    pA, nA, uA, mA, kA,
    # Conductance
    psiemens, nsiemens, usiemens, msiemens,
    pS, nS, uS, mS,
    # Resistance
    pohm, nohm, uohm, mohm, kohm, Mohm, Gohm,
    # Capacitance
    pfarad, nfarad, ufarad, mfarad,
    pF, nF, uF, mF,
    # Frequency
    mhertz, khertz, Mhertz, Ghertz,
    mHz, kHz, MHz, GHz, Hz,
    # Charge
    pcoulomb, ncoulomb, ucoulomb, mcoulomb,
    pC, nC, uC, mC,
    # Mass
    pgram, ngram, ugram, mgram, gram, kgram,
    # Power
    pwatt, nwatt, uwatt, mwatt, kwatt, Mwatt,
    pW, nW, uW, mW, kW, MW,
    # Energy
    pjoule, njoule, ujoule, mjoule, kjoule, Mjoule,
    pJ, nJ, uJ, mJ, kJ, MJ,
    # Area
    metre2, meter2, cm2, mm2, um2,
    # Volume
    metre3, meter3, cm3, mm3, um3,
)

from .constants import (
    e, q_e, electron_charge,
    k_B, kB, boltzmann,
    N_A, avogadro,
    R, gas_constant,
    F, faraday_constant,
    epsilon_0, electric_constant,
    c, speed_of_light,
    h, planck,
)

__all__ = [
    # Core
    'Dimension',
    'DimensionMismatchError',
    'Quantity',
    'Unit',
    'DIMENSIONLESS',
    'get_or_create_dimension',
    'get_dimensions',
    'is_dimensionless',
    'have_same_dimensions',
    'check_units',
    # Base units
    'second', 'metre', 'meter', 'kilogram', 'ampere', 'amp', 'kelvin', 'mole', 'candela',
    'volt', 'ohm', 'siemens', 'farad', 'coulomb', 'hertz', 'watt', 'joule', 'pascal', 'newton',
    # Standard units with prefixes
    'psecond', 'nsecond', 'usecond', 'msecond', 'ksecond',
    'ps', 'ns', 'us', 'ms',
    'pmetre', 'nmetre', 'umetre', 'mmetre', 'cmetre', 'kmetre',
    'pmeter', 'nmeter', 'umeter', 'mmeter', 'cmeter', 'kmeter',
    'pm', 'nm', 'um', 'mm', 'cm', 'km',
    'pvolt', 'nvolt', 'uvolt', 'mvolt', 'kvolt', 'Mvolt',
    'pV', 'nV', 'uV', 'mV', 'kV',
    'pamp', 'namp', 'uamp', 'mamp', 'kamp',
    'pampere', 'nampere', 'uampere', 'mampere', 'kampere',
    'pA', 'nA', 'uA', 'mA', 'kA',
    'psiemens', 'nsiemens', 'usiemens', 'msiemens',
    'pS', 'nS', 'uS', 'mS',
    'pohm', 'nohm', 'uohm', 'mohm', 'kohm', 'Mohm', 'Gohm',
    'pfarad', 'nfarad', 'ufarad', 'mfarad',
    'pF', 'nF', 'uF', 'mF',
    'mhertz', 'khertz', 'Mhertz', 'Ghertz',
    'mHz', 'kHz', 'MHz', 'GHz', 'Hz',
    'pcoulomb', 'ncoulomb', 'ucoulomb', 'mcoulomb',
    'pC', 'nC', 'uC', 'mC',
    'pgram', 'ngram', 'ugram', 'mgram', 'gram', 'kgram',
    'pwatt', 'nwatt', 'uwatt', 'mwatt', 'kwatt', 'Mwatt',
    'pW', 'nW', 'uW', 'mW', 'kW', 'MW',
    'pjoule', 'njoule', 'ujoule', 'mjoule', 'kjoule', 'Mjoule',
    'pJ', 'nJ', 'uJ', 'mJ', 'kJ', 'MJ',
    'metre2', 'meter2', 'cm2', 'mm2', 'um2',
    'metre3', 'meter3', 'cm3', 'mm3', 'um3',
    # Constants
    'e', 'q_e', 'electron_charge',
    'k_B', 'kB', 'boltzmann',
    'N_A', 'avogadro',
    'R', 'gas_constant',
    'F', 'faraday_constant',
    'epsilon_0', 'electric_constant',
    'c', 'speed_of_light',
    'h', 'planck',
]
