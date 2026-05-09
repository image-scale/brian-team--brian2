"""
Standard units with SI prefixes commonly used in neuroscience.
"""

from .core import Unit, SI_PREFIXES
from .baseunits import (
    second, metre, meter, kilogram, ampere, amp, kelvin, mole, candela,
    volt, ohm, siemens, farad, coulomb, hertz, watt, joule, pascal, newton,
)

__all__ = []

def _create_scaled_units(baseunit, prefixes=None):
    """Create scaled versions of a base unit with SI prefixes."""
    if prefixes is None:
        prefixes = ['p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']

    units = {}
    for prefix in prefixes:
        if prefix == '':
            continue
        units[prefix] = Unit.create_scaled(baseunit, prefix)
    return units

# Time units
psecond = Unit.create_scaled(second, 'p')
nsecond = Unit.create_scaled(second, 'n')
usecond = Unit.create_scaled(second, 'u')
msecond = Unit.create_scaled(second, 'm')
ksecond = Unit.create_scaled(second, 'k')

# Short aliases for time
ps = psecond
ns = nsecond
us = usecond
ms = msecond

# Length units
pmetre = Unit.create_scaled(metre, 'p')
nmetre = Unit.create_scaled(metre, 'n')
umetre = Unit.create_scaled(metre, 'u')
mmetre = Unit.create_scaled(metre, 'm')
cmetre = Unit.create_scaled(metre, 'c')
kmetre = Unit.create_scaled(metre, 'k')

pmeter = pmetre
nmeter = nmetre
umeter = umetre
mmeter = mmetre
cmeter = cmetre
kmeter = kmetre

# Short aliases for length
pm = pmetre
nm = nmetre
um = umetre
mm = mmetre
cm = cmetre
km = kmetre

# Voltage units
pvolt = Unit.create_scaled(volt, 'p')
nvolt = Unit.create_scaled(volt, 'n')
uvolt = Unit.create_scaled(volt, 'u')
mvolt = Unit.create_scaled(volt, 'm')
kvolt = Unit.create_scaled(volt, 'k')
Mvolt = Unit.create_scaled(volt, 'M')

# Short aliases for voltage
pV = pvolt
nV = nvolt
uV = uvolt
mV = mvolt
kV = kvolt

# Current units
pamp = Unit.create_scaled(ampere, 'p')
namp = Unit.create_scaled(ampere, 'n')
uamp = Unit.create_scaled(ampere, 'u')
mamp = Unit.create_scaled(ampere, 'm')
kamp = Unit.create_scaled(ampere, 'k')

pampere = pamp
nampere = namp
uampere = uamp
mampere = mamp
kampere = kamp

# Short aliases for current
pA = pamp
nA = namp
uA = uamp
mA = mamp
kA = kamp

# Conductance units (siemens)
psiemens = Unit.create_scaled(siemens, 'p')
nsiemens = Unit.create_scaled(siemens, 'n')
usiemens = Unit.create_scaled(siemens, 'u')
msiemens = Unit.create_scaled(siemens, 'm')

pS = psiemens
nS = nsiemens
uS = usiemens
mS = msiemens

# Resistance units (ohm)
pohm = Unit.create_scaled(ohm, 'p')
nohm = Unit.create_scaled(ohm, 'n')
uohm = Unit.create_scaled(ohm, 'u')
mohm = Unit.create_scaled(ohm, 'm')
kohm = Unit.create_scaled(ohm, 'k')
Mohm = Unit.create_scaled(ohm, 'M')
Gohm = Unit.create_scaled(ohm, 'G')

# Capacitance units (farad)
pfarad = Unit.create_scaled(farad, 'p')
nfarad = Unit.create_scaled(farad, 'n')
ufarad = Unit.create_scaled(farad, 'u')
mfarad = Unit.create_scaled(farad, 'm')

pF = pfarad
nF = nfarad
uF = ufarad
mF = mfarad

# Frequency units (hertz)
mhertz = Unit.create_scaled(hertz, 'm')
khertz = Unit.create_scaled(hertz, 'k')
Mhertz = Unit.create_scaled(hertz, 'M')
Ghertz = Unit.create_scaled(hertz, 'G')

mHz = mhertz
kHz = khertz
MHz = Mhertz
GHz = Ghertz
Hz = hertz

# Charge units (coulomb)
pcoulomb = Unit.create_scaled(coulomb, 'p')
ncoulomb = Unit.create_scaled(coulomb, 'n')
ucoulomb = Unit.create_scaled(coulomb, 'u')
mcoulomb = Unit.create_scaled(coulomb, 'm')

pC = pcoulomb
nC = ncoulomb
uC = ucoulomb
mC = mcoulomb

# Mass units
pgram = Unit.create_scaled(kilogram, 'p')  # Actually smaller than kg
ngram = Unit.create_scaled(kilogram, 'n')
ugram = Unit.create_scaled(kilogram, 'u')
mgram = Unit.create_scaled(kilogram, 'm')
gram = Unit(0.001, dim=kilogram.dim, scale=-3, name='gram', dispname='g')
kgram = kilogram

# Power units
pwatt = Unit.create_scaled(watt, 'p')
nwatt = Unit.create_scaled(watt, 'n')
uwatt = Unit.create_scaled(watt, 'u')
mwatt = Unit.create_scaled(watt, 'm')
kwatt = Unit.create_scaled(watt, 'k')
Mwatt = Unit.create_scaled(watt, 'M')

pW = pwatt
nW = nwatt
uW = uwatt
mW = mwatt
kW = kwatt
MW = Mwatt

# Energy units
pjoule = Unit.create_scaled(joule, 'p')
njoule = Unit.create_scaled(joule, 'n')
ujoule = Unit.create_scaled(joule, 'u')
mjoule = Unit.create_scaled(joule, 'm')
kjoule = Unit.create_scaled(joule, 'k')
Mjoule = Unit.create_scaled(joule, 'M')

pJ = pjoule
nJ = njoule
uJ = ujoule
mJ = mjoule
kJ = kjoule
MJ = Mjoule

# Area units (metre^2)
metre2 = metre ** 2
meter2 = metre2
cm2 = cmetre ** 2
mm2 = mmetre ** 2
um2 = umetre ** 2

# Volume units (metre^3)
metre3 = metre ** 3
meter3 = metre3
cm3 = cmetre ** 3
mm3 = mmetre ** 3
um3 = umetre ** 3

# Build __all__ list
__all__ = [
    # Time
    'psecond', 'nsecond', 'usecond', 'msecond', 'ksecond',
    'ps', 'ns', 'us', 'ms',
    # Length
    'pmetre', 'nmetre', 'umetre', 'mmetre', 'cmetre', 'kmetre',
    'pmeter', 'nmeter', 'umeter', 'mmeter', 'cmeter', 'kmeter',
    'pm', 'nm', 'um', 'mm', 'cm', 'km',
    # Voltage
    'pvolt', 'nvolt', 'uvolt', 'mvolt', 'kvolt', 'Mvolt',
    'pV', 'nV', 'uV', 'mV', 'kV',
    # Current
    'pamp', 'namp', 'uamp', 'mamp', 'kamp',
    'pampere', 'nampere', 'uampere', 'mampere', 'kampere',
    'pA', 'nA', 'uA', 'mA', 'kA',
    # Conductance
    'psiemens', 'nsiemens', 'usiemens', 'msiemens',
    'pS', 'nS', 'uS', 'mS',
    # Resistance
    'pohm', 'nohm', 'uohm', 'mohm', 'kohm', 'Mohm', 'Gohm',
    # Capacitance
    'pfarad', 'nfarad', 'ufarad', 'mfarad',
    'pF', 'nF', 'uF', 'mF',
    # Frequency
    'mhertz', 'khertz', 'Mhertz', 'Ghertz',
    'mHz', 'kHz', 'MHz', 'GHz', 'Hz',
    # Charge
    'pcoulomb', 'ncoulomb', 'ucoulomb', 'mcoulomb',
    'pC', 'nC', 'uC', 'mC',
    # Mass
    'pgram', 'ngram', 'ugram', 'mgram', 'gram', 'kgram',
    # Power
    'pwatt', 'nwatt', 'uwatt', 'mwatt', 'kwatt', 'Mwatt',
    'pW', 'nW', 'uW', 'mW', 'kW', 'MW',
    # Energy
    'pjoule', 'njoule', 'ujoule', 'mjoule', 'kjoule', 'Mjoule',
    'pJ', 'nJ', 'uJ', 'mJ', 'kJ', 'MJ',
    # Area
    'metre2', 'meter2', 'cm2', 'mm2', 'um2',
    # Volume
    'metre3', 'meter3', 'cm3', 'mm3', 'um3',
]
