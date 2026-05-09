"""
Tests for standard units with SI prefixes and physical constants.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from neurosim.units import (
    # Base units
    second, volt, ampere, metre, siemens, farad, hertz, ohm, coulomb,
    # Standard prefixed units
    msecond, usecond, nsecond, psecond,
    ms, us, ns, ps,
    mvolt, uvolt, nvolt,
    mV, uV, nV,
    namp, uamp, mamp, pamp,
    nA, uA, mA, pA,
    nsiemens, usiemens, msiemens,
    nS, uS, mS,
    kohm, Mohm, Gohm,
    pfarad, nfarad, ufarad,
    pF, nF, uF,
    khertz, Mhertz, Ghertz,
    kHz, MHz, GHz, Hz,
    # Length
    cmetre, mmetre, umetre, nmetre, kmetre,
    cm, mm, um, nm, km,
    cm2, mm2, um2,
    # Constants
    e, q_e, electron_charge,
    k_B, kB, boltzmann,
    N_A, avogadro,
    F, faraday_constant,
)


class TestTimeUnits:
    """Tests for time units with SI prefixes."""

    def test_millisecond_to_second(self):
        """1000 milliseconds equals 1 second."""
        result = 1000 * msecond / second
        assert_allclose(result, 1.0)

    def test_microsecond_to_second(self):
        """1e6 microseconds equals 1 second."""
        result = 1e6 * usecond / second
        assert_allclose(result, 1.0)

    def test_nanosecond_to_second(self):
        """1e9 nanoseconds equals 1 second."""
        result = 1e9 * nsecond / second
        assert_allclose(result, 1.0)

    def test_picosecond_to_second(self):
        """1e12 picoseconds equals 1 second."""
        result = 1e12 * psecond / second
        assert_allclose(result, 1.0)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert ms is msecond
        assert us is usecond
        assert ns is nsecond
        assert ps is psecond


class TestVoltageUnits:
    """Tests for voltage units."""

    def test_millivolt_to_volt(self):
        """1000 millivolts equals 1 volt."""
        result = 1000 * mvolt / volt
        assert_allclose(result, 1.0)

    def test_microvolt_to_volt(self):
        """1e6 microvolts equals 1 volt."""
        result = 1e6 * uvolt / volt
        assert_allclose(result, 1.0)

    def test_single_millivolt(self):
        """1 mV equals 0.001 V."""
        result = float(1 * mvolt / volt)
        assert_allclose(result, 0.001)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert mV is mvolt
        assert uV is uvolt
        assert nV is nvolt


class TestCurrentUnits:
    """Tests for current units."""

    def test_nanoamp_to_amp(self):
        """1e9 nanoamps equals 1 ampere."""
        result = 1e9 * namp / ampere
        assert_allclose(result, 1.0)

    def test_microamp_to_amp(self):
        """1e6 microamps equals 1 ampere."""
        result = 1e6 * uamp / ampere
        assert_allclose(result, 1.0)

    def test_milliamp_to_amp(self):
        """1000 milliamps equals 1 ampere."""
        result = 1000 * mamp / ampere
        assert_allclose(result, 1.0)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert nA is namp
        assert uA is uamp
        assert mA is mamp
        assert pA is pamp


class TestConductanceUnits:
    """Tests for conductance units (siemens)."""

    def test_nanosiemens_to_siemens(self):
        """1e9 nanosiemens equals 1 siemens."""
        result = 1e9 * nsiemens / siemens
        assert_allclose(result, 1.0)

    def test_microsiemens_to_siemens(self):
        """1e6 microsiemens equals 1 siemens."""
        result = 1e6 * usiemens / siemens
        assert_allclose(result, 1.0)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert nS is nsiemens
        assert uS is usiemens
        assert mS is msiemens


class TestResistanceUnits:
    """Tests for resistance units (ohm)."""

    def test_kilohm_to_ohm(self):
        """1 kilohm equals 1000 ohm."""
        result = float(kohm / ohm)
        assert_allclose(result, 1000.0)

    def test_megaohm_to_ohm(self):
        """1 megaohm equals 1e6 ohm."""
        result = float(Mohm / ohm)
        assert_allclose(result, 1e6)

    def test_gigaohm_to_ohm(self):
        """1 gigaohm equals 1e9 ohm."""
        result = float(Gohm / ohm)
        assert_allclose(result, 1e9)


class TestCapacitanceUnits:
    """Tests for capacitance units (farad)."""

    def test_picofarad_to_farad(self):
        """1e12 picofarads equals 1 farad."""
        result = 1e12 * pfarad / farad
        assert_allclose(result, 1.0)

    def test_nanofarad_to_farad(self):
        """1e9 nanofarads equals 1 farad."""
        result = 1e9 * nfarad / farad
        assert_allclose(result, 1.0)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert pF is pfarad
        assert nF is nfarad
        assert uF is ufarad


class TestFrequencyUnits:
    """Tests for frequency units (hertz)."""

    def test_kilohertz_to_hertz(self):
        """1 kHz equals 1000 Hz."""
        result = float(khertz / hertz)
        assert_allclose(result, 1000.0)

    def test_megahertz_to_hertz(self):
        """1 MHz equals 1e6 Hz."""
        result = float(Mhertz / hertz)
        assert_allclose(result, 1e6)

    def test_gigahertz_to_hertz(self):
        """1 GHz equals 1e9 Hz."""
        result = float(Ghertz / hertz)
        assert_allclose(result, 1e9)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert Hz is hertz
        assert kHz is khertz
        assert MHz is Mhertz
        assert GHz is Ghertz


class TestLengthUnits:
    """Tests for length units."""

    def test_centimetre_to_metre(self):
        """100 cm equals 1 m."""
        result = 100 * cmetre / metre
        assert_allclose(result, 1.0)

    def test_millimetre_to_metre(self):
        """1000 mm equals 1 m."""
        result = 1000 * mmetre / metre
        assert_allclose(result, 1.0)

    def test_micrometre_to_metre(self):
        """1e6 um equals 1 m."""
        result = 1e6 * umetre / metre
        assert_allclose(result, 1.0)

    def test_kilometre_to_metre(self):
        """1 km equals 1000 m."""
        result = float(kmetre / metre)
        assert_allclose(result, 1000.0)

    def test_short_aliases(self):
        """Short aliases work correctly."""
        assert cm is cmetre
        assert mm is mmetre
        assert um is umetre
        assert nm is nmetre
        assert km is kmetre


class TestAreaUnits:
    """Tests for area units."""

    def test_cm2_dimensions(self):
        """cm2 has length^2 dimensions."""
        assert cm2.dim.get_dimension('length') == 2

    def test_cm2_to_m2(self):
        """1e4 cm2 equals 1 m2."""
        metre2 = metre ** 2
        result = 1e4 * cm2 / metre2
        assert_allclose(float(result), 1.0)

    def test_um2_to_m2(self):
        """1e12 um2 equals 1 m2."""
        metre2 = metre ** 2
        result = 1e12 * um2 / metre2
        assert_allclose(float(result), 1.0)


class TestPhysicalConstants:
    """Tests for physical constants."""

    def test_electron_charge_value(self):
        """Electron charge has correct approximate value."""
        assert_allclose(float(e / coulomb), 1.602176634e-19, rtol=1e-6)

    def test_electron_charge_aliases(self):
        """All electron charge aliases are the same."""
        assert q_e is e
        assert electron_charge is e

    def test_boltzmann_constant_value(self):
        """Boltzmann constant has correct approximate value."""
        # k_B in J/K
        joule = volt * ampere * second
        _dim_jk = k_B.dim
        k_B_value = float(np.asarray(k_B))
        assert_allclose(k_B_value, 1.380649e-23, rtol=1e-6)

    def test_boltzmann_aliases(self):
        """All Boltzmann aliases are the same."""
        assert kB is k_B
        assert boltzmann is k_B

    def test_avogadro_value(self):
        """Avogadro constant has correct approximate value."""
        N_A_value = float(np.asarray(N_A))
        assert_allclose(N_A_value, 6.02214076e23, rtol=1e-6)

    def test_avogadro_alias(self):
        """Avogadro aliases are the same."""
        assert avogadro is N_A

    def test_faraday_constant_value(self):
        """Faraday constant has correct approximate value."""
        F_value = float(np.asarray(F))
        assert_allclose(F_value, 96485.33212, rtol=1e-6)


class TestUnitConversions:
    """Integration tests for unit conversions common in neuroscience."""

    def test_membrane_potential(self):
        """Typical membrane potential -70 mV in volts."""
        Vm = -70 * mV
        result = float(Vm / volt)
        assert_allclose(result, -0.070)

    def test_spike_threshold(self):
        """Spike threshold around -50 mV."""
        threshold = -50 * mV
        resting = -70 * mV
        diff = threshold - resting
        assert_allclose(float(diff / mV), 20.0)

    def test_synaptic_conductance(self):
        """Typical synaptic conductance in nS."""
        g_syn = 1 * nS
        result = float(g_syn / siemens)
        assert_allclose(result, 1e-9)

    def test_membrane_time_constant(self):
        """Typical membrane time constant around 10-20 ms."""
        tau = 20 * ms
        result = float(tau / second)
        assert_allclose(result, 0.020)

    def test_firing_rate(self):
        """Typical firing rate in Hz."""
        rate = 50 * Hz
        period = 1 / rate
        assert_allclose(float(period / ms), 20.0)
