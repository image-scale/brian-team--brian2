"""
Tests for PoissonGroup class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import PoissonGroup, NeuronGroup
from neurosim.synapses import Synapses
from neurosim.monitors import SpikeMonitor
from neurosim.core import Network
from neurosim.core.clocks import Clock, defaultclock
from neurosim.units import second, hertz
from neurosim.units.stdunits import ms, mV, Hz


class TestPoissonGroupCreation:
    """Tests for PoissonGroup creation."""

    def test_create_poissongroup(self):
        """PoissonGroup can be created."""
        G = PoissonGroup(10, rates=10*Hz)
        assert G.N == 10
        assert len(G) == 10

    def test_name(self):
        """PoissonGroup has name."""
        G = PoissonGroup(10, rates=10*Hz, name='test_poisson')
        assert G.name == 'test_poisson'

    def test_auto_name(self):
        """PoissonGroup gets auto-generated name."""
        G = PoissonGroup(10, rates=10*Hz)
        assert 'poissongroup' in G.name

    def test_scalar_rate(self):
        """Scalar rate is broadcast to all neurons."""
        G = PoissonGroup(5, rates=100*Hz)
        assert_allclose(G.rates_, [100, 100, 100, 100, 100])

    def test_array_rates(self):
        """Different rates for different neurons."""
        rates = np.array([10, 20, 30, 40, 50]) * Hz
        G = PoissonGroup(5, rates=rates)
        assert_allclose(G.rates_, [10, 20, 30, 40, 50])

    def test_rates_have_units(self):
        """rates property has units."""
        G = PoissonGroup(5, rates=100*Hz)
        r = G.rates
        assert r.dim == hertz.dim


class TestPoissonGroupSpikes:
    """Tests for spike generation."""

    def test_generates_spikes(self):
        """PoissonGroup generates spikes."""
        np.random.seed(42)
        clock = Clock(dt=1*ms, name='poisson_clock')
        G = PoissonGroup(100, rates=1000*Hz, clock=clock)

        clock.reset()
        G.run()

        assert len(G.spikes) > 0

    def test_zero_rate_no_spikes(self):
        """Zero rate produces no spikes."""
        clock = Clock(dt=1*ms, name='zero_clock')
        G = PoissonGroup(100, rates=0*Hz, clock=clock)

        clock.reset()
        for _ in range(100):
            G.run()
            assert len(G.spikes) == 0
            clock.advance()

    def test_average_rate_correct(self):
        """Average spike rate matches expected rate."""
        np.random.seed(42)
        clock = Clock(dt=0.1*ms, name='avg_clock')
        rate = 100
        G = PoissonGroup(100, rates=rate*Hz, clock=clock)

        clock.reset()
        total_spikes = 0
        steps = 1000

        for _ in range(steps):
            G.run()
            total_spikes += len(G.spikes)
            clock.advance()

        duration = steps * clock.dt_
        n_neurons = G.N
        expected_spikes = rate * duration * n_neurons

        assert 0.5 * expected_spikes < total_spikes < 1.5 * expected_spikes


class TestPoissonGroupRates:
    """Tests for rate modification."""

    def test_set_rates_scalar(self):
        """Can set rates with scalar."""
        G = PoissonGroup(5, rates=100*Hz)
        G.rates = 200*Hz
        assert_allclose(G.rates_, [200, 200, 200, 200, 200])

    def test_set_rates_array(self):
        """Can set rates with array."""
        G = PoissonGroup(3, rates=100*Hz)
        G.rates = np.array([10, 20, 30]) * Hz
        assert_allclose(G.rates_, [10, 20, 30])

    def test_rates_change_affects_spikes(self):
        """Changing rates affects spike generation."""
        np.random.seed(42)
        clock = Clock(dt=1*ms, name='change_clock')
        G = PoissonGroup(100, rates=1000*Hz, clock=clock)

        clock.reset()
        G.run()
        high_rate_spikes = len(G.spikes)

        G.rates = 10*Hz
        G.run()
        low_rate_spikes = len(G.spikes)

        assert high_rate_spikes > low_rate_spikes


class TestPoissonGroupNetwork:
    """Tests for PoissonGroup in Network."""

    def test_poissongroup_in_network(self):
        """PoissonGroup runs in Network."""
        np.random.seed(42)
        clock = Clock(dt=0.1*ms, name='net_poisson_clock')

        G = PoissonGroup(10, rates=1000*Hz, clock=clock)
        mon = SpikeMonitor(G)

        net = Network()
        net.add(G)
        net.add(mon)

        clock.reset()
        net.run(10*ms)

        assert mon.num_spikes > 0

    def test_poissongroup_drives_synapses(self):
        """PoissonGroup can drive synapses."""
        np.random.seed(42)
        clock = Clock(dt=0.1*ms, name='syn_poisson_clock')

        gen = PoissonGroup(10, rates=1000*Hz, clock=clock)
        target = NeuronGroup(10, 'v : volt', clock=clock)

        S = Synapses(gen, target, on_pre='v_post += 1*mV')
        S.connect(j='i')

        target.v = 0*mV

        net = Network()
        net.add(gen)
        net.add(target)
        net.add(S)

        clock.reset()
        net.run(10*ms)

        assert np.any(target.v_ > 0)

    def test_scheduling_phase(self):
        """PoissonGroup runs in thresholds phase."""
        G = PoissonGroup(1, rates=100*Hz)
        assert G.when == 'thresholds'


class TestPoissonGroupRepr:
    """Tests for repr."""

    def test_repr(self):
        """PoissonGroup has repr."""
        G = PoissonGroup(100, rates=50*Hz, name='poisson')
        r = repr(G)
        assert 'PoissonGroup' in r
        assert 'poisson' in r
        assert '100' in r
