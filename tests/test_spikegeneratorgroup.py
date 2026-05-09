"""
Tests for SpikeGeneratorGroup class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import SpikeGeneratorGroup, NeuronGroup
from neurosim.synapses import Synapses
from neurosim.monitors import SpikeMonitor
from neurosim.core import Network
from neurosim.core.clocks import Clock, defaultclock
from neurosim.units import second
from neurosim.units.stdunits import ms, mV


class TestSpikeGeneratorGroupCreation:
    """Tests for SpikeGeneratorGroup creation."""

    def test_create_spikegeneratorgroup(self):
        """SpikeGeneratorGroup can be created."""
        indices = [0, 1, 0]
        times = np.array([1, 2, 3]) * ms
        G = SpikeGeneratorGroup(2, indices, times)
        assert G.N == 2
        assert len(G) == 2

    def test_name(self):
        """SpikeGeneratorGroup has name."""
        G = SpikeGeneratorGroup(2, [0], np.array([1]) * ms, name='test_gen')
        assert G.name == 'test_gen'

    def test_auto_name(self):
        """SpikeGeneratorGroup gets auto-generated name."""
        G = SpikeGeneratorGroup(2, [0], np.array([1]) * ms)
        assert 'spikegeneratorgroup' in G.name

    def test_spike_times_sorted(self):
        """Spike times are sorted internally."""
        times = np.array([3, 1, 2]) * ms
        indices = [0, 1, 2]
        G = SpikeGeneratorGroup(3, indices, times)

        assert_allclose(G._spike_times, [0.001, 0.002, 0.003])

    def test_invalid_indices_raises(self):
        """Invalid indices raise error."""
        with pytest.raises(ValueError):
            SpikeGeneratorGroup(2, [0, 5], np.array([1, 2]) * ms)

    def test_mismatched_lengths_raises(self):
        """Mismatched indices/times lengths raise error."""
        with pytest.raises(ValueError):
            SpikeGeneratorGroup(2, [0, 1], np.array([1]) * ms)


class TestSpikeGeneratorGroupSpikes:
    """Tests for spike generation."""

    def test_spikes_at_correct_times(self):
        """Spikes occur at correct times."""
        clock = Clock(dt=0.1*ms, name='gen_clock')
        G = SpikeGeneratorGroup(2, [0], [0.5*ms], clock=clock)

        clock.reset()
        G.before_run()

        # Run for 5 timesteps to reach t=0.5ms
        for i in range(5):
            G.run()
            if i < 4:
                assert len(G.spikes) == 0
            clock.advance()

        # At t=0.5ms, spike should have occurred
        G.run()
        assert_array_equal(G.spikes, [0])

    def test_multiple_spikes_same_time(self):
        """Multiple neurons can spike at same time."""
        clock = Clock(dt=0.1*ms, name='multi_clock')
        G = SpikeGeneratorGroup(3, [0, 1, 2], np.array([0.1, 0.1, 0.1]) * ms, clock=clock)

        clock.reset()
        G.before_run()

        G.run()
        clock.advance()
        G.run()

        assert len(G.spikes) == 3
        assert set(G.spikes) == {0, 1, 2}

    def test_multiple_spikes_per_neuron(self):
        """Single neuron can spike multiple times."""
        clock = Clock(dt=0.1*ms, name='repeat_clock')
        G = SpikeGeneratorGroup(1, [0, 0, 0], np.array([0.1, 0.2, 0.3]) * ms, clock=clock)

        clock.reset()
        G.before_run()

        spikes_count = 0
        for _ in range(5):
            G.run()
            spikes_count += len(G.spikes)
            clock.advance()

        assert spikes_count == 3

    def test_spikes_only_once(self):
        """Each spike occurs only once."""
        clock = Clock(dt=0.1*ms, name='once_clock')
        G = SpikeGeneratorGroup(1, [0], [0.1*ms], clock=clock)

        clock.reset()
        G.before_run()

        spike_times = []
        for i in range(10):
            G.run()
            if len(G.spikes) > 0:
                spike_times.append(i)
            clock.advance()

        assert len(spike_times) == 1


class TestSpikeGeneratorGroupPeriod:
    """Tests for periodic spike patterns."""

    def test_periodic_spikes(self):
        """Spikes repeat with period."""
        clock = Clock(dt=0.5*ms, name='period_clock')
        G = SpikeGeneratorGroup(1, [0], [0.5*ms], period=1*ms, clock=clock)

        clock.reset()
        G.before_run()

        spike_count = 0
        for _ in range(10):
            G.run()
            spike_count += len(G.spikes)
            clock.advance()

        assert spike_count >= 4


class TestSpikeGeneratorGroupNetwork:
    """Tests for SpikeGeneratorGroup in Network."""

    def test_spikegenerator_in_network(self):
        """SpikeGeneratorGroup runs in Network."""
        clock = Clock(dt=0.1*ms, name='net_gen_clock')

        G = SpikeGeneratorGroup(1, [0], [0.5*ms], clock=clock)
        mon = SpikeMonitor(G)

        net = Network()
        net.add(G)
        net.add(mon)

        clock.reset()
        net.run(1*ms)

        assert mon.num_spikes == 1

    def test_spikegenerator_drives_synapses(self):
        """SpikeGeneratorGroup can drive synapses."""
        clock = Clock(dt=0.1*ms, name='syn_gen_clock')

        gen = SpikeGeneratorGroup(1, [0], [0.2*ms], clock=clock)
        target = NeuronGroup(1, 'v : volt', clock=clock)

        S = Synapses(gen, target, on_pre='v_post += 1*mV')
        S.connect()

        target.v = 0*mV

        net = Network()
        net.add(gen)
        net.add(target)
        net.add(S)

        clock.reset()
        net.run(1*ms)

        assert target.v_[0] > 0

    def test_scheduling_phase(self):
        """SpikeGeneratorGroup runs in thresholds phase."""
        G = SpikeGeneratorGroup(1, [0], [1*ms])
        assert G.when == 'thresholds'


class TestSpikeGeneratorGroupRepr:
    """Tests for repr."""

    def test_repr(self):
        """SpikeGeneratorGroup has repr."""
        G = SpikeGeneratorGroup(5, [0, 1, 2], np.array([1, 2, 3]) * ms, name='gen')
        r = repr(G)
        assert 'SpikeGeneratorGroup' in r
        assert 'gen' in r
        assert '5' in r
