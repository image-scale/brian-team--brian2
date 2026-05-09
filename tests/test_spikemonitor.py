"""
Tests for SpikeMonitor class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import NeuronGroup
from neurosim.monitors import SpikeMonitor
from neurosim.core import Network
from neurosim.core.clocks import Clock, defaultclock
from neurosim.units import second
from neurosim.units.stdunits import ms, mV
from neurosim.units.core import Quantity


class TestSpikeMonitorCreation:
    """Tests for SpikeMonitor creation."""

    def test_create_spikemonitor(self):
        """SpikeMonitor can be created."""
        G = NeuronGroup(10, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)
        assert mon.source is G

    def test_spikemonitor_name(self):
        """SpikeMonitor has name."""
        G = NeuronGroup(10, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G, name='my_spikemon')
        assert mon.name == 'my_spikemon'

    def test_spikemonitor_auto_name(self):
        """SpikeMonitor gets auto-generated name."""
        G = NeuronGroup(10, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)
        assert 'spikemonitor' in mon.name

    def test_spikemonitor_record_all(self):
        """SpikeMonitor records all neurons by default."""
        G = NeuronGroup(10, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)
        assert mon._record_indices is None

    def test_spikemonitor_record_subset(self):
        """SpikeMonitor can record subset of neurons."""
        G = NeuronGroup(10, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G, record=[0, 2, 4])
        assert_array_equal(mon._record_indices, [0, 2, 4])


class TestSpikeMonitorRecording:
    """Tests for spike recording."""

    def test_record_single_spike(self):
        """SpikeMonitor records single spike."""
        G = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)

        G.v = np.array([-1, 1, -1]) * mV
        G._thresholder.run()
        mon.run()

        assert_array_equal(mon.i, [1])
        assert mon.num_spikes == 1

    def test_record_multiple_spikes(self):
        """SpikeMonitor records multiple spikes in same timestep."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)

        G.v = np.array([-1, 1, -1, 1, 1]) * mV
        G._thresholder.run()
        mon.run()

        assert_array_equal(sorted(mon.i), [1, 3, 4])
        assert mon.num_spikes == 3

    def test_record_spike_times(self):
        """SpikeMonitor records correct spike times."""
        clock = Clock(dt=0.1*ms, name='test_spikemon_clock')
        G = NeuronGroup(2, 'v : volt', threshold='v > 0*mV', clock=clock)
        mon = SpikeMonitor(G)

        clock.reset()

        G.v = np.array([1, -1]) * mV
        G._thresholder.run()
        mon.run()

        clock.advance()
        G.v = np.array([-1, 1]) * mV
        G._thresholder.run()
        mon.run()

        assert_array_equal(mon.i, [0, 1])
        assert_allclose(mon.t_, [0.0, 0.0001])

    def test_spike_times_have_units(self):
        """Spike times have proper units."""
        G = NeuronGroup(2, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)

        G.v = np.array([1, -1]) * mV
        G._thresholder.run()
        mon.run()

        t = mon.t
        assert isinstance(t, Quantity)
        assert t.dim == second.dim

    def test_accumulate_spikes(self):
        """SpikeMonitor accumulates spikes across timesteps."""
        clock = Clock(dt=0.1*ms, name='test_accum_clock')
        G = NeuronGroup(3, 'v : volt', threshold='v > 0*mV', clock=clock)
        mon = SpikeMonitor(G)

        clock.reset()

        for i in range(3):
            G.v = np.zeros(3) * mV
            G.v_[i] = 0.001
            G._thresholder.run()
            mon.run()
            clock.advance()

        assert mon.num_spikes == 3
        assert_array_equal(sorted(mon.i), [0, 1, 2])


class TestSpikeMonitorCount:
    """Tests for spike count."""

    def test_count_property(self):
        """count property returns spikes per neuron."""
        G = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)

        G.v = np.array([1, 1, -1]) * mV
        G._thresholder.run()
        mon.run()

        assert_array_equal(mon.count, [1, 1, 0])

    def test_count_multiple_spikes(self):
        """count accumulates multiple spikes per neuron."""
        clock = Clock(dt=0.1*ms, name='test_count_clock')
        G = NeuronGroup(2, 'v : volt', threshold='v > 0*mV', clock=clock)
        mon = SpikeMonitor(G)

        clock.reset()
        G.variables['_refractory_until'].set_value(np.full(2, -np.inf))

        for _ in range(3):
            G.v = np.array([1, -1]) * mV
            G._thresholder.run()
            mon.run()
            clock.advance()

        assert mon.count[0] == 3
        assert mon.count[1] == 0


class TestSpikeMonitorRecord:
    """Tests for selective recording."""

    def test_record_subset(self):
        """Only records spikes from specified neurons."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G, record=[0, 2])

        G.v = np.array([1, 1, 1, 1, 1]) * mV
        G._thresholder.run()
        mon.run()

        assert_array_equal(sorted(mon.i), [0, 2])
        assert mon.num_spikes == 2

    def test_record_false(self):
        """record=False records nothing."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G, record=False)

        G.v = np.ones(5) * mV
        G._thresholder.run()
        mon.run()

        assert mon.num_spikes == 0


class TestSpikeMonitorNetwork:
    """Tests for SpikeMonitor in Network."""

    def test_spikemonitor_in_network(self):
        """SpikeMonitor runs in Network."""
        G = NeuronGroup(1, '''
            dv/dt = 10*mV/ms : volt
        ''', threshold='v > 5*mV', reset='v = 0*mV')
        G.v = 0*mV

        mon = SpikeMonitor(G)

        net = Network()
        net.add(G)
        net.add(mon)

        defaultclock.reset()
        net.run(2*ms)

        assert mon.num_spikes > 0

    def test_scheduling_phase(self):
        """SpikeMonitor runs in thresholds phase."""
        G = NeuronGroup(1, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)

        assert mon.when == 'thresholds'
        assert mon.order == 1

    def test_multiple_neurons_spike_pattern(self):
        """SpikeMonitor captures correct spike pattern."""
        clock = Clock(dt=1*ms, name='pattern_clock')
        G = NeuronGroup(3, '''
            dv/dt = rate : volt
            rate : volt/second
        ''', threshold='v > 10*mV', reset='v = 0*mV', clock=clock)

        G.v = 0*mV
        G.rate_ = np.array([10, 20, 30])

        mon = SpikeMonitor(G)

        net = Network()
        net.add(G)
        net.add(mon)

        clock.reset()
        net.run(5*ms)

        assert mon.count[2] >= mon.count[1] >= mon.count[0]


class TestSpikeMonitorReinit:
    """Tests for reinit method."""

    def test_reinit_clears_data(self):
        """reinit clears all recorded data."""
        G = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        mon = SpikeMonitor(G)

        G.v = np.ones(3) * mV
        G._thresholder.run()
        mon.run()

        assert mon.num_spikes > 0

        mon.reinit()

        assert mon.num_spikes == 0
        assert len(mon.i) == 0
        assert len(mon.t) == 0
        assert_array_equal(mon.count, [0, 0, 0])


class TestSpikeMonitorSpikeTrains:
    """Tests for spike_trains method."""

    def test_spike_trains(self):
        """spike_trains returns dictionary of spike times per neuron."""
        clock = Clock(dt=0.1*ms, name='train_clock')
        G = NeuronGroup(3, 'v : volt', threshold='v > 0*mV', clock=clock)
        mon = SpikeMonitor(G)

        clock.reset()

        G.v = np.array([1, -1, -1]) * mV
        G._thresholder.run()
        mon.run()
        clock.advance()

        G.v = np.array([1, 1, -1]) * mV
        G._thresholder.run()
        mon.run()

        trains = mon.spike_trains()

        assert 0 in trains
        assert 1 in trains
        assert 2 not in trains
        assert len(trains[0]) == 2
        assert len(trains[1]) == 1


class TestSpikeMonitorRepr:
    """Tests for repr."""

    def test_repr(self):
        """SpikeMonitor has repr."""
        G = NeuronGroup(10, 'v : volt', threshold='v > 0*mV', name='neurons')
        mon = SpikeMonitor(G, name='mon')

        r = repr(mon)
        assert 'SpikeMonitor' in r
        assert 'mon' in r
        assert 'neurons' in r
