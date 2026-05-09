"""
Tests for StateMonitor class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import NeuronGroup
from neurosim.monitors import StateMonitor
from neurosim.core import Network
from neurosim.core.clocks import Clock, defaultclock
from neurosim.units import second, volt
from neurosim.units.stdunits import ms, mV
from neurosim.units.core import DIMENSIONLESS, Quantity


class TestStateMonitorCreation:
    """Tests for StateMonitor creation."""

    def test_create_statemonitor(self):
        """StateMonitor can be created."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v', record=True)
        assert mon.source is G

    def test_statemonitor_name(self):
        """StateMonitor has name."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v', name='my_statemon')
        assert mon.name == 'my_statemon'

    def test_statemonitor_auto_name(self):
        """StateMonitor gets auto-generated name."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v')
        assert 'statemonitor' in mon.name

    def test_single_variable(self):
        """StateMonitor records single variable."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v')
        assert mon.variables == ['v']

    def test_multiple_variables(self):
        """StateMonitor records multiple variables."""
        G = NeuronGroup(10, '''
            v : volt
            w : volt
        ''')
        mon = StateMonitor(G, ['v', 'w'])
        assert set(mon.variables) == {'v', 'w'}

    def test_record_all_true(self):
        """record=True records all neurons."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v', record=True)
        assert_array_equal(mon._record_indices, np.arange(10))

    def test_record_subset(self):
        """Can record subset of neurons."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v', record=[0, 3, 7])
        assert_array_equal(mon._record_indices, [0, 3, 7])

    def test_invalid_variable_raises(self):
        """Invalid variable name raises error."""
        G = NeuronGroup(10, 'v : volt')
        with pytest.raises(KeyError):
            StateMonitor(G, 'nonexistent')


class TestStateMonitorRecording:
    """Tests for state recording."""

    def test_record_single_timestep(self):
        """StateMonitor records single timestep."""
        G = NeuronGroup(3, 'v : volt')
        mon = StateMonitor(G, 'v', record=True)

        G.v = np.array([1, 2, 3]) * mV
        mon.run()

        v = mon.v_
        assert v.shape == (3, 1)
        assert_allclose(v[:, 0], [0.001, 0.002, 0.003])

    def test_record_multiple_timesteps(self):
        """StateMonitor records multiple timesteps."""
        clock = Clock(dt=0.1*ms, name='state_clock')
        G = NeuronGroup(2, 'v : volt', clock=clock)
        mon = StateMonitor(G, 'v', record=True)

        clock.reset()

        G.v = np.array([1, 2]) * mV
        mon.run()
        clock.advance()

        G.v = np.array([3, 4]) * mV
        mon.run()

        v = mon.v_
        assert v.shape == (2, 2)
        assert_allclose(v[0, :], [0.001, 0.003])
        assert_allclose(v[1, :], [0.002, 0.004])

    def test_recording_times(self):
        """StateMonitor records correct times."""
        clock = Clock(dt=0.1*ms, name='time_clock')
        G = NeuronGroup(1, 'v : volt', clock=clock)
        mon = StateMonitor(G, 'v', record=True)

        clock.reset()

        for _ in range(3):
            mon.run()
            clock.advance()

        assert_allclose(mon.t_, [0, 0.0001, 0.0002])

    def test_recording_times_have_units(self):
        """Recording times have proper units."""
        G = NeuronGroup(1, 'v : volt')
        mon = StateMonitor(G, 'v')

        mon.run()
        t = mon.t

        assert isinstance(t, Quantity)
        assert t.dim == second.dim

    def test_recorded_values_have_units(self):
        """Recorded values have proper units."""
        G = NeuronGroup(1, 'v : volt')
        mon = StateMonitor(G, 'v')

        G.v = 1*mV
        mon.run()

        v = mon.v
        assert isinstance(v, Quantity)
        assert v.dim == volt.dim

    def test_raw_values_no_units(self):
        """v_ returns raw values without units."""
        G = NeuronGroup(1, 'v : volt')
        mon = StateMonitor(G, 'v')

        G.v = 1*mV
        mon.run()

        v_ = mon.v_
        assert not isinstance(v_, Quantity)


class TestStateMonitorSubset:
    """Tests for recording from subset of neurons."""

    def test_record_single_neuron(self):
        """Can record from single neuron."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v', record=[5])

        G.v = np.arange(10) * mV
        mon.run()

        v = mon.v_
        assert v.shape == (1, 1)
        assert_allclose(v[0, 0], 0.005)

    def test_record_subset_shape(self):
        """Subset recording has correct shape."""
        G = NeuronGroup(10, 'v : volt')
        mon = StateMonitor(G, 'v', record=[0, 5, 9])

        G.v = np.arange(10) * mV
        mon.run()
        mon.run()

        v = mon.v_
        assert v.shape == (3, 2)


class TestStateMonitorMultipleVariables:
    """Tests for recording multiple variables."""

    def test_record_multiple_variables(self):
        """Can record multiple variables."""
        G = NeuronGroup(2, '''
            v : volt
            w : volt
        ''')
        mon = StateMonitor(G, ['v', 'w'], record=True)

        G.v = np.array([1, 2]) * mV
        G.w = np.array([3, 4]) * mV
        mon.run()

        assert_allclose(mon.v_[:, 0], [0.001, 0.002])
        assert_allclose(mon.w_[:, 0], [0.003, 0.004])

    def test_record_all_variables(self):
        """variables=True records all array variables."""
        G = NeuronGroup(2, '''
            v : volt
            w : volt
        ''')
        mon = StateMonitor(G, True, record=True)

        assert 'v' in mon.variables
        assert 'w' in mon.variables


class TestStateMonitorNetwork:
    """Tests for StateMonitor in Network."""

    def test_statemonitor_in_network(self):
        """StateMonitor runs in Network."""
        G = NeuronGroup(1, '''
            dv/dt = 1*mV/ms : volt
        ''')
        G.v = 0*mV

        mon = StateMonitor(G, 'v', record=True)

        net = Network()
        net.add(G)
        net.add(mon)

        defaultclock.reset()
        net.run(1*ms)

        assert len(mon.t) == 10
        assert mon.v_[0, -1] > mon.v_[0, 0]

    def test_scheduling_phase(self):
        """StateMonitor runs in end phase."""
        G = NeuronGroup(1, 'v : volt')
        mon = StateMonitor(G, 'v')

        assert mon.when == 'end'

    def test_records_after_state_update(self):
        """StateMonitor records after state update."""
        clock = Clock(dt=1*ms, name='net_order_clock')
        G = NeuronGroup(1, '''
            dv/dt = 1*mV/ms : volt
        ''', clock=clock)
        G.v = 0*mV

        mon = StateMonitor(G, 'v', record=True)

        net = Network()
        net.add(G)
        net.add(mon)

        clock.reset()
        net.run(2*ms)

        assert_allclose(mon.v_[0, 0], 0.001, rtol=1e-5)
        assert_allclose(mon.v_[0, 1], 0.002, rtol=1e-5)


class TestStateMonitorReinit:
    """Tests for reinit method."""

    def test_reinit_clears_data(self):
        """reinit clears all recorded data."""
        G = NeuronGroup(3, 'v : volt')
        mon = StateMonitor(G, 'v', record=True)

        G.v = np.ones(3) * mV
        mon.run()
        mon.run()

        assert len(mon.t) == 2

        mon.reinit()

        assert len(mon.t) == 0
        assert mon.v_.shape == (3, 0)


class TestStateMonitorDimensionless:
    """Tests for dimensionless variables."""

    def test_dimensionless_variable(self):
        """Dimensionless variables work correctly."""
        G = NeuronGroup(2, 'x : 1')
        mon = StateMonitor(G, 'x', record=True)

        G.x = np.array([1.5, 2.5])
        mon.run()

        x = mon.x
        assert_allclose(x[:, 0], [1.5, 2.5])


class TestStateMonitorRepr:
    """Tests for repr."""

    def test_repr(self):
        """StateMonitor has repr."""
        G = NeuronGroup(10, 'v : volt', name='neurons')
        mon = StateMonitor(G, 'v', name='mon')

        r = repr(mon)
        assert 'StateMonitor' in r
        assert 'mon' in r
        assert 'neurons' in r
