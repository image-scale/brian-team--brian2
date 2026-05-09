"""
Tests for NeuronGroup class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import NeuronGroup
from neurosim.core import Network
from neurosim.core.clocks import Clock, defaultclock
from neurosim.units import second, volt, ampere
from neurosim.units.stdunits import ms, mV
from neurosim.units.core import DIMENSIONLESS, DimensionMismatchError, Quantity


class TestNeuronGroupCreation:
    """Tests for NeuronGroup creation."""

    def test_basic_creation(self):
        """NeuronGroup can be created with equations."""
        G = NeuronGroup(10, 'v : volt')
        assert len(G) == 10
        assert G.N == 10
        assert 'v' in G.variables

    def test_differential_equation(self):
        """NeuronGroup with differential equation."""
        G = NeuronGroup(10, '''
            dv/dt = -v/tau : volt
            tau : second
        ''')
        assert 'v' in G.variables
        assert 'tau' in G.variables

    def test_with_threshold(self):
        """NeuronGroup with threshold condition."""
        G = NeuronGroup(10, 'dv/dt = 1*mV/ms : volt',
                       threshold='v > 10*mV')
        assert G._threshold == 'v > 10*mV'

    def test_with_reset(self):
        """NeuronGroup with reset statement."""
        G = NeuronGroup(10, 'dv/dt = 1*mV/ms : volt',
                       threshold='v > 10*mV',
                       reset='v = 0*mV')
        assert G._reset == 'v = 0*mV'

    def test_with_refractory(self):
        """NeuronGroup with refractory period."""
        G = NeuronGroup(10, 'dv/dt = 1*mV/ms : volt',
                       threshold='v > 10*mV',
                       reset='v = 0*mV',
                       refractory=5*ms)
        assert G._refractory_time == 5*ms

    def test_last_spike_variable(self):
        """NeuronGroup has last_spike variable."""
        G = NeuronGroup(10, 'v : volt')
        assert 'last_spike' in G.variables
        assert_array_equal(G.variables['last_spike'].get_value(), np.full(10, -np.inf))

    def test_custom_clock(self):
        """NeuronGroup with custom clock."""
        clock = Clock(dt=0.05*ms, name='custom_clock')
        G = NeuronGroup(10, 'v : volt', clock=clock)
        assert G.clock is clock

    def test_method_parameter(self):
        """NeuronGroup accepts method parameter."""
        G = NeuronGroup(10, 'dv/dt = -v/tau : volt\ntau : second',
                       method='euler')
        assert G._method == 'euler'


class TestNeuronGroupIntegration:
    """Tests for differential equation integration."""

    def test_euler_integration_simple(self):
        """Simple Euler integration test."""
        G = NeuronGroup(1, '''
            dv/dt = 1*mV/ms : volt
        ''', method='euler')

        clock = G.clock
        clock.reset()

        G.v = 0*mV
        dt = clock.dt_

        G._state_updater.run()

        # 1*mV/ms = 1 V/s (since 1e-3 V / 1e-3 s = 1 V/s)
        # After dt=0.0001s: dv = 1 V/s * 0.0001s = 0.0001 V
        assert_allclose(G.v_[0], 1.0 * dt, rtol=1e-10)

    def test_euler_integration_decay(self):
        """Euler integration of exponential decay."""
        tau_val = 10*ms

        G = NeuronGroup(1, '''
            dv/dt = -v/tau : volt
            tau : second
        ''', method='euler')

        G.v = 1*mV
        G.tau = tau_val

        clock = G.clock
        clock.reset()
        dt = clock.dt_

        for _ in range(10):
            G._state_updater.run()

        expected = 0.001 * (1 - dt / 0.01) ** 10
        assert_allclose(G.v_[0], expected, rtol=1e-5)

    def test_integration_multiple_neurons(self):
        """Integration works for multiple neurons."""
        G = NeuronGroup(5, '''
            dv/dt = rate : volt
            rate : volt/second
        ''', method='euler')

        G.v = np.zeros(5) * mV
        # Set rates in V/s (base units)
        G.rate_ = np.array([1, 2, 3, 4, 5])  # V/s

        clock = G.clock
        clock.reset()
        dt = clock.dt_

        G._state_updater.run()

        expected = np.array([1, 2, 3, 4, 5]) * dt  # V
        assert_allclose(G.v_, expected, rtol=1e-10)


class TestNeuronGroupThreshold:
    """Tests for threshold detection."""

    def test_threshold_detection(self):
        """Threshold detects spikes correctly."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 0.5*mV')

        G.v = np.array([0, 0.4, 0.6, 0.8, 0.2]) * mV

        G._thresholder.run()

        expected_spikes = np.array([2, 3])
        assert_array_equal(G.spikes, expected_spikes)

    def test_threshold_updates_last_spike(self):
        """Threshold updates last_spike variable."""
        G = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')

        G.v = np.array([-1, 1, -1]) * mV

        clock = G.clock
        clock.reset()
        t = clock.t_

        G._thresholder.run()

        last_spike = G.variables['last_spike'].get_value()
        assert last_spike[0] == -np.inf
        assert last_spike[1] == t
        assert last_spike[2] == -np.inf

    def test_no_spikes_when_below_threshold(self):
        """No spikes when all below threshold."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 10*mV')

        G.v = np.array([0, 1, 2, 3, 4]) * mV

        G._thresholder.run()

        assert len(G.spikes) == 0


class TestNeuronGroupReset:
    """Tests for reset after spike."""

    def test_reset_applies_to_spiked_neurons(self):
        """Reset applies to spiked neurons only."""
        G = NeuronGroup(3, 'v : volt',
                       threshold='v > 0*mV',
                       reset='v = -70*mV')

        G.v = np.array([-1, 1, -1]) * mV

        G._thresholder.run()
        G._resetter.run()

        expected = np.array([-0.001, -0.07, -0.001])
        assert_allclose(G.v_, expected, rtol=1e-10)

    def test_reset_multiple_variables(self):
        """Reset can modify multiple variables."""
        G = NeuronGroup(3, '''
            v : volt
            w : volt
        ''', threshold='v > 0*mV',
           reset='''
               v = 0*mV
               w = 5*mV
           ''')

        G.v = np.array([-1, 1, 1]) * mV
        G.w = np.zeros(3) * mV

        G._thresholder.run()
        G._resetter.run()

        assert_allclose(G.v_, np.array([-0.001, 0, 0]), rtol=1e-10)
        assert_allclose(G.w_, np.array([0, 0.005, 0.005]), rtol=1e-10)


class TestNeuronGroupRefractory:
    """Tests for refractory period."""

    def test_refractory_prevents_spike(self):
        """Refractory period prevents immediate re-spiking."""
        G = NeuronGroup(1, 'v : volt',
                       threshold='v > 0*mV',
                       reset='v = -1*mV',
                       refractory=5*ms)

        clock = G.clock
        clock.reset()

        G.v = np.array([1]) * mV

        G._thresholder.run()
        assert len(G.spikes) == 1

        G.v = np.array([1]) * mV
        G._thresholder.run()
        assert len(G.spikes) == 0

    def test_refractory_ends_after_duration(self):
        """Neuron can spike again after refractory period ends."""
        ref_time = 0.5*ms

        G = NeuronGroup(1, 'v : volt',
                       threshold='v > 0*mV',
                       reset='v = -1*mV',
                       refractory=ref_time)

        clock = Clock(dt=0.1*ms, name='test_ref_clock')
        G._clock = clock
        clock.reset()

        G.v = np.array([1]) * mV
        G._thresholder.run()
        assert len(G.spikes) == 1

        # After 4 steps (0.4ms), still refractory (< 0.5ms)
        for _ in range(4):
            clock.advance()

        G.v = np.array([1]) * mV
        G._thresholder.run()
        assert len(G.spikes) == 0

        # After 5th step (0.5ms), refractory period ends (>= 0.5ms)
        clock.advance()
        G._thresholder.run()
        assert len(G.spikes) == 1


class TestNeuronGroupNetwork:
    """Tests for NeuronGroup in Network."""

    def test_neurongroup_in_network(self):
        """NeuronGroup runs in Network."""
        G = NeuronGroup(1, '''
            dv/dt = 1*mV/ms : volt
        ''', threshold='v > 10*mV', reset='v = 0*mV')

        G.v = 0*mV

        net = Network()
        net.add(G)

        defaultclock.reset()
        net.run(1*ms)

        assert G.v_[0] > 0

    def test_threshold_and_reset_in_network(self):
        """Threshold and reset work in Network."""
        G = NeuronGroup(1, '''
            dv/dt = 10*mV/ms : volt
        ''', threshold='v > 5*mV', reset='v = 0*mV')

        G.v = 0*mV

        clock = Clock(dt=0.1*ms, name='net_test_clock')
        G._clock = clock

        net = Network()
        net.add(G)

        clock.reset()
        net.run(2*ms)

        assert G.v_[0] < 0.006

    def test_contained_objects_added(self):
        """NeuronGroup's contained objects are used in network."""
        G = NeuronGroup(1, 'dv/dt = 1*mV/ms : volt',
                       threshold='v > 10*mV', reset='v = 0*mV')

        assert G._state_updater in G.contained_objects
        assert G._thresholder in G.contained_objects
        assert G._resetter in G.contained_objects

    def test_scheduling_order(self):
        """State update, threshold, reset happen in correct order."""
        G = NeuronGroup(1, '''
            dv/dt = 100*mV/ms : volt
        ''', threshold='v > 5*mV', reset='v = 0*mV')

        assert G._state_updater.when == 'groups'
        assert G._thresholder.when == 'thresholds'
        assert G._resetter.when == 'resets'


class TestNeuronGroupNamespace:
    """Tests for namespace handling."""

    def test_namespace_variables(self):
        """Variables are in namespace."""
        G = NeuronGroup(3, '''
            v : volt
            tau : second
        ''')

        G.v = np.array([1, 2, 3]) * mV
        G.tau = 10*ms

        ns = G._get_namespace()
        # Variables with units are Quantities in namespace
        assert_allclose(np.asarray(ns['v']), [0.001, 0.002, 0.003])
        assert_allclose(np.asarray(ns['tau']), [0.01, 0.01, 0.01])

    def test_namespace_time(self):
        """Time variables in namespace."""
        G = NeuronGroup(1, 'v : volt')

        clock = G.clock
        clock.reset()
        ns = G._get_namespace()

        assert 't' in ns
        assert 'dt' in ns
        assert ns['t'] == clock.t_
        assert ns['dt'] == clock.dt_

    def test_namespace_math_functions(self):
        """Math functions in namespace."""
        G = NeuronGroup(1, 'v : volt')
        ns = G._get_namespace()

        assert 'exp' in ns
        assert 'log' in ns
        assert 'sin' in ns
        assert 'cos' in ns
        assert 'sqrt' in ns

    def test_namespace_units(self):
        """Units in namespace."""
        G = NeuronGroup(1, 'v : volt')
        ns = G._get_namespace()

        assert 'mV' in ns
        assert 'ms' in ns
        assert 'second' in ns


class TestNeuronGroupAttributes:
    """Tests for attribute access."""

    def test_get_state_variable(self):
        """Can get state variable with units."""
        G = NeuronGroup(3, 'v : volt')
        G.v = np.array([1, 2, 3]) * mV

        v = G.v
        assert isinstance(v, Quantity)
        assert v.dim == volt.dim

    def test_get_state_variable_raw(self):
        """Can get raw state variable."""
        G = NeuronGroup(3, 'v : volt')
        G.v = np.array([1, 2, 3]) * mV

        v_ = G.v_
        assert not isinstance(v_, Quantity)
        assert_allclose(v_, [0.001, 0.002, 0.003])

    def test_set_state_variable(self):
        """Can set state variable."""
        G = NeuronGroup(3, 'v : volt')
        G.v = 5*mV

        assert_allclose(G.v_, [0.005, 0.005, 0.005])


class TestNeuronGroupRepr:
    """Tests for repr."""

    def test_repr(self):
        """NeuronGroup has repr."""
        G = NeuronGroup(100, 'v : volt', name='test_neurons')
        r = repr(G)
        assert 'NeuronGroup' in r
        assert 'test_neurons' in r
        assert '100' in r
