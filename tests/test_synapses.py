"""
Tests for Synapses class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import NeuronGroup
from neurosim.synapses import Synapses
from neurosim.core import Network
from neurosim.core.clocks import Clock, defaultclock
from neurosim.units import second, volt
from neurosim.units.stdunits import ms, mV
from neurosim.units.core import DIMENSIONLESS, Quantity


class TestSynapsesCreation:
    """Tests for Synapses creation."""

    def test_create_synapses(self):
        """Synapses can be created."""
        pre = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(5, 'v : volt')
        S = Synapses(pre, post)
        assert S.source is pre
        assert S.target is post

    def test_synapses_name(self):
        """Synapses has name."""
        pre = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(5, 'v : volt')
        S = Synapses(pre, post, name='test_syn')
        assert S.name == 'test_syn'

    def test_synapses_self_connect(self):
        """Can create self-connecting synapses."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        S = Synapses(G, G)
        assert S.source is G
        assert S.target is G

    def test_synapses_default_target(self):
        """Target defaults to source."""
        G = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        S = Synapses(G)
        assert S.target is G


class TestSynapsesConnect:
    """Tests for connect method."""

    def test_connect_all_to_all(self):
        """connect() creates all-to-all connections."""
        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(2, 'v : volt')
        S = Synapses(pre, post)
        S.connect()

        assert S.N == 6
        assert len(S.i) == 6
        assert len(S.j) == 6

    def test_connect_one_to_one(self):
        """connect(j='i') creates one-to-one connections."""
        pre = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(5, 'v : volt')
        S = Synapses(pre, post)
        S.connect(j='i')

        assert S.N == 5
        assert_array_equal(sorted(S.i), [0, 1, 2, 3, 4])
        assert_array_equal(sorted(S.j), [0, 1, 2, 3, 4])

    def test_connect_specific_indices(self):
        """connect(i=..., j=...) creates specific connections."""
        pre = NeuronGroup(5, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(5, 'v : volt')
        S = Synapses(pre, post)
        S.connect(i=[0, 1], j=[3, 4])

        assert S.N == 2
        assert_array_equal(S.i, [0, 1])
        assert_array_equal(S.j, [3, 4])

    def test_connect_probabilistic(self):
        """connect(p=...) creates probabilistic connections."""
        np.random.seed(42)
        pre = NeuronGroup(10, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(10, 'v : volt')
        S = Synapses(pre, post)
        S.connect(p=0.5)

        expected_mean = 50
        assert 20 < S.N < 80

    def test_connect_multiple(self):
        """Multiple connects accumulate synapses."""
        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(3, 'v : volt')
        S = Synapses(pre, post)

        S.connect(i=[0], j=[0])
        S.connect(i=[1], j=[1])
        S.connect(i=[2], j=[2])

        assert S.N == 3


class TestSynapsesVariables:
    """Tests for synaptic variables."""

    def test_i_j_indices(self):
        """Can access i and j indices."""
        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(3, 'v : volt')
        S = Synapses(pre, post)
        S.connect(j='i')

        i = S.i
        j = S.j
        assert len(i) == 3
        assert len(j) == 3

    def test_model_creates_variables(self):
        """Model equations create synaptic variables."""
        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(3, 'v : volt')
        S = Synapses(pre, post, model='w : volt')
        S.connect()

        assert 'w' in S.variables

    def test_set_synaptic_variable(self):
        """Can set synaptic variable values."""
        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(3, 'v : volt')
        S = Synapses(pre, post, model='w : volt')
        S.connect(j='i')

        S.w = 1*mV
        assert_allclose(S.w_, [0.001, 0.001, 0.001])

    def test_get_synaptic_variable(self):
        """Can get synaptic variable values."""
        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(3, 'v : volt')
        S = Synapses(pre, post, model='w : volt')
        S.connect(j='i')

        S.w = np.array([1, 2, 3]) * mV
        w = S.w
        assert isinstance(w, Quantity)
        assert w.dim == volt.dim


class TestSynapsesOnPre:
    """Tests for on_pre code execution."""

    def test_on_pre_adds_to_post(self):
        """on_pre code affects post-synaptic neurons."""
        clock = Clock(dt=0.1*ms, name='onpre_clock')

        pre = NeuronGroup(2, 'v : volt', threshold='v > 0*mV', clock=clock)
        post = NeuronGroup(2, 'v : volt', clock=clock)

        S = Synapses(pre, post, on_pre='v_post += 1*mV')
        S.connect(j='i')

        pre.v = np.array([1, -1]) * mV
        post.v = np.zeros(2) * mV

        pre._thresholder.run()
        S._pathways[0].run()

        assert_allclose(post.v_[0], 0.001)
        assert_allclose(post.v_[1], 0.0)

    def test_on_pre_with_weights(self):
        """on_pre uses synaptic weights."""
        clock = Clock(dt=0.1*ms, name='weight_clock')

        pre = NeuronGroup(2, 'v : volt', threshold='v > 0*mV', clock=clock)
        post = NeuronGroup(2, 'v : volt', clock=clock)

        S = Synapses(pre, post, model='w : volt', on_pre='v_post += w')
        S.connect(j='i')
        S.w = np.array([2, 3]) * mV

        pre.v = np.ones(2) * mV
        post.v = np.zeros(2) * mV

        pre._thresholder.run()
        S._pathways[0].run()

        assert_allclose(post.v_[0], 0.002)
        assert_allclose(post.v_[1], 0.003)


class TestSynapsesDelay:
    """Tests for synaptic delay."""

    def test_delay_parameter(self):
        """Synapses accept delay parameter."""
        pre = NeuronGroup(2, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(2, 'v : volt')

        S = Synapses(pre, post, on_pre='v_post += 1*mV', delay=1*ms)

        assert S._pathways[0]._delay == 0.001

    def test_delayed_effect(self):
        """Delay postpones synaptic effect."""
        clock = Clock(dt=0.5*ms, name='delay_clock')

        pre = NeuronGroup(1, 'v : volt', threshold='v > 0*mV', clock=clock)
        post = NeuronGroup(1, 'v : volt', clock=clock)

        S = Synapses(pre, post, on_pre='v_post += 1*mV', delay=1*ms)
        S.connect()

        clock.reset()
        pre.v = np.array([1]) * mV
        post.v = np.zeros(1) * mV

        pre._thresholder.run()
        S._pathways[0].run()
        assert_allclose(post.v_[0], 0.0)

        clock.advance()
        pre.v = np.array([-1]) * mV
        pre._thresholder.run()
        S._pathways[0].run()
        assert_allclose(post.v_[0], 0.0)

        clock.advance()
        pre._thresholder.run()
        S._pathways[0].run()
        assert_allclose(post.v_[0], 0.001)


class TestSynapsesNetwork:
    """Tests for Synapses in Network."""

    def test_synapses_in_network(self):
        """Synapses run in Network."""
        clock = Clock(dt=1*ms, name='net_syn_clock')

        pre = NeuronGroup(1, '''
            dv/dt = 10*mV/ms : volt
        ''', threshold='v > 5*mV', reset='v = 0*mV', clock=clock)

        post = NeuronGroup(1, 'v : volt', clock=clock)

        S = Synapses(pre, post, on_pre='v_post += 1*mV')
        S.connect()

        pre.v = 0*mV
        post.v = 0*mV

        net = Network()
        net.add(pre)
        net.add(post)
        net.add(S)

        clock.reset()
        net.run(5*ms)

        assert post.v_[0] > 0

    def test_scheduling_phase(self):
        """Synapses run in synapses phase."""
        pre = NeuronGroup(1, 'v : volt', threshold='v > 0*mV')
        post = NeuronGroup(1, 'v : volt')
        S = Synapses(pre, post)

        assert S.when == 'synapses'


class TestSynapsesConvergence:
    """Tests for convergent connections."""

    def test_multiple_pre_to_one_post(self):
        """Multiple pre neurons connect to one post."""
        clock = Clock(dt=0.1*ms, name='conv_clock')

        pre = NeuronGroup(3, 'v : volt', threshold='v > 0*mV', clock=clock)
        post = NeuronGroup(1, 'v : volt', clock=clock)

        S = Synapses(pre, post, on_pre='v_post += 1*mV')
        S.connect()

        assert S.N == 3

        pre.v = np.ones(3) * mV
        post.v = np.zeros(1) * mV

        pre._thresholder.run()
        S._pathways[0].run()

        assert_allclose(post.v_[0], 0.003)


class TestSynapsesRepr:
    """Tests for repr."""

    def test_repr(self):
        """Synapses has repr."""
        pre = NeuronGroup(5, 'v : volt', threshold='v > 0*mV', name='pre')
        post = NeuronGroup(5, 'v : volt', name='post')
        S = Synapses(pre, post, name='syn')
        S.connect()

        r = repr(S)
        assert 'Synapses' in r
        assert 'syn' in r
        assert 'pre' in r
        assert 'post' in r
