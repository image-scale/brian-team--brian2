"""
Tests for the simulation clock system.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from neurosim.core import Clock, defaultclock
from neurosim.units import second
from neurosim.units.stdunits import ms, us


class TestClock:
    """Tests for the Clock class."""

    def test_clock_default_dt(self):
        """Clock defaults to 0.1 ms time step."""
        clock = Clock()
        assert_allclose(float(clock.dt / ms), 0.1, rtol=1e-10)

    def test_clock_custom_dt(self):
        """Clock accepts custom dt."""
        clock = Clock(dt=1 * ms)
        assert_allclose(float(clock.dt / ms), 1.0, rtol=1e-10)

    def test_clock_dt_float(self):
        """Clock accepts float dt (interpreted as seconds)."""
        clock = Clock(dt=0.001)  # 1 ms
        assert_allclose(float(clock.dt / ms), 1.0, rtol=1e-10)

    def test_clock_invalid_dt_raises(self):
        """Clock with invalid dt raises error."""
        with pytest.raises(ValueError):
            Clock(dt=0)
        with pytest.raises(ValueError):
            Clock(dt=-1 * ms)

    def test_clock_initial_time(self):
        """Clock starts at time 0."""
        clock = Clock()
        assert_allclose(float(clock.t / second), 0.0)
        assert clock.timestep == 0

    def test_clock_advance(self):
        """Clock advance increments time by dt."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(0 * ms, 10 * ms)
        clock.advance()
        assert_allclose(float(clock.t / ms), 1.0)
        assert clock.timestep == 1
        clock.advance()
        assert_allclose(float(clock.t / ms), 2.0)
        assert clock.timestep == 2

    def test_clock_set_interval(self):
        """Clock set_interval sets time range."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(5 * ms, 10 * ms)
        assert_allclose(float(clock.t / ms), 5.0)
        assert clock.timestep == 5

    def test_clock_stop_iteration(self):
        """Clock raises StopIteration at end time."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(0 * ms, 2 * ms)
        clock.advance()  # t = 1 ms
        clock.advance()  # t = 2 ms
        with pytest.raises(StopIteration):
            clock.advance()  # Should raise

    def test_clock_reset(self):
        """Clock reset returns to time 0."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(0 * ms, 10 * ms)
        clock.advance()
        clock.advance()
        assert clock.timestep > 0
        clock.reset()
        assert clock.timestep == 0
        assert_allclose(float(clock.t / second), 0.0)

    def test_clock_name(self):
        """Clock accepts custom name."""
        clock = Clock(name='myclock')
        assert clock.name == 'myclock'

    def test_clock_auto_name(self):
        """Clock generates automatic name if none given."""
        clock = Clock()
        assert 'clock' in clock.name

    def test_clock_dt_property(self):
        """Clock dt can be changed."""
        clock = Clock(dt=1 * ms)
        assert_allclose(float(clock.dt / ms), 1.0)
        clock.dt = 0.5 * ms
        assert_allclose(float(clock.dt / ms), 0.5)

    def test_clock_dt_underscore(self):
        """Clock dt_ returns float in seconds."""
        clock = Clock(dt=1 * ms)
        assert_allclose(clock.dt_, 0.001)

    def test_clock_t_underscore(self):
        """Clock t_ returns float in seconds."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(0 * ms, 10 * ms)
        clock.advance()
        assert_allclose(clock.t_, 0.001)

    def test_clock_same_time(self):
        """Clocks at same time compare as equal."""
        clock1 = Clock(dt=1 * ms)
        clock2 = Clock(dt=1 * ms)
        clock1.set_interval(0 * ms, 10 * ms)
        clock2.set_interval(0 * ms, 10 * ms)
        assert clock1.same_time(clock2)
        clock1.advance()
        clock2.advance()
        assert clock1.same_time(clock2)

    def test_clock_different_time(self):
        """Clocks at different times compare as not equal."""
        clock1 = Clock(dt=1 * ms)
        clock2 = Clock(dt=1 * ms)
        clock1.set_interval(0 * ms, 10 * ms)
        clock2.set_interval(0 * ms, 10 * ms)
        clock1.advance()
        assert not clock1.same_time(clock2)

    def test_clock_repr(self):
        """Clock has meaningful repr."""
        clock = Clock(dt=1 * ms, name='test')
        r = repr(clock)
        assert 'Clock' in r
        assert 'test' in r


class TestDefaultclock:
    """Tests for the global defaultclock."""

    def setup_method(self):
        """Reset defaultclock before each test."""
        defaultclock.reset()
        defaultclock.dt = 0.1 * ms

    def test_defaultclock_exists(self):
        """defaultclock is available."""
        assert defaultclock is not None

    def test_defaultclock_name(self):
        """defaultclock has standard name."""
        assert defaultclock.name == 'defaultclock'

    def test_defaultclock_default_dt(self):
        """defaultclock has 0.1 ms default dt."""
        assert_allclose(float(defaultclock.dt / ms), 0.1, rtol=1e-10)

    def test_defaultclock_can_set_dt(self):
        """defaultclock dt can be changed."""
        defaultclock.dt = 1 * ms
        assert_allclose(float(defaultclock.dt / ms), 1.0, rtol=1e-10)

    def test_defaultclock_advance(self):
        """defaultclock can advance time."""
        defaultclock.set_interval(0 * ms, 1 * ms)
        defaultclock.advance()
        assert_allclose(float(defaultclock.t / ms), 0.1)


class TestMultipleClocks:
    """Tests for multiple coexisting clocks."""

    def test_independent_clocks(self):
        """Multiple clocks operate independently."""
        clock1 = Clock(dt=1 * ms, name='clock1')
        clock2 = Clock(dt=2 * ms, name='clock2')

        clock1.set_interval(0 * ms, 10 * ms)
        clock2.set_interval(0 * ms, 10 * ms)

        clock1.advance()
        assert_allclose(float(clock1.t / ms), 1.0)
        assert_allclose(float(clock2.t / ms), 0.0)

        clock2.advance()
        assert_allclose(float(clock1.t / ms), 1.0)
        assert_allclose(float(clock2.t / ms), 2.0)

    def test_clocks_different_dt(self):
        """Clocks can have different dt values."""
        clock1 = Clock(dt=0.1 * ms)
        clock2 = Clock(dt=1 * ms)
        clock3 = Clock(dt=10 * us)

        assert_allclose(float(clock1.dt / ms), 0.1)
        assert_allclose(float(clock2.dt / ms), 1.0)
        assert_allclose(float(clock3.dt / us), 10.0)


class TestClockUnits:
    """Tests for clock unit handling."""

    def test_clock_time_has_units(self):
        """Clock time is a Quantity with time units."""
        clock = Clock(dt=1 * ms)
        t = clock.t
        assert hasattr(t, 'dim')
        assert t.dim == second.dim

    def test_clock_dt_has_units(self):
        """Clock dt is a Quantity with time units."""
        clock = Clock(dt=1 * ms)
        dt = clock.dt
        assert hasattr(dt, 'dim')
        assert dt.dim == second.dim

    def test_clock_interval_accepts_units(self):
        """Clock set_interval accepts Quantity arguments."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(5 * ms, 10 * ms)
        assert_allclose(float(clock.t / ms), 5.0)

    def test_clock_interval_accepts_floats(self):
        """Clock set_interval accepts float arguments (seconds)."""
        clock = Clock(dt=1 * ms)
        clock.set_interval(0.005, 0.010)  # 5ms to 10ms
        assert_allclose(float(clock.t / ms), 5.0)
