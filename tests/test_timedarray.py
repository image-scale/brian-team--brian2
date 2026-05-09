"""
Tests for TimedArray class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.input import TimedArray
from neurosim.units import second, volt
from neurosim.units.stdunits import ms, mV
from neurosim.units.core import DIMENSIONLESS, Quantity


class TestTimedArrayCreation:
    """Tests for TimedArray creation."""

    def test_create_timedarray(self):
        """TimedArray can be created."""
        values = np.array([0, 1, 2, 3])
        arr = TimedArray(values, dt=10*ms)
        assert arr._n_values == 4

    def test_name(self):
        """TimedArray has name."""
        arr = TimedArray([0, 1, 2], dt=1*ms, name='my_array')
        assert arr.name == 'my_array'

    def test_auto_name(self):
        """TimedArray gets default name."""
        arr = TimedArray([0, 1, 2], dt=1*ms)
        assert arr.name == 'timedarray'

    def test_dt_property(self):
        """dt property returns time with units."""
        arr = TimedArray([0, 1], dt=10*ms)
        assert_allclose(float(np.asarray(arr.dt)), 0.01)

    def test_1d_values(self):
        """1D values work correctly."""
        values = np.array([0, 1, 2, 3])
        arr = TimedArray(values, dt=1*ms)
        assert arr._ndim == 1
        assert arr._n_neurons is None

    def test_2d_values(self):
        """2D values work correctly."""
        values = np.array([[0, 1], [2, 3], [4, 5]])
        arr = TimedArray(values, dt=1*ms)
        assert arr._ndim == 2
        assert arr._n_neurons == 2

    def test_with_units(self):
        """TimedArray works with units."""
        values = np.array([0, 1, 2, 3]) * mV
        arr = TimedArray(values, dt=10*ms)
        assert arr._dim == volt.dim

    def test_invalid_dt_raises(self):
        """Invalid dt raises error."""
        with pytest.raises(ValueError):
            TimedArray([0, 1], dt=0*ms)


class TestTimedArrayCall:
    """Tests for calling TimedArray."""

    def test_call_scalar_time(self):
        """Call with scalar time returns correct value."""
        values = np.array([0, 10, 20, 30])
        arr = TimedArray(values, dt=10*ms)

        assert arr(5*ms) == 0
        assert arr(15*ms) == 10
        assert arr(25*ms) == 20

    def test_call_with_units(self):
        """Call with units returns Quantity."""
        values = np.array([0, 1, 2]) * mV
        arr = TimedArray(values, dt=10*ms)

        result = arr(15*ms)
        assert isinstance(result, Quantity)
        assert result.dim == volt.dim

    def test_call_array_time(self):
        """Call with array of times returns array."""
        values = np.array([0, 10, 20])
        arr = TimedArray(values, dt=10*ms)

        times = np.array([5, 15, 25]) * ms
        result = arr(times)
        assert_array_equal(result, [0, 10, 20])

    def test_call_clamps_to_bounds(self):
        """Values outside range are clamped."""
        values = np.array([0, 1, 2])
        arr = TimedArray(values, dt=10*ms)

        assert arr(-5*ms) == 0
        assert arr(100*ms) == 2

    def test_call_2d_returns_row(self):
        """Calling 2D array at scalar time returns row."""
        values = np.array([[0, 1], [2, 3], [4, 5]])
        arr = TimedArray(values, dt=10*ms)

        result = arr(15*ms)
        assert_array_equal(result, [2, 3])


class TestTimedArrayProperties:
    """Tests for properties."""

    def test_values_property(self):
        """values property returns stored values."""
        values = np.array([1, 2, 3])
        arr = TimedArray(values, dt=1*ms)
        assert_array_equal(arr.values, [1, 2, 3])

    def test_values_with_units(self):
        """values property preserves units."""
        values = np.array([1, 2, 3]) * mV
        arr = TimedArray(values, dt=1*ms)

        result = arr.values
        assert isinstance(result, Quantity)
        assert result.dim == volt.dim

    def test_dt_raw(self):
        """dt_ returns raw value in seconds."""
        arr = TimedArray([0, 1], dt=10*ms)
        assert_allclose(arr.dt_, 0.01)


class TestTimedArrayDimensionless:
    """Tests for dimensionless values."""

    def test_dimensionless_values(self):
        """Dimensionless values work correctly."""
        values = np.array([0.5, 1.0, 1.5])
        arr = TimedArray(values, dt=10*ms)

        result = arr(15*ms)
        assert not isinstance(result, Quantity)
        assert result == 1.0


class TestTimedArrayRepr:
    """Tests for repr."""

    def test_repr_1d(self):
        """1D TimedArray has repr."""
        arr = TimedArray([0, 1, 2, 3], dt=10*ms, name='test')
        r = repr(arr)
        assert 'TimedArray' in r
        assert 'test' in r
        assert '4' in r

    def test_repr_2d(self):
        """2D TimedArray has repr."""
        arr = TimedArray(np.zeros((3, 5)), dt=10*ms, name='test2d')
        r = repr(arr)
        assert '3' in r
        assert '5' in r
