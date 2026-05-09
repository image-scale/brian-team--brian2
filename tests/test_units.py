"""
Tests for the physical units system.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.units import (
    Dimension, DimensionMismatchError, Quantity, Unit, DIMENSIONLESS,
    get_or_create_dimension, get_dimensions, is_dimensionless, have_same_dimensions,
    second, metre, meter, kilogram, ampere, amp, volt, ohm, siemens, farad, coulomb, hertz,
    kelvin, mole,
)


class TestDimension:
    """Tests for the Dimension class."""

    def test_create_dimension(self):
        """Dimensions can be created with a list of 7 exponents."""
        dim = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert dim.get_dimension('length') == 1
        assert dim.get_dimension('time') == 0

    def test_create_dimension_kwargs(self):
        """Dimensions can be created with keyword arguments."""
        dim = get_or_create_dimension(length=1, time=-2)
        assert dim.get_dimension('length') == 1
        assert dim.get_dimension('time') == -2
        assert dim.get_dimension('mass') == 0

    def test_dimension_singleton(self):
        """Same dimension specification returns the same object."""
        dim1 = get_or_create_dimension(length=1)
        dim2 = get_or_create_dimension(length=1)
        assert dim1 is dim2

    def test_dimensionless(self):
        """DIMENSIONLESS has all zero exponents."""
        assert DIMENSIONLESS.is_dimensionless
        assert DIMENSIONLESS.get_dimension('length') == 0
        assert DIMENSIONLESS.get_dimension('time') == 0

    def test_dimension_multiplication(self):
        """Multiplying dimensions adds exponents."""
        length = get_or_create_dimension(length=1)
        time = get_or_create_dimension(time=1)
        result = length * time
        assert result.get_dimension('length') == 1
        assert result.get_dimension('time') == 1

    def test_dimension_division(self):
        """Dividing dimensions subtracts exponents."""
        length = get_or_create_dimension(length=1)
        time = get_or_create_dimension(time=1)
        result = length / time
        assert result.get_dimension('length') == 1
        assert result.get_dimension('time') == -1

    def test_dimension_power(self):
        """Raising dimension to power multiplies exponents."""
        length = get_or_create_dimension(length=1)
        result = length ** 2
        assert result.get_dimension('length') == 2

    def test_dimension_equality(self):
        """Dimensions with same exponents are equal."""
        dim1 = get_or_create_dimension(length=1, time=-2)
        dim2 = get_or_create_dimension(length=1, time=-2)
        assert dim1 == dim2

    def test_dimension_inequality(self):
        """Dimensions with different exponents are not equal."""
        dim1 = get_or_create_dimension(length=1)
        dim2 = get_or_create_dimension(time=1)
        assert dim1 != dim2

    def test_dimension_str(self):
        """String representation shows dimension labels."""
        dim = get_or_create_dimension(length=1, time=-2)
        s = str(dim)
        assert 'm' in s
        assert 's' in s


class TestQuantity:
    """Tests for the Quantity class."""

    def test_quantity_construction_with_unit(self):
        """Quantities can be created by multiplying a number with a unit."""
        q = 5 * second
        assert isinstance(q, Quantity)
        assert float(q) == 5.0
        assert q.dim == second.dim

    def test_quantity_array(self):
        """Quantities can contain numpy arrays."""
        q = np.array([1, 2, 3]) * second
        assert isinstance(q, Quantity)
        assert_array_equal(np.asarray(q), [1, 2, 3])

    def test_quantity_unit_conversion(self):
        """500 milliseconds should be 0.5 when divided by second."""
        ms = 0.001 * second
        q = 500 * ms
        value_in_seconds = float(q / second)
        assert_allclose(value_in_seconds, 0.5)

    def test_quantity_addition_same_dimensions(self):
        """Quantities with same dimensions can be added."""
        q1 = 1 * second
        ms = 0.001 * second
        q2 = 500 * ms
        result = q1 + q2
        assert isinstance(result, Quantity)
        assert_allclose(float(result / second), 1.5)

    def test_quantity_addition_dimension_mismatch(self):
        """Adding quantities with different dimensions raises error."""
        q1 = 1 * second
        q2 = 1 * metre
        with pytest.raises(DimensionMismatchError):
            q1 + q2

    def test_quantity_subtraction(self):
        """Quantities with same dimensions can be subtracted."""
        q1 = 5 * second
        q2 = 2 * second
        result = q1 - q2
        assert_allclose(float(result / second), 3.0)

    def test_quantity_multiplication(self):
        """Multiplying quantities multiplies values and adds dimensions."""
        q1 = 2 * ampere
        q2 = 3 * second
        result = q1 * q2
        assert isinstance(result, Quantity)
        assert_allclose(float(result / coulomb), 6.0)
        assert result.dim == coulomb.dim

    def test_quantity_division(self):
        """Dividing quantities divides values and subtracts dimensions."""
        q1 = 10 * volt
        q2 = 2 * ampere
        result = q1 / q2
        assert isinstance(result, Quantity)
        assert_allclose(float(result / ohm), 5.0)

    def test_quantity_power(self):
        """Raising quantity to power raises value and multiplies dimensions."""
        q = 3 * metre
        result = q ** 2
        assert isinstance(result, Quantity)
        assert_allclose(float(np.asarray(result)), 9.0)
        assert result.dim.get_dimension('length') == 2

    def test_quantity_comparison_same_dimension(self):
        """Quantities with same dimensions can be compared."""
        q1 = 5 * second
        q2 = 3 * second
        assert q1 > q2
        assert q2 < q1
        assert q1 >= q2
        assert q2 <= q1
        assert q1 != q2

    def test_quantity_comparison_dimension_mismatch(self):
        """Comparing quantities with different dimensions raises error."""
        q1 = 5 * second
        q2 = 5 * metre
        with pytest.raises(DimensionMismatchError):
            q1 > q2

    def test_quantity_equality(self):
        """Equal quantities compare as equal."""
        q1 = 5 * second
        q2 = 5 * second
        assert q1 == q2

    def test_dimensionless_quantity(self):
        """Dimensionless quantities work correctly."""
        q1 = 10 * second
        q2 = 5 * second
        ratio = q1 / q2
        assert_allclose(ratio, 2.0)

    def test_with_dimensions_static_method(self):
        """Quantity.with_dimensions creates quantities with given dimensions."""
        q = Quantity.with_dimensions(5, time=1)
        assert float(q / second) == 5.0

    def test_has_same_dimensions(self):
        """has_same_dimensions checks dimension compatibility."""
        q1 = 5 * second
        q2 = 10 * second
        q3 = 5 * metre
        assert q1.has_same_dimensions(q2)
        assert not q1.has_same_dimensions(q3)

    def test_get_dimensions(self):
        """get_dimensions returns dimensions of any object."""
        q = 5 * second
        assert get_dimensions(q) == second.dim
        assert get_dimensions(5) is DIMENSIONLESS
        assert get_dimensions(np.array([1, 2, 3])) is DIMENSIONLESS

    def test_is_dimensionless(self):
        """is_dimensionless checks if object has no dimensions."""
        assert is_dimensionless(5)
        assert is_dimensionless(np.array([1, 2, 3]))
        assert not is_dimensionless(5 * second)

    def test_have_same_dimensions(self):
        """have_same_dimensions compares dimensions of two objects."""
        q1 = 5 * second
        q2 = 10 * second
        q3 = 5 * metre
        assert have_same_dimensions(q1, q2)
        assert not have_same_dimensions(q1, q3)


class TestUnit:
    """Tests for the Unit class."""

    def test_base_units_exist(self):
        """SI base units are defined."""
        assert second is not None
        assert metre is not None
        assert kilogram is not None
        assert ampere is not None
        assert kelvin is not None
        assert mole is not None

    def test_unit_dimensions(self):
        """Units have correct dimensions."""
        assert second.dim.get_dimension('time') == 1
        assert metre.dim.get_dimension('length') == 1
        assert ampere.dim.get_dimension('current') == 1

    def test_derived_unit_dimensions(self):
        """Derived units have correct combined dimensions."""
        # Volt = kg*m^2/(A*s^3)
        assert volt.dim.get_dimension('mass') == 1
        assert volt.dim.get_dimension('length') == 2
        assert volt.dim.get_dimension('time') == -3
        assert volt.dim.get_dimension('current') == -1

    def test_unit_multiplication(self):
        """Multiplying units combines dimensions correctly."""
        result = ampere * second
        assert result.dim == coulomb.dim

    def test_unit_division(self):
        """Dividing units combines dimensions correctly."""
        result = volt / ampere
        assert result.dim == ohm.dim

    def test_unit_name(self):
        """Units have names and display names."""
        assert second.name == 'second'
        assert second.dispname == 's'
        assert volt.name == 'volt'
        assert volt.dispname == 'V'

    def test_meter_metre_alias(self):
        """meter and metre are the same unit."""
        assert meter is metre
        assert meter.dim == metre.dim


class TestDimensionMismatchError:
    """Tests for DimensionMismatchError."""

    def test_error_message(self):
        """Error contains dimension information."""
        dim1 = get_or_create_dimension(time=1)
        dim2 = get_or_create_dimension(length=1)
        error = DimensionMismatchError("Test error", dim1, dim2)
        msg = str(error)
        assert "Test error" in msg
        assert "s" in msg or "time" in msg

    def test_error_single_dimension(self):
        """Error with single dimension includes it."""
        dim = get_or_create_dimension(time=1)
        error = DimensionMismatchError("Test", dim)
        msg = str(error)
        assert "Test" in msg


class TestNumpyIntegration:
    """Tests for numpy integration."""

    def test_numpy_functions_preserve_dimensions(self):
        """Numpy functions preserve dimensions where appropriate."""
        q = np.array([1, -2, 3]) * second
        result = np.abs(q)
        assert isinstance(result, Quantity)
        assert_array_equal(np.asarray(result), [1, 2, 3])
        assert result.dim == second.dim

    def test_numpy_sqrt_changes_dimensions(self):
        """sqrt changes dimensions appropriately."""
        area = 9 * (metre ** 2)
        result = np.sqrt(area)
        assert isinstance(result, Quantity)
        assert_allclose(float(result / metre), 3.0)

    def test_numpy_sin_requires_dimensionless(self):
        """Trigonometric functions require dimensionless input."""
        with pytest.raises(DimensionMismatchError):
            np.sin(5 * second)
        # But work fine with dimensionless
        result = np.sin(np.pi / 2)
        assert_allclose(result, 1.0)

    def test_numpy_array_operations(self):
        """Array operations work with quantities."""
        q = np.array([1, 2, 3]) * second
        assert_allclose(np.sum(np.asarray(q)), 6.0)
        assert_allclose(np.mean(np.asarray(q)), 2.0)
