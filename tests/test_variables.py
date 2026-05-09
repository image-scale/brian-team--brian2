"""
Tests for the state variables system.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.core import (
    Variable,
    Constant,
    ArrayVariable,
    DynamicArrayVariable,
    Subexpression,
    Variables,
)
from neurosim.units.core import DIMENSIONLESS, get_dimensions
from neurosim.units.baseunits import volt, second, ampere


class TestVariable:
    """Tests for the Variable base class."""

    def test_variable_name(self):
        """Variable stores name."""
        var = Constant('test', 1.0)
        assert var.name == 'test'

    def test_variable_dimensions(self):
        """Variable stores dimensions."""
        var = Constant('v', 1.0, dimensions=volt.dim)
        assert var.dim == volt.dim

    def test_variable_dtype(self):
        """Variable stores dtype."""
        var = ArrayVariable('x', 10, dtype=np.float32)
        assert var.dtype == np.float32

    def test_variable_scalar(self):
        """Variable tracks scalar flag."""
        var = Constant('N', 100)
        assert var.scalar is True

    def test_variable_constant(self):
        """Variable tracks constant flag."""
        var = Constant('c', 1.0)
        assert var.constant is True

    def test_variable_read_only(self):
        """Variable tracks read_only flag."""
        var = Constant('c', 1.0)
        assert var.read_only is True

    def test_boolean_must_be_dimensionless(self):
        """Boolean variables must be dimensionless."""
        with pytest.raises(ValueError):
            ArrayVariable('b', 10, dimensions=volt.dim, dtype=bool)

    def test_is_boolean(self):
        """is_boolean property works."""
        var = ArrayVariable('b', 10, dtype=bool)
        assert var.is_boolean is True

        var2 = ArrayVariable('x', 10, dtype=float)
        assert var2.is_boolean is False

    def test_is_integer(self):
        """is_integer property works."""
        var = ArrayVariable('n', 10, dtype=int)
        assert var.is_integer is True

        var2 = ArrayVariable('x', 10, dtype=float)
        assert var2.is_integer is False


class TestConstant:
    """Tests for the Constant class."""

    def test_constant_value(self):
        """Constant stores and returns value."""
        c = Constant('N', 100)
        assert c.get_value() == 100

    def test_constant_is_scalar(self):
        """Constants are scalar."""
        c = Constant('x', 1.0)
        assert c.scalar is True

    def test_constant_is_constant(self):
        """Constants are constant."""
        c = Constant('x', 1.0)
        assert c.constant is True

    def test_constant_is_readonly(self):
        """Constants are read-only."""
        c = Constant('x', 1.0)
        assert c.read_only is True

    def test_constant_cannot_set(self):
        """Cannot set constant value."""
        c = Constant('x', 1.0)
        with pytest.raises(TypeError):
            c.set_value(2.0)

    def test_constant_with_dimensions(self):
        """Constant can have dimensions."""
        c = Constant('tau', 0.01, dimensions=second.dim)
        assert c.dim == second.dim

    def test_constant_with_unit(self):
        """Constant get_value_with_unit works."""
        c = Constant('tau', 0.01, dimensions=second.dim)
        v = c.get_value_with_unit()
        assert v.dim == second.dim
        assert_allclose(float(v), 0.01)

    def test_constant_bool(self):
        """Boolean constant works."""
        c = Constant('flag', True)
        assert c.get_value() is True
        assert c.is_boolean is True

    def test_constant_int(self):
        """Integer constant works."""
        c = Constant('N', 100)
        assert c.get_value() == 100
        assert c.is_integer is True


class TestArrayVariable:
    """Tests for the ArrayVariable class."""

    def test_array_creation(self):
        """ArrayVariable creates array of correct size."""
        var = ArrayVariable('v', 100)
        assert len(var) == 100

    def test_array_default_zeros(self):
        """ArrayVariable defaults to zeros."""
        var = ArrayVariable('v', 10)
        assert_array_equal(var.get_value(), np.zeros(10))

    def test_array_get_value(self):
        """ArrayVariable returns numpy array."""
        var = ArrayVariable('v', 10)
        val = var.get_value()
        assert isinstance(val, np.ndarray)
        assert len(val) == 10

    def test_array_set_value_scalar(self):
        """Setting scalar broadcasts to all elements."""
        var = ArrayVariable('v', 10)
        var.set_value(5.0)
        assert_array_equal(var.get_value(), np.full(10, 5.0))

    def test_array_set_value_array(self):
        """Setting array copies values."""
        var = ArrayVariable('v', 5)
        var.set_value([1, 2, 3, 4, 5])
        assert_array_equal(var.get_value(), [1, 2, 3, 4, 5])

    def test_array_dtype(self):
        """ArrayVariable uses specified dtype."""
        var = ArrayVariable('x', 10, dtype=np.float32)
        assert var.get_value().dtype == np.float32

    def test_array_with_dimensions(self):
        """ArrayVariable can have dimensions."""
        var = ArrayVariable('v', 10, dimensions=volt.dim)
        assert var.dim == volt.dim

    def test_array_readonly_cannot_set(self):
        """Read-only array cannot be set."""
        var = ArrayVariable('t', 10, read_only=True)
        with pytest.raises(TypeError):
            var.set_value(1.0)

    def test_array_constant_cannot_set(self):
        """Constant array cannot be modified."""
        var = ArrayVariable('c', 10, constant=True)
        with pytest.raises(TypeError):
            var.set_value(1.0)

    def test_array_initial_values(self):
        """ArrayVariable with initial values via Variables container."""
        vars = Variables()
        vars.add_array('x', 5, values=[1, 2, 3, 4, 5])
        assert_array_equal(vars['x'].get_value(), [1, 2, 3, 4, 5])

    def test_array_get_value_with_unit(self):
        """get_value_with_unit returns Quantity."""
        var = ArrayVariable('v', 10, dimensions=volt.dim)
        var.set_value(np.arange(10) * 0.001)
        val = var.get_value_with_unit()
        assert val.dim == volt.dim


class TestDynamicArrayVariable:
    """Tests for DynamicArrayVariable."""

    def test_dynamic_initial_empty(self):
        """DynamicArrayVariable starts empty."""
        var = DynamicArrayVariable('w')
        assert len(var) == 0

    def test_dynamic_set_value(self):
        """Can set value of dynamic array."""
        var = DynamicArrayVariable('w')
        var.set_value([1, 2, 3])
        assert len(var) == 3
        assert_array_equal(var.get_value(), [1, 2, 3])

    def test_dynamic_resize_larger(self):
        """Resize to larger preserves data."""
        var = DynamicArrayVariable('w')
        var.set_value([1, 2, 3])
        var.resize(5)
        assert len(var) == 5
        assert_array_equal(var.get_value()[:3], [1, 2, 3])
        assert_array_equal(var.get_value()[3:], [0, 0])

    def test_dynamic_resize_smaller(self):
        """Resize to smaller truncates."""
        var = DynamicArrayVariable('w')
        var.set_value([1, 2, 3, 4, 5])
        var.resize(3)
        assert len(var) == 3
        assert_array_equal(var.get_value(), [1, 2, 3])

    def test_dynamic_dtype(self):
        """DynamicArrayVariable uses specified dtype."""
        var = DynamicArrayVariable('w', dtype=np.float32)
        var.set_value([1.0, 2.0])
        assert var.get_value().dtype == np.float32


class TestSubexpression:
    """Tests for Subexpression class."""

    def test_subexpression_stores_expr(self):
        """Subexpression stores expression."""
        sub = Subexpression('I_total', 'I_syn + I_ext')
        assert sub.expr == 'I_syn + I_ext'

    def test_subexpression_identifiers(self):
        """Subexpression extracts identifiers."""
        sub = Subexpression('I_total', 'I_syn + I_ext * weight')
        assert 'I_syn' in sub.identifiers
        assert 'I_ext' in sub.identifiers
        assert 'weight' in sub.identifiers

    def test_subexpression_is_readonly(self):
        """Subexpressions are read-only."""
        sub = Subexpression('x', 'a + b')
        assert sub.read_only is True

    def test_subexpression_cannot_set(self):
        """Cannot set subexpression directly."""
        sub = Subexpression('x', 'a + b')
        with pytest.raises(TypeError):
            sub.set_value(1.0)

    def test_subexpression_get_value_raises(self):
        """get_value raises for subexpressions."""
        sub = Subexpression('x', 'a + b')
        with pytest.raises(NotImplementedError):
            sub.get_value()

    def test_subexpression_with_dimensions(self):
        """Subexpression can have dimensions."""
        sub = Subexpression('v_total', 'v + v_rest', dimensions=volt.dim)
        assert sub.dim == volt.dim


class TestVariables:
    """Tests for Variables container."""

    def test_add_constant(self):
        """Can add constant to Variables."""
        vars = Variables()
        vars.add_constant('N', 100)
        assert 'N' in vars
        assert vars['N'].get_value() == 100

    def test_add_array(self):
        """Can add array to Variables."""
        vars = Variables()
        vars.add_array('v', 100, dimensions=volt.dim)
        assert 'v' in vars
        assert len(vars['v']) == 100

    def test_add_dynamic_array(self):
        """Can add dynamic array to Variables."""
        vars = Variables()
        vars.add_dynamic_array('w')
        assert 'w' in vars
        assert isinstance(vars['w'], DynamicArrayVariable)

    def test_add_subexpression(self):
        """Can add subexpression to Variables."""
        vars = Variables()
        vars.add_subexpression('x', 'a + b')
        assert 'x' in vars
        assert isinstance(vars['x'], Subexpression)

    def test_duplicate_raises(self):
        """Adding duplicate name raises."""
        vars = Variables()
        vars.add_constant('x', 1.0)
        with pytest.raises(KeyError):
            vars.add_constant('x', 2.0)

    def test_len(self):
        """len() returns number of variables."""
        vars = Variables()
        vars.add_constant('a', 1)
        vars.add_array('b', 10)
        assert len(vars) == 2

    def test_iter(self):
        """Can iterate over variable names."""
        vars = Variables()
        vars.add_constant('a', 1)
        vars.add_constant('b', 2)
        names = list(vars)
        assert 'a' in names
        assert 'b' in names

    def test_contains(self):
        """Can check if variable exists."""
        vars = Variables()
        vars.add_constant('x', 1)
        assert 'x' in vars
        assert 'y' not in vars

    def test_add_variable_object(self):
        """Can add existing Variable object."""
        vars = Variables()
        c = Constant('x', 1.0)
        vars.add('x', c)
        assert vars['x'] is c

    def test_default_dtype(self):
        """Variables uses default dtype."""
        vars = Variables(default_dtype=np.float32)
        vars.add_array('x', 10)
        assert vars['x'].dtype == np.float32


class TestVariableDimensions:
    """Tests for dimension handling in variables."""

    def test_constant_dimension(self):
        """Constant preserves dimension."""
        c = Constant('tau', 0.01, dimensions=second.dim)
        assert c.dim == second.dim

    def test_array_dimension(self):
        """Array preserves dimension."""
        var = ArrayVariable('v', 10, dimensions=volt.dim)
        assert var.dim == volt.dim

    def test_subexpr_dimension(self):
        """Subexpression preserves dimension."""
        sub = Subexpression('I', 'g * V', dimensions=ampere.dim)
        assert sub.dim == ampere.dim

    def test_dimensionless_default(self):
        """Variables default to dimensionless."""
        c = Constant('x', 1.0)
        assert c.dim == DIMENSIONLESS
