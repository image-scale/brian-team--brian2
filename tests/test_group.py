"""
Tests for the Group base class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neurosim.groups import Group
from neurosim.core import Variables, ArrayVariable, Constant
from neurosim.units import second, volt, ampere
from neurosim.units.stdunits import ms, mV
from neurosim.units.core import DIMENSIONLESS, DimensionMismatchError, Quantity


class TestGroupCreation:
    """Tests for Group creation."""

    def test_group_size(self):
        """Group has correct size."""
        g = Group(100)
        assert len(g) == 100
        assert g.N == 100

    def test_group_name(self):
        """Group has name."""
        g = Group(10, name='mygroup')
        assert g.name == 'mygroup'

    def test_group_auto_name(self):
        """Group gets auto-generated name."""
        g = Group(10)
        assert 'group' in g.name

    def test_group_has_variables(self):
        """Group has variables container."""
        g = Group(10)
        assert hasattr(g, 'variables')
        assert isinstance(g.variables, Variables)

    def test_group_N_constant(self):
        """Group has N variable."""
        g = Group(100)
        assert 'N' in g.variables
        assert g.variables['N'].get_value() == 100

    def test_group_i_index(self):
        """Group has i index variable."""
        g = Group(10)
        assert 'i' in g.variables
        assert_array_equal(g.variables['i'].get_value(), np.arange(10))


class TestGroupEquations:
    """Tests for Group with equations."""

    def test_group_with_equations(self):
        """Group creates variables from equations."""
        g = Group(10, equations='''
            v : volt
            tau : second
        ''')
        assert 'v' in g.variables
        assert 'tau' in g.variables

    def test_parameter_dimensions(self):
        """Parameters have correct dimensions."""
        g = Group(10, equations='tau : second')
        assert g.variables['tau'].dim == second.dim

    def test_diff_eq_creates_variable(self):
        """Differential equations create array variables."""
        g = Group(10, equations='dv/dt = -v/tau : volt')
        assert 'v' in g.variables
        assert g.variables['v'].dim == volt.dim

    def test_subexpression_created(self):
        """Subexpressions are created."""
        g = Group(10, equations='''
            v : volt
            v_squared = v * v : volt**2
        ''')
        assert 'v_squared' in g.variables


class TestGroupAttributeAccess:
    """Tests for attribute access to variables."""

    def test_get_variable_with_units(self):
        """Accessing variable returns Quantity."""
        g = Group(10, equations='v : volt')
        g.variables['v'].set_value(np.arange(10) * 0.001)
        v = g.v
        assert isinstance(v, Quantity)
        assert v.dim == volt.dim

    def test_get_variable_without_units(self):
        """Accessing variable_ returns array."""
        g = Group(10, equations='v : volt')
        g.variables['v'].set_value(np.arange(10) * 0.001)
        v = g.v_
        assert isinstance(v, np.ndarray)
        assert not isinstance(v, Quantity)

    def test_set_variable_with_units(self):
        """Setting variable with units works."""
        g = Group(10, equations='v : volt')
        g.v = np.ones(10) * mV
        assert_allclose(g.v_, np.ones(10) * 0.001)

    def test_set_variable_without_units(self):
        """Setting variable_ works."""
        g = Group(10, equations='v : volt')
        g.v_ = np.ones(10) * 0.001
        assert_allclose(g.v_, np.ones(10) * 0.001)

    def test_set_variable_scalar(self):
        """Setting with scalar broadcasts."""
        g = Group(10, equations='v : volt')
        g.v = 5 * mV
        assert_allclose(g.v_, np.ones(10) * 0.005)

    def test_set_wrong_dimension_raises(self):
        """Setting with wrong dimension raises error."""
        g = Group(10, equations='v : volt')
        with pytest.raises(DimensionMismatchError):
            g.v = 1 * second

    def test_dimensionless_variable(self):
        """Dimensionless variables work."""
        g = Group(10, equations='x : 1')
        g.x = np.arange(10)
        assert_array_equal(g.x, np.arange(10))

    def test_unknown_attribute_raises(self):
        """Unknown attribute raises AttributeError."""
        g = Group(10)
        with pytest.raises(AttributeError):
            _ = g.unknown_var

    def test_readonly_cannot_set(self):
        """Read-only variables cannot be set."""
        g = Group(10)
        with pytest.raises(TypeError):
            g.i = np.zeros(10)


class TestGroupIndexing:
    """Tests for Group indexing."""

    def test_integer_index(self):
        """Integer indexing works."""
        g = Group(10, equations='v : volt')
        g.v = np.arange(10) * mV
        view = g[5]
        assert_allclose(view.v_, [0.005])

    def test_slice_index(self):
        """Slice indexing works."""
        g = Group(10, equations='v : volt')
        g.v = np.arange(10) * mV
        view = g[2:5]
        assert len(view) == 3
        assert_allclose(view.v_, [0.002, 0.003, 0.004])

    def test_array_index(self):
        """Array indexing works."""
        g = Group(10, equations='v : volt')
        g.v = np.arange(10) * mV
        view = g[[0, 5, 9]]
        assert len(view) == 3
        assert_allclose(view.v_, [0.0, 0.005, 0.009])

    def test_boolean_index(self):
        """Boolean indexing works."""
        g = Group(5, equations='v : volt')
        g.v = np.arange(5) * mV
        mask = np.array([True, False, True, False, True])
        view = g[mask]
        assert len(view) == 3
        assert_allclose(view.v_, [0.0, 0.002, 0.004])

    def test_index_set_variable(self):
        """Can set variable via index."""
        g = Group(10, equations='v : volt')
        g.v = np.zeros(10) * mV
        g[5].v = 1 * mV
        assert_allclose(g.v_[5], 0.001)

    def test_index_set_slice(self):
        """Can set variable via slice."""
        g = Group(10, equations='v : volt')
        g.v = np.zeros(10) * mV
        g[0:3].v = 5 * mV
        assert_allclose(g.v_[:3], [0.005, 0.005, 0.005])
        assert_allclose(g.v_[3:], np.zeros(7))


class TestGroupVariables:
    """Tests for variable management in Group."""

    def test_constant_flag(self):
        """Constant flag creates constant variable."""
        g = Group(10, equations='tau : second (constant)')
        assert g.variables['tau'].constant is True

    def test_access_N(self):
        """Can access N."""
        g = Group(100)
        assert g.N == 100

    def test_access_i(self):
        """Can access i."""
        g = Group(10)
        i = g.i
        assert_array_equal(i, np.arange(10))

    def test_multiple_variables(self):
        """Multiple variables work together."""
        g = Group(10, equations='''
            v : volt
            I : ampere
            g : 1
        ''')
        g.v = np.ones(10) * mV
        g.I = np.ones(10) * 1e-9 * ampere
        g.g = np.ones(10) * 0.5

        assert g.variables['v'].dim == volt.dim
        assert g.variables['I'].dim == ampere.dim
        assert g.variables['g'].dim == DIMENSIONLESS


class TestGroupRepr:
    """Tests for Group repr."""

    def test_repr(self):
        """Group has repr."""
        g = Group(100, name='test')
        r = repr(g)
        assert 'Group' in r
        assert 'test' in r
        assert '100' in r
