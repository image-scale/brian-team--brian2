"""
Group base class for neural models.

Groups manage collections of elements (neurons, synapses) with state variables.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.core.variables import Variables, Constant, ArrayVariable, Subexpression
from neurosim.equations import Equations, PARAMETER, DIFFERENTIAL_EQUATION, SUBEXPRESSION
from neurosim.units.core import DIMENSIONLESS, Quantity, get_dimensions, fail_for_dimension_mismatch
from neurosim.utils.logger import get_logger

__all__ = ['Group']

logger = get_logger(__name__)


class GroupView:
    """
    A view into a subset of a Group.

    Provides indexing into group state variables.

    Parameters
    ----------
    group : Group
        The parent group.
    index : array-like
        The indices into the group.
    """

    def __init__(self, group, index):
        self.group = group
        if isinstance(index, slice):
            self._indices = np.arange(*index.indices(len(group)))
        elif isinstance(index, (int, np.integer)):
            self._indices = np.array([index])
        elif hasattr(index, 'dtype') and index.dtype == bool:
            self._indices = np.nonzero(index)[0]
        else:
            self._indices = np.asarray(index)

    def __len__(self):
        return len(self._indices)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        use_units = not name.endswith('_')
        var_name = name[:-1] if name.endswith('_') else name

        if var_name not in self.group.variables:
            raise AttributeError(f"No variable '{var_name}' in group")

        var = self.group.variables[var_name]
        values = var.get_value()[self._indices]

        if use_units and var.dim != DIMENSIONLESS:
            return Quantity(values, dim=var.dim)
        return values

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('group',):
            object.__setattr__(self, name, value)
            return

        use_units = not name.endswith('_')
        var_name = name[:-1] if name.endswith('_') else name

        if var_name not in self.group.variables:
            raise AttributeError(f"No variable '{var_name}' in group")

        var = self.group.variables[var_name]

        if var.read_only:
            raise TypeError(f"Variable '{var_name}' is read-only")

        if use_units:
            fail_for_dimension_mismatch(value, var.dim,
                f"Cannot set '{var_name}' with value of different dimensions")

        if isinstance(value, Quantity):
            value = np.asarray(value)

        data = var.get_value()
        data[self._indices] = value


class Group(SimObject):
    """
    Base class for groups of neurons or other elements.

    Groups manage state variables and provide attribute access to them.
    Subclasses like NeuronGroup add specific functionality.

    Parameters
    ----------
    N : int
        Number of elements in the group.
    equations : str or Equations, optional
        Equations defining the state variables.
    namespace : dict, optional
        Additional namespace for equations.
    dtype : dtype, optional
        Default dtype for state variables.
    name : str, optional
        Name of the group.
    """

    def __init__(self, N, equations=None, namespace=None, dtype=None, name='group*', **kwargs):
        super().__init__(name=name, **kwargs)

        self._N = int(N)
        self._namespace = namespace or {}
        self._dtype = dtype or np.float64

        self.variables = Variables(owner=self, default_dtype=self._dtype)
        self.variables.add_constant('N', self._N)
        self.variables.add_array('i', self._N, dtype=np.int32, read_only=True,
                                values=np.arange(self._N))

        if equations is not None:
            self.equations = Equations(equations) if isinstance(equations, str) else equations
            self._create_variables_from_equations()
        else:
            self.equations = Equations('')

        self._group_attribute_access_active = True

        logger.debug(f"Created Group '{self.name}' with N={self._N}")

    def _create_variables_from_equations(self):
        """Create Variables from parsed Equations."""
        for name, eq in self.equations.items():
            if name in self.variables:
                continue

            if eq.type == PARAMETER:
                self.variables.add_array(name, self._N,
                                        dimensions=eq.dim,
                                        constant='constant' in eq.flags)
            elif eq.type == DIFFERENTIAL_EQUATION:
                self.variables.add_array(name, self._N,
                                        dimensions=eq.dim)
            elif eq.type == SUBEXPRESSION:
                self.variables.add_subexpression(name, eq.expr.code,
                                                dimensions=eq.dim)

    @property
    def N(self):
        """Number of elements in the group."""
        return self._N

    def __len__(self):
        return self._N

    def __getitem__(self, item):
        """Return a view into a subset of the group."""
        return GroupView(self, item)

    def __getattr__(self, name):
        if name.startswith('_') or name in ('variables', 'equations', 'clock', 'name', 'id'):
            raise AttributeError(name)

        if not getattr(self, '_group_attribute_access_active', False):
            raise AttributeError(name)

        use_units = not name.endswith('_')
        var_name = name[:-1] if name.endswith('_') else name

        if var_name not in self.variables:
            raise AttributeError(f"No attribute '{name}' (and no variable '{var_name}')")

        var = self.variables[var_name]

        if isinstance(var, Subexpression):
            raise NotImplementedError(
                f"Subexpression '{var_name}' must be evaluated in context"
            )

        values = var.get_value()

        if var.scalar:
            if use_units and var.dim != DIMENSIONLESS:
                return Quantity(values, dim=var.dim)
            return values

        if use_units and var.dim != DIMENSIONLESS:
            return Quantity(values, dim=var.dim)
        return values

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('variables', 'equations', 'clock',
                                             'name', 'id', 'when', 'order', 'active'):
            object.__setattr__(self, name, value)
            return

        if not getattr(self, '_group_attribute_access_active', False):
            object.__setattr__(self, name, value)
            return

        use_units = not name.endswith('_')
        var_name = name[:-1] if name.endswith('_') else name

        if var_name in self.variables:
            var = self.variables[var_name]

            if var.read_only:
                raise TypeError(f"Variable '{var_name}' is read-only")

            if isinstance(var, Subexpression):
                raise TypeError(f"Cannot set subexpression '{var_name}'")

            if use_units:
                fail_for_dimension_mismatch(value, var.dim,
                    f"Cannot set '{var_name}' with value of different dimensions")

            if isinstance(value, Quantity):
                value = np.asarray(value)

            var.set_value(value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' of {self._N} elements>"
