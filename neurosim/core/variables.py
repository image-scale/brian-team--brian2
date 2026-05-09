"""
Variable classes for storing and managing state variables.

This module provides classes for representing variables in neural models,
including state arrays, constants, and computed subexpressions.
"""

import numpy as np
from collections.abc import Mapping

from neurosim.units.core import DIMENSIONLESS, Dimension, Quantity, get_dimensions
from neurosim.utils.logger import get_logger

__all__ = [
    'Variable',
    'Constant',
    'ArrayVariable',
    'DynamicArrayVariable',
    'Subexpression',
    'Variables',
]

logger = get_logger(__name__)


def get_dtype(obj):
    """Get the numpy dtype of an object."""
    if hasattr(obj, 'dtype'):
        return obj.dtype
    return np.dtype(type(obj))


class Variable:
    """
    Base class for all variables.

    Parameters
    ----------
    name : str
        The variable name.
    dimensions : Dimension, optional
        Physical dimensions. Defaults to DIMENSIONLESS.
    owner : object, optional
        The object that owns this variable.
    dtype : dtype, optional
        The numpy dtype. Defaults to float64.
    scalar : bool, optional
        Whether this is a scalar (True) or array (False). Defaults to False.
    constant : bool, optional
        Whether the value is constant during a run. Defaults to False.
    read_only : bool, optional
        Whether the value can be modified by users. Defaults to False.
    """

    def __init__(self, name, dimensions=DIMENSIONLESS, owner=None, dtype=None,
                 scalar=False, constant=False, read_only=False):
        self.name = name
        self.dim = get_dimensions(dimensions)
        self.owner = owner
        self.dtype = dtype if dtype is not None else np.float64
        self.scalar = scalar
        self.constant = constant
        self.read_only = read_only

        if self.is_boolean and self.dim != DIMENSIONLESS:
            raise ValueError("Boolean variables must be dimensionless")

    @property
    def is_boolean(self):
        """Whether this is a boolean variable."""
        return np.issubdtype(self.dtype, np.bool_)

    @property
    def is_integer(self):
        """Whether this is an integer variable."""
        return np.issubdtype(self.dtype, np.integer)

    def get_value(self):
        """Get the variable's value (without units)."""
        raise NotImplementedError("Subclasses must implement get_value")

    def set_value(self, value):
        """Set the variable's value."""
        raise NotImplementedError("Subclasses must implement set_value")

    def get_value_with_unit(self):
        """Get the variable's value with units."""
        return Quantity(self.get_value(), dim=self.dim)

    def __repr__(self):
        return (f"<{self.__class__.__name__}(name={self.name!r}, "
                f"dim={self.dim}, dtype={self.dtype}, "
                f"scalar={self.scalar}, constant={self.constant})>")


class Constant(Variable):
    """
    A constant scalar value.

    Parameters
    ----------
    name : str
        The variable name.
    value : number
        The constant value.
    dimensions : Dimension, optional
        Physical dimensions. Defaults to DIMENSIONLESS.
    owner : object, optional
        The object that owns this variable.
    """

    def __init__(self, name, value, dimensions=DIMENSIONLESS, owner=None):
        is_bool = value is True or value is False or value is np.True_ or value is np.False_

        if is_bool:
            dtype = np.dtype(bool)
        else:
            dtype = get_dtype(value)

        if hasattr(value, 'shape') and value.shape == ():
            if np.can_cast(value.dtype, int):
                value = int(value)
            elif np.can_cast(value.dtype, float):
                value = float(value)

        self._value = value

        super().__init__(name, dimensions=dimensions, owner=owner, dtype=dtype,
                        scalar=True, constant=True, read_only=True)

    def get_value(self):
        """Get the constant value."""
        return self._value

    def set_value(self, value):
        """Constants cannot be set."""
        raise TypeError(f"Cannot set value of constant '{self.name}'")


class ArrayVariable(Variable):
    """
    A variable stored in a numpy array.

    Parameters
    ----------
    name : str
        The variable name.
    size : int
        The size of the array.
    dimensions : Dimension, optional
        Physical dimensions. Defaults to DIMENSIONLESS.
    owner : object, optional
        The object that owns this variable.
    dtype : dtype, optional
        The numpy dtype. Defaults to float64.
    constant : bool, optional
        Whether the value is constant. Defaults to False.
    read_only : bool, optional
        Whether the value is read-only. Defaults to False.
    scalar : bool, optional
        Whether this is a scalar (1-element array). Defaults to False.
    values : array-like, optional
        Initial values for the array.
    """

    def __init__(self, name, size, dimensions=DIMENSIONLESS, owner=None,
                 dtype=None, constant=False, read_only=False, scalar=False, values=None):
        self.size = size
        dtype = dtype if dtype is not None else np.float64

        if values is not None:
            self._data = np.asarray(values, dtype=dtype)
            if len(self._data) != size:
                raise ValueError(f"Initial values length {len(self._data)} does not match size {size}")
        else:
            self._data = np.zeros(size, dtype=dtype)

        super().__init__(name, dimensions=dimensions, owner=owner, dtype=dtype,
                        scalar=scalar, constant=constant, read_only=read_only)

    def get_value(self):
        """Get the array data."""
        return self._data

    def set_value(self, value):
        """Set the array data."""
        if self.read_only:
            raise TypeError(f"Cannot set read-only variable '{self.name}'")
        if self.constant:
            raise TypeError(f"Cannot modify constant variable '{self.name}'")

        value = np.asarray(value, dtype=self.dtype)
        if value.shape == ():
            self._data[:] = value
        else:
            self._data[:] = value

    def __len__(self):
        return self.size


class DynamicArrayVariable(Variable):
    """
    A variable stored in a dynamically-sized array.

    Used for variables like synapse weights where the size may change.

    Parameters
    ----------
    name : str
        The variable name.
    dimensions : Dimension, optional
        Physical dimensions. Defaults to DIMENSIONLESS.
    owner : object, optional
        The object that owns this variable.
    dtype : dtype, optional
        The numpy dtype. Defaults to float64.
    """

    def __init__(self, name, dimensions=DIMENSIONLESS, owner=None, dtype=None):
        dtype = dtype if dtype is not None else np.float64
        self._data = np.array([], dtype=dtype)

        super().__init__(name, dimensions=dimensions, owner=owner, dtype=dtype,
                        scalar=False, constant=False, read_only=False)

    @property
    def size(self):
        """Current size of the array."""
        return len(self._data)

    def get_value(self):
        """Get the array data."""
        return self._data

    def set_value(self, value):
        """Set the array data."""
        self._data = np.asarray(value, dtype=self.dtype)

    def resize(self, new_size):
        """Resize the array, preserving existing data."""
        if new_size > len(self._data):
            new_data = np.zeros(new_size, dtype=self.dtype)
            new_data[:len(self._data)] = self._data
            self._data = new_data
        else:
            self._data = self._data[:new_size].copy()

    def __len__(self):
        return len(self._data)


class Subexpression(Variable):
    """
    A computed variable based on an expression.

    Parameters
    ----------
    name : str
        The variable name.
    expr : str
        The expression defining this variable.
    dimensions : Dimension, optional
        Physical dimensions. Defaults to DIMENSIONLESS.
    owner : object, optional
        The object that owns this variable.
    dtype : dtype, optional
        The numpy dtype. Defaults to float64.
    scalar : bool, optional
        Whether this is a scalar expression. Defaults to False.
    """

    def __init__(self, name, expr, dimensions=DIMENSIONLESS, owner=None,
                 dtype=None, scalar=False):
        self.expr = expr.strip()

        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        import re
        reserved = {'and', 'or', 'not', 'True', 'False', 'None', 'if', 'else'}
        self.identifiers = set(re.findall(pattern, expr)) - reserved

        super().__init__(name, dimensions=dimensions, owner=owner,
                        dtype=dtype if dtype else np.float64,
                        scalar=scalar, constant=False, read_only=True)

    def get_value(self):
        """Subexpressions must be evaluated in context."""
        raise NotImplementedError(
            f"Subexpression '{self.name}' must be evaluated in group context"
        )

    def set_value(self, value):
        """Subexpressions cannot be set directly."""
        raise TypeError(f"Cannot set subexpression '{self.name}' directly")

    def __repr__(self):
        return (f"<Subexpression(name={self.name!r}, "
                f"expr={self.expr!r}, dim={self.dim})>")


class Variables(Mapping):
    """
    Container for managing multiple variables.

    Parameters
    ----------
    owner : object, optional
        The object that owns these variables.
    default_dtype : dtype, optional
        Default dtype for new variables.
    """

    def __init__(self, owner=None, default_dtype=None):
        self.owner = owner
        self.default_dtype = default_dtype if default_dtype else np.float64
        self._variables = {}

    def add_constant(self, name, value, dimensions=DIMENSIONLESS):
        """
        Add a constant variable.

        Parameters
        ----------
        name : str
            Variable name.
        value : number
            The constant value.
        dimensions : Dimension, optional
            Physical dimensions.
        """
        if name in self._variables:
            raise KeyError(f"Variable '{name}' already exists")

        var = Constant(name, value, dimensions=dimensions, owner=self.owner)
        self._variables[name] = var
        logger.debug(f"Added constant '{name}' with value {value}")

    def add_array(self, name, size, dimensions=DIMENSIONLESS, dtype=None,
                  constant=False, read_only=False, scalar=False, values=None):
        """
        Add an array variable.

        Parameters
        ----------
        name : str
            Variable name.
        size : int
            Array size.
        dimensions : Dimension, optional
            Physical dimensions.
        dtype : dtype, optional
            Array dtype.
        constant : bool, optional
            Whether constant.
        read_only : bool, optional
            Whether read-only.
        scalar : bool, optional
            Whether scalar (1-element).
        values : array-like, optional
            Initial values.
        """
        if name in self._variables:
            raise KeyError(f"Variable '{name}' already exists")

        dtype = dtype if dtype else self.default_dtype
        var = ArrayVariable(name, size, dimensions=dimensions, owner=self.owner,
                           dtype=dtype, constant=constant, read_only=read_only,
                           scalar=scalar, values=values)

        self._variables[name] = var
        logger.debug(f"Added array '{name}' with size {size}")

    def add_dynamic_array(self, name, dimensions=DIMENSIONLESS, dtype=None):
        """
        Add a dynamic array variable.

        Parameters
        ----------
        name : str
            Variable name.
        dimensions : Dimension, optional
            Physical dimensions.
        dtype : dtype, optional
            Array dtype.
        """
        if name in self._variables:
            raise KeyError(f"Variable '{name}' already exists")

        dtype = dtype if dtype else self.default_dtype
        var = DynamicArrayVariable(name, dimensions=dimensions,
                                   owner=self.owner, dtype=dtype)
        self._variables[name] = var
        logger.debug(f"Added dynamic array '{name}'")

    def add_subexpression(self, name, expr, dimensions=DIMENSIONLESS,
                         dtype=None, scalar=False):
        """
        Add a subexpression variable.

        Parameters
        ----------
        name : str
            Variable name.
        expr : str
            The expression.
        dimensions : Dimension, optional
            Physical dimensions.
        dtype : dtype, optional
            Result dtype.
        scalar : bool, optional
            Whether scalar.
        """
        if name in self._variables:
            raise KeyError(f"Variable '{name}' already exists")

        dtype = dtype if dtype else self.default_dtype
        var = Subexpression(name, expr, dimensions=dimensions,
                           owner=self.owner, dtype=dtype, scalar=scalar)
        self._variables[name] = var
        logger.debug(f"Added subexpression '{name}' = {expr}")

    def add(self, name, var):
        """
        Add an existing Variable object.

        Parameters
        ----------
        name : str
            Variable name.
        var : Variable
            The variable to add.
        """
        if name in self._variables:
            raise KeyError(f"Variable '{name}' already exists")
        self._variables[name] = var

    def __getitem__(self, key):
        return self._variables[key]

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __contains__(self, key):
        return key in self._variables

    def __repr__(self):
        names = ', '.join(self._variables.keys())
        return f"<Variables({names})>"
