"""
Core implementation of physical units and quantities.

Defines Dimension, Quantity, and Unit classes for handling physical quantities
with dimension checking.
"""

import functools
import numbers
import numpy as np

__all__ = [
    'Dimension',
    'DimensionMismatchError',
    'Quantity',
    'Unit',
    'DIMENSIONLESS',
    'get_or_create_dimension',
    'get_dimensions',
    'is_dimensionless',
    'have_same_dimensions',
    'fail_for_dimension_mismatch',
    'check_units',
]

# Dimension indices for SI base units
_DIMENSION_NAMES = {
    'length': 0, 'metre': 0, 'meter': 0, 'm': 0,
    'mass': 1, 'kilogram': 1, 'kg': 1,
    'time': 2, 'second': 2, 's': 2,
    'current': 3, 'ampere': 3, 'amp': 3, 'A': 3,
    'temperature': 4, 'kelvin': 4, 'K': 4,
    'substance': 5, 'mole': 5, 'mol': 5,
    'luminosity': 6, 'candela': 6, 'cd': 6,
}

_DIMENSION_LABELS = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd']
_DIMENSION_NAMES_FULL = ['metre', 'kilogram', 'second', 'amp', 'kelvin', 'mole', 'candela']

# SI prefixes
SI_PREFIXES = {
    'y': -24, 'z': -21, 'a': -18, 'f': -15, 'p': -12,
    'n': -9, 'u': -6, 'm': -3, 'c': -2, 'd': -1,
    '': 0,
    'da': 1, 'h': 2, 'k': 3, 'M': 6, 'G': 9,
    'T': 12, 'P': 15, 'E': 18, 'Z': 21, 'Y': 24,
}


class Dimension:
    """
    Represents the physical dimensions of a quantity.

    Stores the exponents of the 7 SI base dimensions:
    length, mass, time, current, temperature, substance, luminosity
    """

    __slots__ = ['_dims']

    def __init__(self, dims):
        self._dims = tuple(dims)

    def get_dimension(self, name):
        """Get the exponent for a specific dimension."""
        return self._dims[_DIMENSION_NAMES[name]]

    @property
    def is_dimensionless(self):
        """Check if all dimension exponents are zero."""
        return all(d == 0 for d in self._dims)

    @property
    def dim(self):
        """Return self for compatibility with Quantity interface."""
        return self

    def _repr_str(self, use_names=False):
        """Build string representation."""
        parts = []
        labels = _DIMENSION_NAMES_FULL if use_names else _DIMENSION_LABELS
        for i, exp in enumerate(self._dims):
            if exp != 0:
                if exp == 1:
                    parts.append(labels[i])
                else:
                    if use_names:
                        parts.append(f"{labels[i]} ** {exp}")
                    else:
                        parts.append(f"{labels[i]}^{exp}")
        if not parts:
            return "1"
        sep = " * " if use_names else " "
        return sep.join(parts)

    def __repr__(self):
        return self._repr_str(use_names=True)

    def __str__(self):
        return self._repr_str(use_names=False)

    def __mul__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        new_dims = [a + b for a, b in zip(self._dims, other._dims)]
        return get_or_create_dimension(new_dims)

    def __truediv__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        new_dims = [a - b for a, b in zip(self._dims, other._dims)]
        return get_or_create_dimension(new_dims)

    def __pow__(self, exponent):
        exponent = float(exponent)
        new_dims = [d * exponent for d in self._dims]
        return get_or_create_dimension(new_dims)

    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False
        return self._dims == other._dims

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._dims)


# Singleton storage for dimensions
_dimension_cache = {}

def get_or_create_dimension(*args, **kwargs):
    """
    Get or create a Dimension object.

    Ensures only one Dimension instance exists for each unique set of exponents.
    """
    if args:
        dims = tuple(args[0])
        if len(dims) != 7:
            raise TypeError("Need exactly 7 dimension indices")
    else:
        dims = [0] * 7
        for name, value in kwargs.items():
            if name not in _DIMENSION_NAMES:
                raise KeyError(f"Unknown dimension: {name}")
            dims[_DIMENSION_NAMES[name]] = value
        dims = tuple(dims)

    if dims not in _dimension_cache:
        _dimension_cache[dims] = Dimension(dims)
    return _dimension_cache[dims]


DIMENSIONLESS = get_or_create_dimension([0, 0, 0, 0, 0, 0, 0])


class DimensionMismatchError(Exception):
    """
    Exception raised when operations are attempted with incompatible dimensions.
    """

    def __init__(self, description, *dims):
        self.description = description
        self.dims = dims
        super().__init__(description, *dims)

    def __str__(self):
        msg = self.description
        if len(self.dims) == 1:
            msg += f" (unit is {self.dims[0]})"
        elif len(self.dims) == 2:
            msg += f" (units are {self.dims[0]} and {self.dims[1]})"
        elif len(self.dims) > 2:
            units_str = ', '.join(str(d) for d in self.dims)
            msg += f" (units are {units_str})"
        return msg


def get_dimensions(obj):
    """
    Get the dimensions of an object.

    Returns DIMENSIONLESS for plain numbers and numpy arrays.
    """
    try:
        return obj.dim
    except AttributeError:
        if isinstance(obj, (numbers.Number, np.ndarray, np.number)):
            return DIMENSIONLESS
        raise TypeError(f"Object of type {type(obj)} does not have dimensions")


def is_dimensionless(obj):
    """Check if an object is dimensionless."""
    return get_dimensions(obj) is DIMENSIONLESS


def have_same_dimensions(obj1, obj2):
    """Check if two objects have the same dimensions."""
    dim1 = get_dimensions(obj1)
    dim2 = get_dimensions(obj2)
    return dim1 is dim2 or dim1 == dim2


def _fail_for_dimension_mismatch(obj1, obj2=None, error_message=None):
    """Check dimensions match and raise DimensionMismatchError if not."""
    dim1 = get_dimensions(obj1)
    dim2 = DIMENSIONLESS if obj2 is None else get_dimensions(obj2)

    if dim1 is not dim2 and dim1 != dim2:
        # Allow zero to have any dimension
        if dim1 is DIMENSIONLESS and np.all(np.asarray(obj1) == 0):
            return dim1, dim2
        if dim2 is DIMENSIONLESS and obj2 is not None and np.all(np.asarray(obj2) == 0):
            return dim1, dim2

        if error_message is None:
            error_message = "Dimension mismatch"
        if obj2 is None:
            raise DimensionMismatchError(error_message, dim1)
        else:
            raise DimensionMismatchError(error_message, dim1, dim2)
    return dim1, dim2


def fail_for_dimension_mismatch(obj1, obj2=None, error_message=None):
    """
    Check that two objects have matching dimensions.

    Parameters
    ----------
    obj1 : Quantity or number
        First object to check.
    obj2 : Quantity or number, optional
        Second object to check. If None, checks that obj1 is dimensionless.
    error_message : str, optional
        Custom error message.

    Raises
    ------
    DimensionMismatchError
        If the dimensions don't match.
    """
    return _fail_for_dimension_mismatch(obj1, obj2, error_message)


# Lists of numpy ufuncs for proper handling
_UFUNCS_PRESERVE = ['absolute', 'rint', 'negative', 'positive', 'floor', 'ceil', 'trunc']
_UFUNCS_CHANGE = ['multiply', 'divide', 'true_divide', 'floor_divide', 'sqrt', 'square', 'reciprocal']
_UFUNCS_MATCHING = ['add', 'subtract', 'maximum', 'minimum', 'remainder', 'mod', 'fmod']
_UFUNCS_COMPARE = ['less', 'less_equal', 'greater', 'greater_equal', 'equal', 'not_equal']
_UFUNCS_LOGICAL = ['logical_and', 'logical_or', 'logical_xor', 'logical_not',
                   'isfinite', 'isinf', 'isnan', 'isreal', 'iscomplex']
_UFUNCS_DIMENSIONLESS = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                         'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                         'log', 'log2', 'log10', 'log1p', 'exp', 'exp2', 'expm1']


class Quantity(np.ndarray):
    """
    A physical quantity with associated dimensions.

    Subclasses numpy ndarray to provide dimension-aware arithmetic.
    """

    __slots__ = ['dim']
    __array_priority__ = 1000

    def __new__(cls, value, dim=None, dtype=None, copy=False):
        if dim is DIMENSIONLESS:
            # Return plain array for dimensionless
            if copy:
                arr = np.array(value, dtype=dtype)
            else:
                arr = np.asarray(value, dtype=dtype)
            if arr.shape == ():
                return arr.item()
            return arr

        if copy:
            arr = np.array(value, dtype=dtype).view(cls)
        else:
            arr = np.asarray(value, dtype=dtype).view(cls)

        if dim is not None:
            arr.dim = dim
        elif hasattr(value, 'dim'):
            arr.dim = value.dim
        else:
            # Check if value contains Quantities
            try:
                flat = list(_flatten(value))
                if flat and all(isinstance(x, Quantity) for x in flat):
                    dims = [x.dim for x in flat]
                    if not all(d == dims[0] for d in dims):
                        raise DimensionMismatchError("Mixed dimensions", *dims[:2])
                    arr.dim = dims[0]
                elif flat and any(isinstance(x, Quantity) for x in flat):
                    raise TypeError("Cannot mix Quantity and non-Quantity values")
                else:
                    arr.dim = DIMENSIONLESS
            except TypeError:
                arr.dim = DIMENSIONLESS

        return arr

    def __array_finalize__(self, orig):
        self.dim = getattr(orig, 'dim', DIMENSIONLESS)

    def __array_ufunc__(self, uf, method, *inputs, **kwargs):
        if method not in ('__call__', 'reduce'):
            return NotImplemented

        uf_method = getattr(uf, method)
        name = uf.__name__

        if name in _UFUNCS_LOGICAL + ['sign', 'ones_like']:
            return uf_method(*[np.asarray(a) for a in inputs], **kwargs)

        elif name in _UFUNCS_PRESERVE:
            return Quantity(
                uf_method(*[np.asarray(a) for a in inputs], **kwargs),
                dim=self.dim
            )

        elif name in _UFUNCS_CHANGE + ['power']:
            if name == 'sqrt':
                dim = self.dim ** 0.5
            elif name == 'square':
                dim = self.dim ** 2
            elif name == 'reciprocal':
                dim = self.dim ** -1
            elif name == 'power':
                _fail_for_dimension_mismatch(inputs[1], error_message="Exponent must be dimensionless")
                dim = get_dimensions(inputs[0]) ** float(np.asarray(inputs[1]))
            elif name in ('divide', 'true_divide', 'floor_divide'):
                dim = get_dimensions(inputs[0]) / get_dimensions(inputs[1])
            elif name == 'multiply':
                if method == 'reduce':
                    dim = self.dim
                else:
                    dim = get_dimensions(inputs[0]) * get_dimensions(inputs[1])
            else:
                return NotImplemented
            return Quantity(
                uf_method(*[np.asarray(a) for a in inputs], **kwargs),
                dim=dim
            )

        elif name in _UFUNCS_MATCHING:
            if method == '__call__':
                _fail_for_dimension_mismatch(
                    inputs[0], inputs[1],
                    error_message=f"Cannot {name} quantities with different dimensions"
                )
            return Quantity(
                uf_method(*[np.asarray(a) for a in inputs], **kwargs),
                dim=self.dim
            )

        elif name in _UFUNCS_COMPARE:
            if method == '__call__':
                _fail_for_dimension_mismatch(
                    inputs[0], inputs[1],
                    error_message=f"Cannot compare quantities with different dimensions"
                )
            return uf_method(*[np.asarray(a) for a in inputs], **kwargs)

        elif name in _UFUNCS_DIMENSIONLESS:
            _fail_for_dimension_mismatch(
                inputs[0],
                error_message=f"{name} expects dimensionless argument"
            )
            return uf_method(np.asarray(inputs[0]), *inputs[1:], **kwargs)

        return NotImplemented

    @staticmethod
    def with_dimensions(value, *args, **kwargs):
        """Create a Quantity with specified dimensions."""
        if args and isinstance(args[0], Dimension):
            dim = args[0]
        else:
            dim = get_or_create_dimension(*args, **kwargs)
        return Quantity(value, dim=dim)

    @property
    def dimensions(self):
        """The physical dimensions of this quantity."""
        return self.dim

    @property
    def is_dimensionless(self):
        """Whether this quantity is dimensionless."""
        return self.dim is DIMENSIONLESS or self.dim.is_dimensionless

    def has_same_dimensions(self, other):
        """Check if this quantity has the same dimensions as another."""
        return have_same_dimensions(self, other)

    def in_unit(self, unit):
        """Express this quantity in the given unit."""
        _fail_for_dimension_mismatch(self, unit, "Cannot convert to incompatible unit")
        return np.asarray(self / unit)

    def _repr_str(self):
        """Build representation string."""
        value_str = str(np.asarray(self))
        if self.dim is DIMENSIONLESS or self.dim.is_dimensionless:
            return value_str
        return f"{value_str} * {self.dim!r}"

    def __repr__(self):
        return self._repr_str()

    def __str__(self):
        return self._repr_str()

    # Override comparison to handle dimension checking
    def __lt__(self, other):
        _fail_for_dimension_mismatch(self, other, "Cannot compare")
        return np.asarray(self) < np.asarray(other)

    def __le__(self, other):
        _fail_for_dimension_mismatch(self, other, "Cannot compare")
        return np.asarray(self) <= np.asarray(other)

    def __gt__(self, other):
        _fail_for_dimension_mismatch(self, other, "Cannot compare")
        return np.asarray(self) > np.asarray(other)

    def __ge__(self, other):
        _fail_for_dimension_mismatch(self, other, "Cannot compare")
        return np.asarray(self) >= np.asarray(other)

    def __eq__(self, other):
        try:
            _fail_for_dimension_mismatch(self, other, "Cannot compare")
            return np.asarray(self) == np.asarray(other)
        except DimensionMismatchError:
            return False

    def __ne__(self, other):
        try:
            _fail_for_dimension_mismatch(self, other, "Cannot compare")
            return np.asarray(self) != np.asarray(other)
        except DimensionMismatchError:
            return True

    def __hash__(self):
        # Quantities are mutable (arrays), so not hashable by default
        raise TypeError("Quantity objects are unhashable")


def _flatten(iterable):
    """Flatten a nested iterable."""
    for item in iterable:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


class Unit(Quantity):
    """
    A named physical unit.

    Units are Quantities with a scale factor and display name.
    """

    __slots__ = ['dim', 'scale', '_name', '_dispname']
    __array_priority__ = 100

    _registered_units = {}

    def __new__(cls, value, dim=None, scale=0, name=None, dispname=None, dtype=None, copy=False):
        if dim is None:
            dim = DIMENSIONLESS
        obj = np.asarray(value, dtype=dtype).view(cls)
        obj.dim = dim
        return obj

    def __init__(self, value, dim=None, scale=0, name=None, dispname=None):
        if dim is None:
            dim = DIMENSIONLESS
        self.dim = dim
        self.scale = scale

        if name is None:
            name = repr(dim) if dim is not DIMENSIONLESS else "1"
        if dispname is None:
            dispname = str(dim) if dim is not DIMENSIONLESS else "1"

        self._name = name
        self._dispname = dispname

        # Register this unit
        Unit._registered_units[(dim, scale)] = self

    @property
    def name(self):
        return self._name

    @property
    def dispname(self):
        return self._dispname

    @property
    def is_dimensionless(self):
        return self.dim is DIMENSIONLESS or self.dim.is_dimensionless

    @staticmethod
    def create(dim, name, dispname, scale=0):
        """Create a new named unit."""
        return Unit(
            10.0 ** scale,
            dim=dim,
            scale=scale,
            name=name,
            dispname=dispname
        )

    @staticmethod
    def create_scaled(baseunit, prefix):
        """Create a scaled version of a unit."""
        scale = SI_PREFIXES[prefix] + baseunit.scale
        name = prefix + baseunit.name
        dispname = prefix + baseunit.dispname
        return Unit(
            10.0 ** scale,
            dim=baseunit.dim,
            scale=scale,
            name=name,
            dispname=dispname
        )

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._dispname

    def __hash__(self):
        return hash((self.dim, self.scale))


def check_units(**units_spec):
    """
    Decorator to check units of function arguments.

    Usage:
        @check_units(x=second, y=volt)
        def my_func(x, y):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, expected_unit in units_spec.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if value is None:
                        continue
                    expected_dim = get_dimensions(expected_unit)
                    actual_dim = get_dimensions(value)
                    if actual_dim is not expected_dim and actual_dim != expected_dim:
                        raise DimensionMismatchError(
                            f"Argument '{param_name}' has wrong dimensions",
                            actual_dim, expected_dim
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator
