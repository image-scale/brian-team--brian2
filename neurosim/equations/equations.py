"""
Differential equations for neural models.

Provides classes for parsing and representing differential equations,
subexpressions (aliases), and parameters commonly used in neuroscience models.
"""

import re
import keyword
from collections.abc import Mapping

from neurosim.units.core import get_dimensions, DIMENSIONLESS
from neurosim.units.baseunits import second, volt, ampere, siemens, ohm, farad, metre, kilogram, kelvin, mole
from neurosim.utils.logger import get_logger

__all__ = [
    'Equations',
    'SingleEquation',
    'Expression',
    'EquationError',
    'PARAMETER',
    'DIFFERENTIAL_EQUATION',
    'SUBEXPRESSION',
]

logger = get_logger(__name__)

PARAMETER = 'parameter'
DIFFERENTIAL_EQUATION = 'differential equation'
SUBEXPRESSION = 'subexpression'

FLOAT = 'float'
INTEGER = 'integer'
BOOLEAN = 'boolean'

UNIT_NAMESPACE = {
    'second': second, 'metre': metre, 'meter': metre, 'kilogram': kilogram,
    'ampere': ampere, 'kelvin': kelvin, 'mole': mole,
    'volt': volt, 'siemens': siemens, 'ohm': ohm, 'farad': farad,
    'Hz': 1 / second, 'hertz': 1 / second,
    's': second, 'm': metre, 'kg': kilogram, 'A': ampere, 'K': kelvin,
    'V': volt, 'S': siemens, 'F': farad,
}


class EquationError(Exception):
    """Exception raised for errors in equation definitions."""
    pass


def get_identifiers(code):
    """
    Extract all identifiers from a code string.

    Parameters
    ----------
    code : str
        The code string to extract identifiers from.

    Returns
    -------
    set
        Set of identifier names.
    """
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    identifiers = set(re.findall(pattern, code))
    reserved = {'and', 'or', 'not', 'True', 'False', 'None',
                'if', 'else', 'for', 'while', 'in', 'is'}
    return identifiers - reserved


class Expression:
    """
    A code expression with extracted identifiers.

    Parameters
    ----------
    code : str
        The expression code.
    """

    def __init__(self, code):
        self._code = code.strip()
        self._identifiers = get_identifiers(self._code)

    @property
    def code(self):
        """The expression code string."""
        return self._code

    @property
    def identifiers(self):
        """Set of identifiers used in this expression."""
        return self._identifiers

    def __str__(self):
        return self._code

    def __repr__(self):
        return f"Expression({self._code!r})"

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        return self._code == other._code

    def __hash__(self):
        return hash(self._code)


def parse_unit_string(unit_string):
    """
    Parse a unit string and return the corresponding dimensions.

    Parameters
    ----------
    unit_string : str
        The unit specification (e.g., "volt", "siemens/metre**2", "1").

    Returns
    -------
    tuple
        (dimensions, var_type) where dimensions is a Dimension object
        and var_type is FLOAT, INTEGER, or BOOLEAN.

    Raises
    ------
    ValueError
        If the unit string cannot be parsed.
    """
    unit_string = unit_string.strip()

    if unit_string == '1':
        return DIMENSIONLESS, FLOAT

    if unit_string == 'boolean':
        return DIMENSIONLESS, BOOLEAN

    if unit_string == 'integer':
        return DIMENSIONLESS, INTEGER

    try:
        result = eval(unit_string, {"__builtins__": {}}, UNIT_NAMESPACE)
        return get_dimensions(result), FLOAT
    except Exception as e:
        raise ValueError(f"Cannot parse unit string '{unit_string}': {e}")


def check_identifier(identifier):
    """
    Check that an identifier is valid.

    Parameters
    ----------
    identifier : str
        The identifier to check.

    Raises
    ------
    SyntaxError
        If the identifier is not valid.
    """
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise SyntaxError(f"'{identifier}' is not a valid variable name")

    if keyword.iskeyword(identifier):
        raise SyntaxError(f"'{identifier}' is a Python keyword")

    if identifier.startswith('_'):
        raise SyntaxError(f"Variable '{identifier}' starts with underscore (reserved)")

    reserved = {'t', 'dt', 'i', 'N', 'xi'}
    if identifier in reserved or identifier.startswith('xi_'):
        raise SyntaxError(f"'{identifier}' is a reserved name")


class SingleEquation:
    """
    A single equation (parameter, differential equation, or subexpression).

    Parameters
    ----------
    eq_type : str
        The type: PARAMETER, DIFFERENTIAL_EQUATION, or SUBEXPRESSION.
    varname : str
        The variable name defined by this equation.
    dimensions : Dimension
        The physical dimensions of the variable.
    var_type : str, optional
        The variable type: FLOAT, INTEGER, or BOOLEAN.
    expr : Expression, optional
        The right-hand side expression (None for parameters).
    flags : list, optional
        Additional flags like 'constant', 'shared'.
    """

    def __init__(self, eq_type, varname, dimensions, var_type=FLOAT, expr=None, flags=None):
        self.type = eq_type
        self.varname = varname
        self.dim = get_dimensions(dimensions)
        self.var_type = var_type
        self.expr = expr
        self.flags = list(flags) if flags else []

        if self.dim != DIMENSIONLESS:
            if var_type == BOOLEAN:
                raise TypeError("Boolean variables must be dimensionless")
            if var_type == INTEGER:
                raise TypeError("Integer variables must be dimensionless")

        if eq_type == DIFFERENTIAL_EQUATION and var_type != FLOAT:
            raise TypeError("Differential equations can only define float variables")

    @property
    def identifiers(self):
        """Identifiers in the right-hand side expression."""
        if self.expr is not None:
            return self.expr.identifiers
        return set()

    def __str__(self):
        if self.type == DIFFERENTIAL_EQUATION:
            s = f"d{self.varname}/dt"
        else:
            s = self.varname

        if self.expr is not None:
            s += f" = {self.expr}"

        s += f" : {self.dim}"

        if self.flags:
            s += f" ({', '.join(self.flags)})"

        return s

    def __repr__(self):
        return f"<{self.type} {self.varname}: {self.dim}>"

    def __eq__(self, other):
        if not isinstance(other, SingleEquation):
            return NotImplemented
        return (self.type == other.type and
                self.varname == other.varname and
                self.dim == other.dim and
                self.expr == other.expr)

    def __hash__(self):
        return hash((self.type, self.varname, self.dim, self.expr))


DIFF_EQ_PATTERN = re.compile(
    r'^d([a-zA-Z_][a-zA-Z0-9_]*)/dt\s*=\s*(.+?)\s*:\s*(\S+(?:\s*/\s*\S+)?(?:\s*\*\*\s*\d+)?)\s*(?:\(([^)]*)\))?\s*$'
)

SUBEXPR_PATTERN = re.compile(
    r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*:\s*(\S+(?:\s*/\s*\S+)?(?:\s*\*\*\s*\d+)?)\s*(?:\(([^)]*)\))?\s*$'
)

PARAM_PATTERN = re.compile(
    r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\S+(?:\s*/\s*\S+)?(?:\s*\*\*\s*\d+)?)\s*(?:\(([^)]*)\))?\s*$'
)


def parse_equation_line(line):
    """
    Parse a single equation line.

    Parameters
    ----------
    line : str
        The equation line to parse.

    Returns
    -------
    SingleEquation
        The parsed equation.

    Raises
    ------
    EquationError
        If the line cannot be parsed.
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    line = re.sub(r'#.*$', '', line).strip()
    if not line:
        return None

    match = DIFF_EQ_PATTERN.match(line)
    if match:
        varname, expr_str, unit_str, flags_str = match.groups()
        check_identifier(varname)
        dims, var_type = parse_unit_string(unit_str)
        expr = Expression(expr_str)
        flags = [f.strip() for f in flags_str.split(',')] if flags_str else []
        return SingleEquation(DIFFERENTIAL_EQUATION, varname, dims, var_type, expr, flags)

    match = SUBEXPR_PATTERN.match(line)
    if match:
        varname, expr_str, unit_str, flags_str = match.groups()
        check_identifier(varname)
        dims, var_type = parse_unit_string(unit_str)
        expr = Expression(expr_str)
        flags = [f.strip() for f in flags_str.split(',')] if flags_str else []
        return SingleEquation(SUBEXPRESSION, varname, dims, var_type, expr, flags)

    match = PARAM_PATTERN.match(line)
    if match:
        varname, unit_str, flags_str = match.groups()
        check_identifier(varname)
        dims, var_type = parse_unit_string(unit_str)
        flags = [f.strip() for f in flags_str.split(',')] if flags_str else []
        return SingleEquation(PARAMETER, varname, dims, var_type, None, flags)

    raise EquationError(f"Cannot parse equation: '{line}'")


class Equations(Mapping):
    """
    Container for a set of equations.

    Equations can be created from a multi-line string defining differential
    equations, subexpressions, and parameters.

    Parameters
    ----------
    eqs : str or Equations
        The equations as a multi-line string, or another Equations object.

    Examples
    --------
    >>> eqs = Equations('''
    ...     dv/dt = (v_rest - v) / tau : volt
    ...     v_rest : volt
    ...     tau : second
    ... ''')
    >>> 'v' in eqs
    True
    >>> eqs['v'].type
    'differential equation'
    """

    def __init__(self, eqs=''):
        self._equations = {}

        if isinstance(eqs, Equations):
            self._equations = dict(eqs._equations)
        elif isinstance(eqs, str):
            self._parse_string(eqs)
        else:
            raise TypeError(f"Expected string or Equations, got {type(eqs).__name__}")

    def _parse_string(self, eqs_string):
        """Parse equations from a string."""
        for line in eqs_string.split('\n'):
            eq = parse_equation_line(line)
            if eq is not None:
                if eq.varname in self._equations:
                    raise EquationError(f"Duplicate variable definition: '{eq.varname}'")
                self._equations[eq.varname] = eq

    def __getitem__(self, key):
        return self._equations[key]

    def __iter__(self):
        return iter(self._equations)

    def __len__(self):
        return len(self._equations)

    def __contains__(self, key):
        return key in self._equations

    @property
    def diff_eq_names(self):
        """Names of variables defined by differential equations."""
        return [name for name, eq in self._equations.items()
                if eq.type == DIFFERENTIAL_EQUATION]

    @property
    def parameter_names(self):
        """Names of parameter variables."""
        return [name for name, eq in self._equations.items()
                if eq.type == PARAMETER]

    @property
    def subexpr_names(self):
        """Names of subexpression variables."""
        return [name for name, eq in self._equations.items()
                if eq.type == SUBEXPRESSION]

    @property
    def names(self):
        """All variable names."""
        return list(self._equations.keys())

    def get_dimension(self, name):
        """Get the dimensions of a variable."""
        return self._equations[name].dim

    def __add__(self, other):
        """Combine two Equations objects."""
        if not isinstance(other, Equations):
            return NotImplemented

        result = Equations(self)
        for name, eq in other._equations.items():
            if name in result._equations:
                raise EquationError(f"Duplicate variable: '{name}'")
            result._equations[name] = eq
        return result

    def __str__(self):
        return '\n'.join(str(eq) for eq in self._equations.values())

    def __repr__(self):
        return f"<Equations with {len(self)} variables: {', '.join(self.names)}>"

    def __eq__(self, other):
        if not isinstance(other, Equations):
            return NotImplemented
        return self._equations == other._equations

    def __hash__(self):
        return hash(tuple(sorted(self._equations.items(), key=lambda x: x[0])))
