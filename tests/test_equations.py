"""
Tests for the equation parsing system.
"""

import pytest

from neurosim.equations import (
    Equations,
    SingleEquation,
    Expression,
    EquationError,
    PARAMETER,
    DIFFERENTIAL_EQUATION,
    SUBEXPRESSION,
)
from neurosim.units.core import DIMENSIONLESS, get_dimensions
from neurosim.units.baseunits import second, volt, ampere, siemens


class TestExpression:
    """Tests for the Expression class."""

    def test_expression_code(self):
        """Expression stores the code."""
        expr = Expression("-v / tau")
        assert expr.code == "-v / tau"

    def test_expression_identifiers(self):
        """Expression extracts identifiers."""
        expr = Expression("-v / tau + I_ext")
        assert 'v' in expr.identifiers
        assert 'tau' in expr.identifiers
        assert 'I_ext' in expr.identifiers

    def test_expression_strips_whitespace(self):
        """Expression strips leading/trailing whitespace."""
        expr = Expression("  x + y  ")
        assert expr.code == "x + y"

    def test_expression_str(self):
        """Expression converts to string."""
        expr = Expression("a * b")
        assert str(expr) == "a * b"

    def test_expression_repr(self):
        """Expression has repr."""
        expr = Expression("a * b")
        assert "a * b" in repr(expr)

    def test_expression_equality(self):
        """Expressions with same code are equal."""
        expr1 = Expression("x + y")
        expr2 = Expression("x + y")
        assert expr1 == expr2

    def test_expression_hash(self):
        """Expressions are hashable."""
        expr = Expression("x + y")
        d = {expr: 1}
        assert d[expr] == 1


class TestSingleEquation:
    """Tests for the SingleEquation class."""

    def test_differential_equation(self):
        """Create a differential equation."""
        expr = Expression("-v / tau")
        eq = SingleEquation(DIFFERENTIAL_EQUATION, 'v', volt.dim, expr=expr)
        assert eq.type == DIFFERENTIAL_EQUATION
        assert eq.varname == 'v'
        assert eq.dim == volt.dim
        assert eq.expr == expr

    def test_parameter(self):
        """Create a parameter."""
        eq = SingleEquation(PARAMETER, 'tau', second.dim)
        assert eq.type == PARAMETER
        assert eq.varname == 'tau'
        assert eq.dim == second.dim
        assert eq.expr is None

    def test_subexpression(self):
        """Create a subexpression."""
        expr = Expression("v + v_rest")
        eq = SingleEquation(SUBEXPRESSION, 'v_total', volt.dim, expr=expr)
        assert eq.type == SUBEXPRESSION
        assert eq.varname == 'v_total'

    def test_identifiers(self):
        """SingleEquation provides identifiers from expression."""
        expr = Expression("a + b * c")
        eq = SingleEquation(DIFFERENTIAL_EQUATION, 'x', DIMENSIONLESS, expr=expr)
        assert 'a' in eq.identifiers
        assert 'b' in eq.identifiers
        assert 'c' in eq.identifiers

    def test_parameter_no_identifiers(self):
        """Parameters have no identifiers."""
        eq = SingleEquation(PARAMETER, 'x', DIMENSIONLESS)
        assert len(eq.identifiers) == 0

    def test_flags(self):
        """SingleEquation stores flags."""
        eq = SingleEquation(PARAMETER, 'x', DIMENSIONLESS, flags=['constant', 'shared'])
        assert 'constant' in eq.flags
        assert 'shared' in eq.flags

    def test_boolean_must_be_dimensionless(self):
        """Boolean variables must be dimensionless."""
        from neurosim.equations.equations import BOOLEAN
        with pytest.raises(TypeError):
            SingleEquation(PARAMETER, 'x', volt.dim, var_type=BOOLEAN)

    def test_integer_must_be_dimensionless(self):
        """Integer variables must be dimensionless."""
        from neurosim.equations.equations import INTEGER
        with pytest.raises(TypeError):
            SingleEquation(PARAMETER, 'x', volt.dim, var_type=INTEGER)

    def test_str(self):
        """SingleEquation has string representation."""
        expr = Expression("-v / tau")
        eq = SingleEquation(DIFFERENTIAL_EQUATION, 'v', volt.dim, expr=expr)
        s = str(eq)
        assert 'dv/dt' in s
        assert '-v / tau' in s


class TestEquationsParsing:
    """Tests for parsing equation strings."""

    def test_parse_differential_equation(self):
        """Parse a differential equation."""
        eqs = Equations("dv/dt = -v / tau : volt")
        assert 'v' in eqs
        assert eqs['v'].type == DIFFERENTIAL_EQUATION

    def test_parse_parameter(self):
        """Parse a parameter."""
        eqs = Equations("tau : second")
        assert 'tau' in eqs
        assert eqs['tau'].type == PARAMETER
        assert eqs['tau'].dim == second.dim

    def test_parse_subexpression(self):
        """Parse a subexpression."""
        eqs = Equations("I_total = I_syn + I_ext : ampere")
        assert 'I_total' in eqs
        assert eqs['I_total'].type == SUBEXPRESSION

    def test_parse_multiline(self):
        """Parse multiple equations."""
        eqs = Equations('''
            dv/dt = (v_rest - v) / tau : volt
            v_rest : volt
            tau : second
        ''')
        assert len(eqs) == 3
        assert 'v' in eqs
        assert 'v_rest' in eqs
        assert 'tau' in eqs

    def test_parse_with_comments(self):
        """Comments are ignored."""
        eqs = Equations('''
            # This is a comment
            dv/dt = -v / tau : volt  # inline comment
            tau : second
        ''')
        assert len(eqs) == 2

    def test_parse_dimensionless(self):
        """Parse dimensionless variable."""
        eqs = Equations("x : 1")
        assert eqs['x'].dim == DIMENSIONLESS

    def test_parse_with_flags(self):
        """Parse equation with flags."""
        eqs = Equations("tau : second (constant)")
        assert 'constant' in eqs['tau'].flags

    def test_parse_multiple_flags(self):
        """Parse equation with multiple flags."""
        eqs = Equations("x : 1 (constant, shared)")
        assert 'constant' in eqs['x'].flags
        assert 'shared' in eqs['x'].flags

    def test_parse_complex_expression(self):
        """Parse equation with complex expression."""
        eqs = Equations("dv/dt = (v_rest - v + R * I) / tau : volt")
        assert 'v' in eqs
        identifiers = eqs['v'].identifiers
        assert 'v_rest' in identifiers
        assert 'v' in identifiers
        assert 'R' in identifiers
        assert 'I' in identifiers
        assert 'tau' in identifiers

    def test_parse_empty_string(self):
        """Empty string creates empty Equations."""
        eqs = Equations("")
        assert len(eqs) == 0

    def test_duplicate_variable_raises(self):
        """Duplicate variable definitions raise error."""
        with pytest.raises(EquationError):
            Equations('''
                x : volt
                x : second
            ''')

    def test_invalid_syntax_raises(self):
        """Invalid syntax raises EquationError."""
        with pytest.raises(EquationError):
            Equations("this is not valid")

    def test_reserved_name_raises(self):
        """Reserved names raise error."""
        with pytest.raises((SyntaxError, EquationError)):
            Equations("t : second")

    def test_keyword_name_raises(self):
        """Python keywords raise error."""
        with pytest.raises((SyntaxError, EquationError)):
            Equations("for : second")


class TestEquationsAccess:
    """Tests for Equations container access."""

    def test_getitem(self):
        """Can access equations by name."""
        eqs = Equations("v : volt")
        eq = eqs['v']
        assert eq.varname == 'v'

    def test_contains(self):
        """Can check if variable exists."""
        eqs = Equations("v : volt")
        assert 'v' in eqs
        assert 'x' not in eqs

    def test_len(self):
        """len() returns number of equations."""
        eqs = Equations('''
            v : volt
            tau : second
        ''')
        assert len(eqs) == 2

    def test_iter(self):
        """Can iterate over variable names."""
        eqs = Equations('''
            a : volt
            b : second
        ''')
        names = list(eqs)
        assert 'a' in names
        assert 'b' in names

    def test_diff_eq_names(self):
        """Get names of differential equations."""
        eqs = Equations('''
            dv/dt = -v / tau : volt
            tau : second
        ''')
        assert eqs.diff_eq_names == ['v']

    def test_parameter_names(self):
        """Get names of parameters."""
        eqs = Equations('''
            dv/dt = -v / tau : volt
            tau : second
            v_rest : volt
        ''')
        assert 'tau' in eqs.parameter_names
        assert 'v_rest' in eqs.parameter_names
        assert 'v' not in eqs.parameter_names

    def test_subexpr_names(self):
        """Get names of subexpressions."""
        eqs = Equations('''
            v_total = v + v_rest : volt
            v : volt
            v_rest : volt
        ''')
        assert eqs.subexpr_names == ['v_total']

    def test_names(self):
        """Get all variable names."""
        eqs = Equations('''
            dv/dt = -v / tau : volt
            tau : second
        ''')
        assert set(eqs.names) == {'v', 'tau'}

    def test_get_dimension(self):
        """Get dimension of a variable."""
        eqs = Equations("v : volt")
        assert eqs.get_dimension('v') == volt.dim


class TestEquationsOperations:
    """Tests for Equations operations."""

    def test_add_equations(self):
        """Can combine Equations with +."""
        eqs1 = Equations("v : volt")
        eqs2 = Equations("tau : second")
        combined = eqs1 + eqs2
        assert 'v' in combined
        assert 'tau' in combined

    def test_add_duplicate_raises(self):
        """Adding equations with duplicate variables raises error."""
        eqs1 = Equations("v : volt")
        eqs2 = Equations("v : ampere")
        with pytest.raises(EquationError):
            eqs1 + eqs2

    def test_str(self):
        """Equations has string representation."""
        eqs = Equations('''
            dv/dt = -v / tau : volt
            tau : second
        ''')
        s = str(eqs)
        assert 'dv/dt' in s
        assert 'tau' in s

    def test_repr(self):
        """Equations has repr."""
        eqs = Equations("v : volt")
        r = repr(eqs)
        assert 'Equations' in r
        assert 'v' in r

    def test_copy_constructor(self):
        """Can create Equations from another Equations."""
        eqs1 = Equations("v : volt")
        eqs2 = Equations(eqs1)
        assert 'v' in eqs2

    def test_equality(self):
        """Equations with same content are equal."""
        eqs1 = Equations("v : volt")
        eqs2 = Equations("v : volt")
        assert eqs1 == eqs2


class TestUnits:
    """Tests for unit parsing in equations."""

    def test_volt_unit(self):
        """Parse volt unit."""
        eqs = Equations("v : volt")
        assert eqs['v'].dim == volt.dim

    def test_second_unit(self):
        """Parse second unit."""
        eqs = Equations("tau : second")
        assert eqs['tau'].dim == second.dim

    def test_ampere_unit(self):
        """Parse ampere unit."""
        eqs = Equations("I : ampere")
        assert eqs['I'].dim == ampere.dim

    def test_siemens_unit(self):
        """Parse siemens unit."""
        eqs = Equations("g : siemens")
        assert eqs['g'].dim == siemens.dim
