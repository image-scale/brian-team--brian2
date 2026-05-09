"""
TimedArray - time-varying input signals.

Provides time-varying values that can be used in neuron group equations.
"""

import numpy as np

from neurosim.units import second
from neurosim.units.core import DIMENSIONLESS, Quantity, get_dimensions
from neurosim.utils.logger import get_logger

__all__ = ['TimedArray']

logger = get_logger(__name__)


class TimedArray:
    """
    A time-varying array that can be used as input in equations.

    Parameters
    ----------
    values : array-like or Quantity
        The values. Can be 1D (same value for all neurons at each time)
        or 2D (different values per neuron, shape: (n_times, n_neurons)).
    dt : Quantity or float
        Time step between values.
    name : str, optional
        Name for this TimedArray.

    Examples
    --------
    >>> from neurosim.input import TimedArray
    >>> from neurosim.units.stdunits import ms, mV
    >>> import numpy as np
    >>> # Values that change every 10ms
    >>> values = np.array([0, 1, 2, 3]) * mV
    >>> arr = TimedArray(values, dt=10*ms)
    >>> arr(15*ms)  # Returns 1 mV (second value)
    """

    def __init__(self, values, dt, name=None):
        if isinstance(values, Quantity):
            self._values = np.asarray(values)
            self._dim = values.dim
        else:
            self._values = np.asarray(values)
            self._dim = DIMENSIONLESS

        if isinstance(dt, Quantity):
            self._dt = float(np.asarray(dt))
        else:
            self._dt = float(dt)

        if self._dt <= 0:
            raise ValueError("dt must be positive")

        self._ndim = self._values.ndim
        if self._ndim == 1:
            self._n_values = len(self._values)
            self._n_neurons = None
        elif self._ndim == 2:
            self._n_values, self._n_neurons = self._values.shape
        else:
            raise ValueError("values must be 1D or 2D")

        self._name = name or 'timedarray'

        logger.debug(f"Created TimedArray '{self._name}' with {self._n_values} time points, dt={self._dt*1000:.2f}ms")

    @property
    def name(self):
        """Name of this TimedArray."""
        return self._name

    @property
    def dt(self):
        """Time step between values (with units)."""
        return self._dt * second

    @property
    def dt_(self):
        """Time step between values (in seconds)."""
        return self._dt

    @property
    def values(self):
        """The stored values (with units if applicable)."""
        if self._dim == DIMENSIONLESS:
            return self._values
        return Quantity(self._values, dim=self._dim)

    def __call__(self, t):
        """
        Get the value(s) at time t.

        Parameters
        ----------
        t : Quantity or float or array
            Time(s) at which to get values.

        Returns
        -------
        value : Quantity or array
            The value(s) at the given time(s).
        """
        if isinstance(t, Quantity):
            t = np.asarray(t)
        else:
            t = np.asarray(t)

        idx = np.asarray(t / self._dt, dtype=np.int64)

        idx = np.clip(idx, 0, self._n_values - 1)

        if self._ndim == 1:
            result = self._values[idx]
        else:
            if np.isscalar(idx):
                result = self._values[idx, :]
            else:
                result = self._values[idx]

        if self._dim != DIMENSIONLESS:
            return Quantity(result, dim=self._dim)
        return result

    def __repr__(self):
        shape_str = f"({self._n_values},)" if self._ndim == 1 else f"({self._n_values}, {self._n_neurons})"
        return f"<TimedArray '{self._name}' shape={shape_str}, dt={self._dt*1000:.2f}ms>"
