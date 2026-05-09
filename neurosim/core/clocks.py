"""
Simulation clocks that manage time stepping.
"""

import numpy as np

from neurosim.units import second, Quantity, get_dimensions
from neurosim.units.stdunits import ms
from neurosim.utils import get_logger

__all__ = ['Clock', 'defaultclock']

logger = get_logger(__name__)


class Clock:
    """
    A clock that tracks simulation time and timestep.

    Parameters
    ----------
    dt : Quantity or float
        The time step of the simulation. If a float, it's assumed to be in seconds.
    name : str, optional
        A name for the clock. If not specified, generates an automatic name.
    """

    _clock_counter = 0
    epsilon_dt = 1e-4  # Relative tolerance for time comparisons

    def __init__(self, dt=None, name=None):
        if dt is None:
            dt = 0.1 * ms

        # Convert dt to float (in seconds)
        if isinstance(dt, Quantity):
            self._dt = float(np.asarray(dt))
        else:
            self._dt = float(dt)

        if self._dt <= 0:
            raise ValueError("dt must be positive")

        if name is None:
            Clock._clock_counter += 1
            name = f"clock_{Clock._clock_counter}"

        self._name = name
        self._timestep = 0
        self._t = 0.0
        self._i_end = None

        logger.debug(f"Created clock '{self._name}' with dt={self._dt * 1000:.4f} ms")

    @property
    def name(self):
        """The name of this clock."""
        return self._name

    @property
    def dt(self):
        """The time step as a Quantity."""
        return self._dt * second

    @dt.setter
    def dt(self, value):
        """Set the time step."""
        if isinstance(value, Quantity):
            new_dt = float(np.asarray(value))
        else:
            new_dt = float(value)

        if new_dt <= 0:
            raise ValueError("dt must be positive")

        # Check if current time can be represented with new dt
        if self._t > 0:
            old_timestep = int(round(self._t / self._dt))
            new_timestep = int(round(self._t / new_dt))
            if abs(old_timestep * self._dt - new_timestep * new_dt) > self.epsilon_dt * new_dt:
                raise ValueError(
                    f"Cannot change dt from {self._dt*1000:.4f} ms to {new_dt*1000:.4f} ms, "
                    f"the current time {self._t*1000:.4f} ms is not a multiple of the new dt."
                )
            self._timestep = new_timestep

        self._dt = new_dt

    @property
    def dt_(self):
        """The time step as a float (in seconds)."""
        return self._dt

    @property
    def t(self):
        """The current simulation time as a Quantity."""
        return self._t * second

    @property
    def t_(self):
        """The current simulation time as a float (in seconds)."""
        return self._t

    @property
    def timestep(self):
        """The current timestep number (integer)."""
        return self._timestep

    def set_interval(self, start, end):
        """
        Set the simulation time interval.

        Parameters
        ----------
        start : Quantity or float
            The start time
        end : Quantity or float
            The end time
        """
        if isinstance(start, Quantity):
            start = float(np.asarray(start))
        if isinstance(end, Quantity):
            end = float(np.asarray(end))

        if end <= start:
            raise ValueError("end time must be greater than start time")

        self._timestep = int(round(start / self._dt))
        self._t = self._timestep * self._dt
        self._i_end = int(round(end / self._dt))

    def advance(self):
        """
        Advance the clock by one time step.

        Raises
        ------
        StopIteration
            If the end time has been reached
        """
        if self._i_end is not None and self._timestep >= self._i_end:
            raise StopIteration("Clock has reached the end time")

        self._timestep += 1
        self._t = self._timestep * self._dt

    def reset(self):
        """Reset the clock to time 0."""
        self._timestep = 0
        self._t = 0.0
        self._i_end = None

    def same_time(self, other):
        """
        Check if this clock is at the same time as another clock.

        Parameters
        ----------
        other : Clock
            The other clock to compare with

        Returns
        -------
        bool
            True if both clocks are at the same time (within epsilon)
        """
        return abs(self._t - other._t) < self.epsilon_dt * self._dt

    def __repr__(self):
        return f"Clock(dt={self._dt*1000:.4f}*ms, name='{self._name}')"


# Global default clock
defaultclock = Clock(dt=0.1 * ms, name='defaultclock')
