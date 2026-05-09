"""
State monitoring for neural simulations.

Records the values of state variables over time.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.units import second
from neurosim.units.core import DIMENSIONLESS, Quantity
from neurosim.utils.logger import get_logger

__all__ = ['StateMonitor']

logger = get_logger(__name__)


class StateMonitor(SimObject):
    """
    Records state variables from a Group over time.

    Parameters
    ----------
    source : Group
        The group to record from.
    variables : str or list of str
        Which variables to record. Can be a single variable name or list.
        Use True to record all state variables.
    record : bool or array-like, optional
        Which neurons to record. True (default) records all neurons.
        An array of indices records only those neurons.
    dt : Quantity, optional
        Recording timestep. If not specified, records every simulation timestep.
    name : str, optional
        Name of this monitor.

    Attributes
    ----------
    t : Quantity
        Times at which recordings were made.
    variables : list
        Names of recorded variables.

    Examples
    --------
    >>> from neurosim.groups import NeuronGroup
    >>> from neurosim.monitors import StateMonitor
    >>> G = NeuronGroup(10, 'dv/dt = -v/tau : volt\\ntau : second')
    >>> mon = StateMonitor(G, 'v', record=True)
    """

    def __init__(self, source, variables, record=True, dt=None, name='statemonitor*'):
        if dt is not None:
            super().__init__(name=name, dt=dt)
        else:
            super().__init__(name=name, clock=source.clock)

        self.source = source
        self.when = 'end'
        self.order = 0

        if variables is True:
            from neurosim.core.variables import ArrayVariable, Constant
            self._variables = [name for name, var in source.variables.items()
                             if isinstance(var, ArrayVariable) and not name.startswith('_')]
        elif isinstance(variables, str):
            self._variables = [variables]
        else:
            self._variables = list(variables)

        if record is True:
            self._record_indices = np.arange(source.N)
        elif record is False:
            self._record_indices = np.array([], dtype=np.int32)
        else:
            self._record_indices = np.asarray(record, dtype=np.int32)

        self._times = []
        self._data = {var: [] for var in self._variables}

        for var in self._variables:
            if var not in source.variables:
                raise KeyError(f"Variable '{var}' not found in source group")

        logger.debug(f"Created StateMonitor '{self.name}' for {source.name}, "
                    f"recording {self._variables}")

    def reinit(self):
        """Reset the monitor, clearing all recorded data."""
        self._times = []
        self._data = {var: [] for var in self._variables}

    def run(self):
        """Record state variables at current timestep."""
        t = self.clock.t_
        self._times.append(t)

        for var_name in self._variables:
            var = self.source.variables[var_name]
            values = var.get_value()[self._record_indices]
            self._data[var_name].append(values.copy())

    @property
    def t(self):
        """Array of recording times (with units)."""
        return Quantity(np.array(self._times), dim=second.dim)

    @property
    def t_(self):
        """Array of recording times (raw, in seconds)."""
        return np.array(self._times)

    @property
    def variables(self):
        """List of recorded variable names."""
        return list(self._variables)

    @property
    def n_indices(self):
        """Number of recorded neurons."""
        return len(self._record_indices)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        if name in ('source', 'when', 'order', 'clock', 'name', 'id', 'active'):
            raise AttributeError(name)

        use_units = not name.endswith('_')
        var_name = name[:-1] if name.endswith('_') else name

        if not hasattr(self, '_data') or var_name not in self._data:
            raise AttributeError(f"'{var_name}' is not a recorded variable")

        data_list = self._data[var_name]
        if len(data_list) == 0:
            data = np.zeros((len(self._record_indices), 0))
        else:
            data = np.array(data_list).T

        if use_units:
            var = self.source.variables[var_name]
            if var.dim != DIMENSIONLESS:
                return Quantity(data, dim=var.dim)

        return data

    def __repr__(self):
        n_times = len(self._times)
        return (f"<StateMonitor '{self.name}' recording {self._variables} "
                f"from {self.source.name}, {n_times} timesteps>")
