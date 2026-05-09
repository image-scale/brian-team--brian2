"""
NeuronGroup - groups of neurons with dynamics.

Provides the main NeuronGroup class for simulating populations of neurons
with differential equations, thresholds, resets, and refractory periods.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.core.clocks import defaultclock
from neurosim.core.variables import Variables, Constant, ArrayVariable
from neurosim.equations import Equations, PARAMETER, DIFFERENTIAL_EQUATION, SUBEXPRESSION
from neurosim.units.core import DIMENSIONLESS, Quantity, get_dimensions
from neurosim.units import second
from neurosim.utils.logger import get_logger
from .group import Group

__all__ = ['NeuronGroup']

logger = get_logger(__name__)


class Thresholder(SimObject):
    """
    Handles threshold checking for a NeuronGroup.

    Scheduled in the 'thresholds' phase.
    """

    def __init__(self, group, threshold):
        super().__init__(name=f'{group.name}_thresholder', clock=group.clock)
        self.group = group
        self.threshold = threshold
        self.when = 'thresholds'

    def run(self):
        """Check threshold and record spikes."""
        if self.threshold is None:
            return

        namespace = self.group._get_namespace()
        condition = eval(self.threshold, {"__builtins__": {}}, namespace)

        if isinstance(condition, (bool, np.bool_)):
            condition = np.full(self.group.N, condition)

        refractory_until = self.group.variables['_refractory_until'].get_value()
        t = self.group.clock.t_
        not_refractory = t >= refractory_until

        spiked = np.asarray(condition) & not_refractory
        spike_indices = np.nonzero(spiked)[0]

        self.group._spikes = spike_indices

        if len(spike_indices) > 0:
            last_spike = self.group.variables['last_spike'].get_value()
            last_spike[spike_indices] = t

            refractory = self.group._refractory_time
            if refractory is not None:
                if isinstance(refractory, Quantity):
                    refractory = float(np.asarray(refractory))
                refractory_until[spike_indices] = t + refractory


class Resetter(SimObject):
    """
    Handles reset after spikes for a NeuronGroup.

    Scheduled in the 'resets' phase.
    """

    def __init__(self, group, reset):
        super().__init__(name=f'{group.name}_resetter', clock=group.clock)
        self.group = group
        self.reset = reset
        self.when = 'resets'

    def run(self):
        """Apply reset to spiked neurons."""
        if self.reset is None or len(self.group._spikes) == 0:
            return

        spike_indices = self.group._spikes

        for stmt in self.reset.split('\n'):
            stmt = stmt.strip()
            if not stmt or stmt.startswith('#'):
                continue

            if '=' not in stmt:
                continue

            var_name, expr = stmt.split('=', 1)
            var_name = var_name.strip()
            expr = expr.strip()

            if var_name not in self.group.variables:
                continue

            namespace = self.group._get_namespace(indices=spike_indices)
            value = eval(expr, {"__builtins__": {}}, namespace)

            if isinstance(value, Quantity):
                value = np.asarray(value)

            var_data = self.group.variables[var_name].get_value()
            var_data[spike_indices] = value


class StateUpdater(SimObject):
    """
    Handles state variable updates for a NeuronGroup.

    Integrates differential equations using the specified method.
    Scheduled in the 'groups' phase.
    """

    def __init__(self, group, method='euler'):
        super().__init__(name=f'{group.name}_stateupdater', clock=group.clock)
        self.group = group
        self.method = method
        self.when = 'groups'

    def run(self):
        """Update state variables by integrating differential equations."""
        if self.method == 'euler':
            self._euler_step()
        elif self.method == 'exact':
            self._euler_step()
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

    def _euler_step(self):
        """Euler integration step."""
        dt = self.group.clock.dt_
        namespace = self.group._get_namespace()

        for name in self.group.equations.diff_eq_names:
            eq = self.group.equations[name]

            deriv = eval(eq.expr.code, {"__builtins__": {}}, namespace)
            if isinstance(deriv, Quantity):
                deriv = np.asarray(deriv)

            var = self.group.variables[name]
            current = var.get_value()
            var.set_value(current + deriv * dt)


class NeuronGroup(Group):
    """
    A group of neurons with differential equations and spiking.

    Parameters
    ----------
    N : int
        Number of neurons.
    model : str or Equations
        Equations defining the neuron model.
    threshold : str, optional
        Condition for spike generation (e.g., "v > -50*mV").
    reset : str, optional
        Statement(s) to execute after a spike (e.g., "v = -70*mV").
    refractory : Quantity or float, optional
        Duration of refractory period after each spike.
    method : str, optional
        Integration method ('euler' or 'exact'). Default is 'euler'.
    clock : Clock, optional
        The clock to use. Defaults to defaultclock.
    namespace : dict, optional
        Additional namespace for equations.
    dtype : dtype, optional
        Default dtype for state variables.
    name : str, optional
        Name of the group.

    Examples
    --------
    >>> from neurosim.units import mV, ms
    >>> neurons = NeuronGroup(100, '''
    ...     dv/dt = (v_rest - v) / tau : volt
    ...     v_rest : volt
    ...     tau : second
    ... ''', threshold='v > -50*mV', reset='v = -70*mV')
    """

    def __init__(self, N, model, threshold=None, reset=None, refractory=None,
                 method='euler', clock=None, namespace=None, dtype=None, name='neurongroup*'):
        if clock is None:
            clock = defaultclock

        super().__init__(N, equations=model, namespace=namespace, dtype=dtype,
                        name=name, clock=clock)

        self._threshold = threshold
        self._reset = reset
        self._refractory_time = refractory
        self._method = method
        self._spikes = np.array([], dtype=np.int32)

        self.variables.add_array('last_spike', N, dimensions=second.dim,
                                dtype=np.float64)
        self.variables['last_spike'].set_value(np.full(N, -np.inf))

        self.variables.add_array('_refractory_until', N, dimensions=DIMENSIONLESS,
                                dtype=np.float64)
        self.variables['_refractory_until'].set_value(np.full(N, -np.inf))

        self._thresholder = Thresholder(self, threshold)
        self._resetter = Resetter(self, reset)
        self._state_updater = StateUpdater(self, method)

        self._contained_objects = [self._state_updater, self._thresholder, self._resetter]

        logger.debug(f"Created NeuronGroup '{self.name}' with N={N}, method={method}")

    @property
    def spikes(self):
        """Indices of neurons that spiked in the last timestep."""
        return self._spikes

    def _get_namespace(self, indices=None):
        """
        Build namespace for evaluating expressions.

        Parameters
        ----------
        indices : array-like, optional
            If provided, restrict to these neuron indices.

        Returns
        -------
        dict
            Namespace containing variable values and constants.
        """
        ns = dict(self._namespace) if self._namespace else {}

        ns['t'] = self.clock.t_
        ns['dt'] = self.clock.dt_

        for name, var in self.variables.items():
            if isinstance(var, Constant):
                ns[name] = var.get_value()
            elif isinstance(var, ArrayVariable):
                data = var.get_value()
                if indices is not None:
                    data = data[indices]
                # Include units for proper dimension checking
                if var.dim != DIMENSIONLESS:
                    data = Quantity(data, dim=var.dim)
                ns[name] = data

        ns['exp'] = np.exp
        ns['log'] = np.log
        ns['sin'] = np.sin
        ns['cos'] = np.cos
        ns['tan'] = np.tan
        ns['sqrt'] = np.sqrt
        ns['abs'] = np.abs
        ns['clip'] = np.clip
        ns['floor'] = np.floor
        ns['ceil'] = np.ceil
        ns['int'] = lambda x: np.asarray(x, dtype=np.int32)
        ns['rand'] = lambda: np.random.rand(self.N) if indices is None else np.random.rand(len(indices))
        ns['randn'] = lambda: np.random.randn(self.N) if indices is None else np.random.randn(len(indices))

        from neurosim import units
        for attr in dir(units):
            obj = getattr(units, attr)
            if isinstance(obj, (Quantity, float, int)) and not attr.startswith('_'):
                ns[attr] = obj

        return ns

    def __repr__(self):
        return f"<NeuronGroup '{self.name}' of {self._N} neurons>"
