"""
PoissonGroup - generates random Poisson-distributed spikes.

Provides random input to neural networks with specified firing rates.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.core.clocks import defaultclock
from neurosim.core.variables import Variables, ArrayVariable
from neurosim.units import second, hertz
from neurosim.units.core import DIMENSIONLESS, Quantity
from neurosim.utils.logger import get_logger

__all__ = ['PoissonGroup']

logger = get_logger(__name__)


class PoissonGroup(SimObject):
    """
    A group that generates Poisson-distributed random spikes.

    Parameters
    ----------
    N : int
        Number of neurons in the group.
    rates : Quantity or float or array-like
        Firing rate(s) in Hz. Can be a single value (same for all neurons)
        or an array of rates (one per neuron).
    clock : Clock, optional
        The clock to use. Defaults to defaultclock.
    name : str, optional
        Name of this group.

    Examples
    --------
    >>> from neurosim.groups import PoissonGroup
    >>> from neurosim.units.stdunits import Hz
    >>> # 100 neurons firing at 10 Hz
    >>> G = PoissonGroup(100, rates=10*Hz)
    """

    def __init__(self, N, rates, clock=None, name='poissongroup*'):
        if clock is None:
            clock = defaultclock

        super().__init__(name=name, clock=clock)

        self._N = int(N)
        self.when = 'thresholds'

        self.variables = Variables(owner=self)

        freq_dim = hertz.dim
        self.variables.add_array('rates', N, dimensions=freq_dim, dtype=np.float64)

        if isinstance(rates, Quantity):
            rate_values = np.asarray(rates)
        else:
            rate_values = np.asarray(rates)

        if np.isscalar(rate_values) or rate_values.shape == ():
            rate_values = np.full(N, float(rate_values))

        if len(rate_values) != N:
            raise ValueError(f"rates must be scalar or have length {N}")

        self.variables['rates'].set_value(rate_values)

        self._spikes = np.array([], dtype=np.int32)

        logger.debug(f"Created PoissonGroup '{self.name}' with N={N}")

    @property
    def N(self):
        """Number of neurons in the group."""
        return self._N

    def __len__(self):
        return self._N

    @property
    def spikes(self):
        """Indices of neurons that spiked in the last timestep."""
        return self._spikes

    @property
    def rates(self):
        """Firing rates (with units)."""
        return Quantity(self.variables['rates'].get_value(), dim=hertz.dim)

    @rates.setter
    def rates(self, value):
        """Set firing rates."""
        if isinstance(value, Quantity):
            value = np.asarray(value)
        else:
            value = np.asarray(value)

        if np.isscalar(value) or value.shape == ():
            value = np.full(self._N, float(value))

        self.variables['rates'].set_value(value)

    @property
    def rates_(self):
        """Firing rates (raw, in Hz)."""
        return self.variables['rates'].get_value()

    def run(self):
        """Generate Poisson spikes for current timestep."""
        dt = self.clock.dt_
        rates = self.variables['rates'].get_value()

        probs = rates * dt

        rand = np.random.rand(self._N)
        spiked = rand < probs

        self._spikes = np.nonzero(spiked)[0].astype(np.int32)

    def __repr__(self):
        mean_rate = np.mean(self.variables['rates'].get_value())
        return f"<PoissonGroup '{self.name}' of {self._N} neurons, mean rate={mean_rate:.1f} Hz>"
