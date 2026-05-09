"""
SpikeGeneratorGroup - produces spikes at specified times.

Provides input to neural networks by generating spikes at predetermined times.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.core.clocks import defaultclock
from neurosim.core.variables import Variables
from neurosim.units import second
from neurosim.units.core import Quantity
from neurosim.utils.logger import get_logger

__all__ = ['SpikeGeneratorGroup']

logger = get_logger(__name__)


class SpikeGeneratorGroup(SimObject):
    """
    A group that produces spikes at specified times.

    Parameters
    ----------
    N : int
        Number of neurons in the group.
    indices : array-like
        Indices of neurons that spike.
    times : array-like or Quantity
        Times at which spikes occur.
    period : Quantity, optional
        If specified, spike pattern repeats with this period.
    clock : Clock, optional
        The clock to use. Defaults to defaultclock.
    name : str, optional
        Name of this group.

    Examples
    --------
    >>> from neurosim.groups import SpikeGeneratorGroup
    >>> from neurosim.units.stdunits import ms
    >>> # Neuron 0 spikes at 1ms and 3ms, neuron 1 at 2ms
    >>> indices = [0, 1, 0]
    >>> times = [1, 2, 3] * ms
    >>> G = SpikeGeneratorGroup(2, indices, times)
    """

    def __init__(self, N, indices, times, period=None, clock=None, name='spikegeneratorgroup*'):
        if clock is None:
            clock = defaultclock

        super().__init__(name=name, clock=clock)

        self._N = int(N)
        self.when = 'thresholds'

        self.variables = Variables(owner=self)

        indices = np.asarray(indices, dtype=np.int32)

        if isinstance(times, Quantity):
            times = np.asarray(times)
        elif hasattr(times, '__iter__'):
            times_list = list(times)
            if len(times_list) > 0 and isinstance(times_list[0], Quantity):
                times = np.array([float(np.asarray(t)) for t in times_list])
            else:
                times = np.asarray(times_list, dtype=np.float64)
        else:
            times = np.asarray(times, dtype=np.float64)

        if len(indices) != len(times):
            raise ValueError("indices and times must have the same length")

        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError(f"Neuron indices must be between 0 and {N-1}")

        sorted_order = np.argsort(times)
        self._spike_indices = indices[sorted_order]
        self._spike_times = times[sorted_order]

        self._period = None
        if period is not None:
            if isinstance(period, Quantity):
                self._period = float(np.asarray(period))
            else:
                self._period = float(period)

        self._next_spike_idx = 0
        self._spikes = np.array([], dtype=np.int32)
        self._period_offset = 0.0

        logger.debug(f"Created SpikeGeneratorGroup '{self.name}' with N={N}, "
                    f"{len(self._spike_times)} spikes")

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

    def before_run(self, namespace=None):
        """Reset spike index before run."""
        self._next_spike_idx = 0
        self._period_offset = 0.0

    def run(self):
        """Check for spikes at the current time."""
        t = self.clock.t_
        dt = self.clock.dt_

        epsilon = dt * 0.001

        spike_list = []

        while self._next_spike_idx < len(self._spike_times):
            next_time = self._spike_times[self._next_spike_idx] + self._period_offset

            if next_time <= t + epsilon:
                if next_time >= t - epsilon:
                    spike_list.append(self._spike_indices[self._next_spike_idx])
                self._next_spike_idx += 1
            else:
                break

        if self._period is not None and self._next_spike_idx >= len(self._spike_times):
            self._next_spike_idx = 0
            self._period_offset += self._period

            while self._next_spike_idx < len(self._spike_times):
                next_time = self._spike_times[self._next_spike_idx] + self._period_offset
                if next_time <= t + epsilon:
                    if next_time >= t - epsilon:
                        spike_list.append(self._spike_indices[self._next_spike_idx])
                    self._next_spike_idx += 1
                else:
                    break

        self._spikes = np.array(spike_list, dtype=np.int32)

    def __repr__(self):
        return f"<SpikeGeneratorGroup '{self.name}' of {self._N} neurons, {len(self._spike_times)} spikes>"
