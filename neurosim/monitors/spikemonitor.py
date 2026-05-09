"""
Spike monitoring for neural simulations.

Records when neurons in a group fire spikes.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.units import second
from neurosim.units.core import Quantity
from neurosim.utils.logger import get_logger

__all__ = ['SpikeMonitor']

logger = get_logger(__name__)


class SpikeMonitor(SimObject):
    """
    Records spikes from a NeuronGroup.

    Parameters
    ----------
    source : NeuronGroup
        The group to record spikes from.
    record : bool or array-like, optional
        Which neurons to record. True (default) records all neurons.
        An array of indices records only those neurons.
    name : str, optional
        Name of this monitor.

    Attributes
    ----------
    i : array
        Indices of neurons that spiked.
    t : Quantity
        Times when spikes occurred.
    count : array
        Number of spikes per neuron.
    num_spikes : int
        Total number of recorded spikes.

    Examples
    --------
    >>> from neurosim.groups import NeuronGroup
    >>> from neurosim.monitors import SpikeMonitor
    >>> G = NeuronGroup(100, 'dv/dt = 1*mV/ms : volt', threshold='v > 10*mV', reset='v = 0*mV')
    >>> mon = SpikeMonitor(G)
    """

    def __init__(self, source, record=True, name='spikemonitor*'):
        super().__init__(name=name, clock=source.clock)

        self.source = source
        self.when = 'thresholds'
        self.order = 1

        if record is True:
            self._record_indices = None
        elif record is False:
            self._record_indices = np.array([], dtype=np.int32)
        else:
            self._record_indices = np.asarray(record, dtype=np.int32)

        self._spike_indices = []
        self._spike_times = []
        self._count = np.zeros(source.N, dtype=np.int32)

        logger.debug(f"Created SpikeMonitor '{self.name}' for {source.name}")

    def reinit(self):
        """Reset the monitor, clearing all recorded spikes."""
        self._spike_indices = []
        self._spike_times = []
        self._count = np.zeros(self.source.N, dtype=np.int32)

    def run(self):
        """Record spikes from the current timestep."""
        spikes = self.source.spikes

        if len(spikes) == 0:
            return

        if self._record_indices is not None:
            mask = np.isin(spikes, self._record_indices)
            spikes = spikes[mask]

        if len(spikes) == 0:
            return

        t = self.clock.t_

        self._spike_indices.extend(spikes.tolist())
        self._spike_times.extend([t] * len(spikes))
        self._count[spikes] += 1

    @property
    def i(self):
        """Array of indices of neurons that spiked."""
        return np.array(self._spike_indices, dtype=np.int32)

    @property
    def t(self):
        """Array of spike times (with units)."""
        return Quantity(np.array(self._spike_times), dim=second.dim)

    @property
    def t_(self):
        """Array of spike times (raw, in seconds)."""
        return np.array(self._spike_times)

    @property
    def count(self):
        """Number of spikes per neuron."""
        return self._count.copy()

    @property
    def num_spikes(self):
        """Total number of recorded spikes."""
        return len(self._spike_indices)

    def spike_trains(self):
        """
        Return a dictionary mapping neuron indices to their spike times.

        Returns
        -------
        dict
            Dictionary with neuron indices as keys and arrays of spike times as values.
        """
        trains = {}
        for idx, t in zip(self._spike_indices, self._spike_times):
            if idx not in trains:
                trains[idx] = []
            trains[idx].append(t)

        for idx in trains:
            trains[idx] = Quantity(np.array(trains[idx]), dim=second.dim)

        return trains

    def __repr__(self):
        return f"<SpikeMonitor '{self.name}' recording {self.source.name}, {self.num_spikes} spikes>"
