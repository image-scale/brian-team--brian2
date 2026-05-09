"""
Group classes for neural models.

Groups are collections of neurons or other objects with shared state variables.
"""

from .group import Group
from .neurongroup import NeuronGroup
from .spikegeneratorgroup import SpikeGeneratorGroup

__all__ = ['Group', 'NeuronGroup', 'SpikeGeneratorGroup']
