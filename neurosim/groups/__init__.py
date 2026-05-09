"""
Group classes for neural models.

Groups are collections of neurons or other objects with shared state variables.
"""

from .group import Group
from .neurongroup import NeuronGroup

__all__ = ['Group', 'NeuronGroup']
