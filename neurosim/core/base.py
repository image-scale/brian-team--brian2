"""
Base classes for simulation objects.

Provides tracking, naming, and scheduling infrastructure for all simulator objects.
"""

import re
import uuid
from collections import defaultdict
from weakref import ref

from neurosim.utils.logger import get_logger

__all__ = [
    'Trackable',
    'Nameable',
    'SimObject',
    'SCHEDULE_PHASES',
]

logger = get_logger(__name__)

SCHEDULE_PHASES = ('start', 'groups', 'thresholds', 'synapses', 'resets', 'end')


class InstanceTrackerSet(set):
    """
    A set of weak references to all existing objects of a certain class.

    Automatically removes entries when objects are garbage collected.
    """

    def add(self, value):
        """Add a weak reference to the value."""
        wr = ref(value, self.remove)
        set.add(self, wr)

    def remove(self, value):
        """Remove the weak reference if it exists."""
        try:
            set.remove(self, value)
        except KeyError:
            pass


class InstanceFollower:
    """
    Track all instances of classes derived from Trackable.

    Maintains a dictionary mapping class objects to sets of their instances.
    """

    instance_sets = defaultdict(InstanceTrackerSet)

    def add(self, value):
        """Add an instance to tracking sets for all its classes in MRO."""
        for cls in value.__class__.__mro__:
            self.instance_sets[cls].add(value)

    def get(self, cls):
        """Get the set of instances for a class."""
        return self.instance_sets[cls]

    def clear(self, cls=None):
        """Clear tracked instances, optionally for a specific class only."""
        if cls is None:
            self.instance_sets.clear()
        elif cls in self.instance_sets:
            self.instance_sets[cls].clear()


class Trackable:
    """
    Base class that tracks all instances.

    Classes derived from this will have their instances tracked via weak
    references. The classmethod __instances__() returns the set of instances.
    """

    __instancefollower__ = InstanceFollower()

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__instancefollower__.add(obj)
        return obj

    @classmethod
    def __instances__(cls):
        """Return the set of weak references to all instances of this class."""
        return cls.__instancefollower__.get(cls)


def find_name(name, existing_names=None):
    """
    Find a unique name, generating variants if needed.

    If name ends with '*', tries name without asterisk first, then name_1,
    name_2, etc. until a unique name is found. Otherwise uses the name as-is.

    Parameters
    ----------
    name : str
        The desired name, possibly ending in '*' for auto-increment.
    existing_names : set, optional
        Set of names already taken. If None, uses names of all Nameable instances.

    Returns
    -------
    str
        A unique name.
    """
    if not name.endswith('*'):
        return name

    name = name[:-1]

    if existing_names is None:
        instances = set(Nameable.__instances__())
        existing_names = {obj().name for obj in instances if obj() is not None and hasattr(obj(), 'name')}

    if name not in existing_names:
        return name

    i = 1
    while f"{name}_{i}" in existing_names:
        i += 1
    return f"{name}_{i}"


class Nameable(Trackable):
    """
    Mixin providing unique naming for objects.

    Names can be explicitly provided or auto-generated. If a name ends with '*',
    variants will be tried (name, name_1, name_2, ...) until a unique one is found.

    Parameters
    ----------
    name : str
        The name for this object. May end with '*' for auto-increment behavior.

    Raises
    ------
    TypeError
        If name is not a string.
    ValueError
        If name is not a valid identifier.
    """

    def __init__(self, name):
        if getattr(self, '_name', None) is not None and name is None:
            return

        self._assign_id()

        if not isinstance(name, str):
            raise TypeError(f"'name' must be a string, got {type(name).__name__}")

        if not re.match(r'^[_A-Za-z][_a-zA-Z0-9]*\*?$', name):
            raise ValueError(f"Invalid name '{name}': must be a valid identifier")

        self._name = find_name(name)
        logger.debug(f"Created {self.__class__.__name__} with name '{self._name}'")

    def _assign_id(self):
        """Assign a unique UUID to this object."""
        self._id = uuid.uuid4()

    @property
    def name(self):
        """The unique name for this object."""
        return self._name

    @property
    def id(self):
        """A unique identifier (UUID) for this object."""
        return self._id


class SimObject(Nameable):
    """
    Base class for all simulation objects.

    Provides common functionality including clock association, scheduling,
    and object lifecycle management.

    Parameters
    ----------
    dt : Quantity, optional
        The time step for this object. Cannot be combined with clock.
    clock : Clock, optional
        The clock for this object. If neither dt nor clock is given,
        uses the defaultclock.
    when : str, optional
        Scheduling phase. One of: 'start', 'groups', 'thresholds',
        'synapses', 'resets', 'end'. Defaults to 'start'.
    order : int, optional
        Priority within the same phase. Lower values run first. Defaults to 0.
    name : str, optional
        Name for this object. Defaults to 'simobject*' for auto-naming.
    """

    def __init__(self, dt=None, clock=None, when='start', order=0, name='simobject*'):
        if dt is not None and clock is not None:
            raise ValueError("Cannot specify both dt and clock")

        if not isinstance(when, str):
            raise TypeError("'when' must be a string specifying the scheduling phase")

        if when not in SCHEDULE_PHASES:
            raise ValueError(f"Invalid scheduling phase '{when}'. "
                           f"Must be one of: {', '.join(SCHEDULE_PHASES)}")

        Nameable.__init__(self, name)

        self._clock = clock
        if clock is None:
            from neurosim.core.clocks import Clock, defaultclock
            if dt is not None:
                self._clock = Clock(dt=dt, name=f'{self.name}_clock*')
            else:
                self._clock = defaultclock

        self._when = when
        self._order = order
        self._active = True
        self._network = None
        self._contained_objects = []

        logger.debug(f"SimObject '{self.name}' created with clock={self._clock.name}, "
                    f"when={self._when}, order={self._order}")

    @property
    def clock(self):
        """The Clock determining when this object updates."""
        return self._clock

    @property
    def when(self):
        """The scheduling phase for this object."""
        return self._when

    @when.setter
    def when(self, value):
        if value not in SCHEDULE_PHASES:
            raise ValueError(f"Invalid scheduling phase '{value}'")
        self._when = value

    @property
    def order(self):
        """The priority within the scheduling phase (lower runs first)."""
        return self._order

    @order.setter
    def order(self, value):
        self._order = int(value)

    @property
    def active(self):
        """Whether this object should be updated during simulation."""
        return self._active

    @active.setter
    def active(self, value):
        self._active = bool(value)
        for obj in self._contained_objects:
            obj.active = value

    @property
    def contained_objects(self):
        """List of objects contained within this object."""
        return self._contained_objects

    def before_run(self, namespace=None):
        """Called before the simulation starts. Override in subclasses."""
        pass

    def after_run(self):
        """Called after the simulation ends. Override in subclasses."""
        pass

    def run(self):
        """Execute one time step. Override in subclasses."""
        pass

    def __repr__(self):
        return (f"{self.__class__.__name__}(clock={self._clock.name}, "
                f"when={self._when}, order={self._order}, name={self._name!r})")
