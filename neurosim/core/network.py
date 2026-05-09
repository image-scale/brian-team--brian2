"""
Network module for managing simulation runs.

The Network collects simulation objects and runs them in the correct order.
"""

from collections.abc import Mapping

from neurosim.core.base import Nameable, SimObject, SCHEDULE_PHASES
from neurosim.core.clocks import defaultclock
from neurosim.units.core import Quantity
from neurosim.units.baseunits import second
from neurosim.utils.logger import get_logger

__all__ = ['Network']

logger = get_logger(__name__)


def _get_all_objects(objs):
    """
    Recursively collect all objects including contained objects.

    Parameters
    ----------
    objs : iterable
        Collection of SimObjects.

    Returns
    -------
    set
        All objects and their contained objects.
    """
    all_objects = set()
    for obj in objs:
        all_objects.add(obj)
        if hasattr(obj, 'contained_objects'):
            all_objects |= _get_all_objects(obj.contained_objects)
    return all_objects


class Network(Nameable):
    """
    The main simulation controller.

    Network manages running a simulation by collecting SimObjects and calling
    their run methods in the correct order determined by scheduling phase
    and order.

    Parameters
    ----------
    *objs : SimObject or iterable
        Objects to add to the network immediately.
    name : str, optional
        Name for this network. Defaults to 'network*' for auto-naming.

    Examples
    --------
    >>> from neurosim.core import Network, SimObject
    >>> obj1 = SimObject(name='obj1')
    >>> obj2 = SimObject(name='obj2')
    >>> net = Network(obj1, obj2)
    >>> net.run(10 * ms)
    """

    def __init__(self, *objs, name='network*'):
        Nameable.__init__(self, name)

        self.objects = set()
        self._t = 0.0
        self._stopped = False
        self._schedule = None

        for obj in objs:
            self.add(obj)

        logger.debug(f"Created Network '{self.name}'")

    @property
    def t(self):
        """Current simulation time as a Quantity."""
        return Quantity(self._t, dim=second.dim)

    @property
    def t_(self):
        """Current simulation time as a float (seconds)."""
        return self._t

    @property
    def schedule(self):
        """The scheduling order for simulation phases."""
        if self._schedule is None:
            return list(SCHEDULE_PHASES)
        return list(self._schedule)

    @schedule.setter
    def schedule(self, value):
        if value is None:
            self._schedule = None
        else:
            if not all(isinstance(s, str) for s in value):
                raise TypeError("Schedule must be a sequence of strings")
            self._schedule = list(value)

    def add(self, *objs):
        """
        Add objects to the network.

        Parameters
        ----------
        *objs : SimObject or iterable
            Objects to add. Can be SimObjects, lists, dicts (values are added).
        """
        for obj in objs:
            if isinstance(obj, SimObject):
                if obj._network is not None and obj._network != self.id:
                    raise RuntimeError(
                        f"'{obj.name}' has already been added to another network"
                    )
                self.objects.add(obj)
                logger.debug(f"Added '{obj.name}' to network '{self.name}'")
            elif isinstance(obj, Mapping):
                self.add(*obj.values())
            else:
                try:
                    for o in obj:
                        if o is obj:
                            raise TypeError()
                        self.add(o)
                except TypeError:
                    raise TypeError(
                        f"Can only add SimObject or containers of SimObjects, "
                        f"got {type(obj).__name__}"
                    )

    def remove(self, *objs):
        """
        Remove objects from the network.

        Parameters
        ----------
        *objs : SimObject or iterable
            Objects to remove.
        """
        for obj in objs:
            if isinstance(obj, SimObject):
                self.objects.discard(obj)
                logger.debug(f"Removed '{obj.name}' from network '{self.name}'")
            else:
                try:
                    for o in obj:
                        self.remove(o)
                except TypeError:
                    raise TypeError(
                        f"Can only remove SimObject or containers of SimObjects"
                    )

    @property
    def sorted_objects(self):
        """
        Objects sorted by scheduling order.

        Objects are sorted first by their scheduling phase (when), then by
        order (lower first), then by name for determinism.
        """
        all_objects = _get_all_objects(self.objects)
        schedule = self.schedule

        when_to_int = {when: i for i, when in enumerate(schedule)}
        when_to_int.update(
            (f'before_{when}', i - 0.5) for i, when in enumerate(schedule)
        )
        when_to_int.update(
            (f'after_{when}', i + 0.5) for i, when in enumerate(schedule)
        )

        def sort_key(obj):
            when_val = when_to_int.get(obj.when, len(schedule))
            return (when_val, obj.order, obj.name)

        return sorted(all_objects, key=sort_key)

    def before_run(self, namespace=None):
        """
        Prepare the network for a run.

        Calls before_run on all active objects.
        """
        for obj in self.sorted_objects:
            if obj.active:
                obj.before_run(namespace)
                if obj._network is None:
                    obj._network = self.id

    def after_run(self):
        """
        Clean up after a run.

        Calls after_run on all active objects.
        """
        for obj in self.sorted_objects:
            if obj.active:
                obj.after_run()

    def run(self, duration, namespace=None):
        """
        Run the simulation for the specified duration.

        Parameters
        ----------
        duration : Quantity or float
            Duration to simulate. If float, interpreted as seconds.
        namespace : dict, optional
            Namespace for variable resolution.
        """
        if hasattr(duration, 'dim'):
            duration_seconds = float(duration / second)
        else:
            duration_seconds = float(duration)

        if duration_seconds < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")

        all_objects = self.sorted_objects
        if not all_objects:
            logger.debug(f"Network '{self.name}' has no objects, nothing to run")
            return

        clocks = {obj.clock for obj in all_objects}

        t_start = self._t
        t_end = self._t + duration_seconds

        for clock in clocks:
            clock.set_interval(t_start * second, t_end * second)

        self._stopped = False
        self.before_run(namespace)

        logger.debug(f"Running network '{self.name}' from {t_start}s to {t_end}s")

        active_objects = [obj for obj in all_objects if obj.active]

        if len(clocks) == 1:
            clock = list(clocks)[0]
            while not self._stopped and clock.running:
                self._t = clock.t_
                for obj in active_objects:
                    obj.run()
                clock.advance()
        else:
            all_running = any(c.running for c in clocks)
            while not self._stopped and all_running:
                min_clock = min(clocks, key=lambda c: c.t_)
                current_clocks = {c for c in clocks if c.same_time(min_clock)}

                self._t = min_clock.t_

                for obj in active_objects:
                    if obj.clock in current_clocks:
                        obj.run()

                for c in current_clocks:
                    if c.running:
                        c.advance()

                all_running = any(c.running for c in clocks)

        self._t = t_end
        self.after_run()

        logger.debug(f"Finished running network '{self.name}'")

    def stop(self):
        """Stop the simulation."""
        self._stopped = True
        logger.debug(f"Network '{self.name}' stopped")

    def __contains__(self, item):
        """Check if an object (by name) is in the network."""
        if isinstance(item, str):
            return any(obj.name == item for obj in _get_all_objects(self.objects))
        return item in _get_all_objects(self.objects)

    def __getitem__(self, name):
        """Get an object by name."""
        if not isinstance(name, str):
            raise TypeError(f"Expected string name, got {type(name).__name__}")
        for obj in _get_all_objects(self.objects):
            if obj.name == name:
                return obj
        raise KeyError(f"No object named '{name}' in network")

    def __len__(self):
        """Number of objects in the network."""
        return len(_get_all_objects(self.objects))

    def __iter__(self):
        """Iterate over objects in the network."""
        return iter(_get_all_objects(self.objects))

    def __repr__(self):
        obj_names = ', '.join(obj.name for obj in self.sorted_objects)
        return f"<Network '{self.name}' at t={self.t}, objects: {obj_names}>"
