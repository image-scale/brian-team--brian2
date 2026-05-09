"""
Tests for the base classes for simulation objects.
"""

import pytest
import gc

from neurosim.core import (
    Trackable,
    Nameable,
    SimObject,
    SCHEDULE_PHASES,
    Clock,
    defaultclock,
)
from neurosim.units.stdunits import ms


class TestTrackable:
    """Tests for the Trackable base class."""

    def test_trackable_instances_tracked(self):
        """Trackable instances are tracked."""
        class MyTrackable(Trackable):
            pass

        obj1 = MyTrackable()
        obj2 = MyTrackable()
        instances = MyTrackable.__instances__()
        refs = [ref() for ref in instances if ref() is not None]
        assert obj1 in refs
        assert obj2 in refs

    def test_trackable_weak_references(self):
        """Tracked instances use weak references."""
        class MyTrackable(Trackable):
            pass

        obj = MyTrackable()
        instances_before = len([r for r in MyTrackable.__instances__() if r() is not None])
        del obj
        gc.collect()
        instances_after = len([r for r in MyTrackable.__instances__() if r() is not None])
        assert instances_after < instances_before

    def test_trackable_subclass_tracking(self):
        """Subclasses are tracked under parent class too."""
        class Parent(Trackable):
            pass

        class Child(Parent):
            pass

        child = Child()
        parent_instances = [r() for r in Parent.__instances__() if r() is not None]
        child_instances = [r() for r in Child.__instances__() if r() is not None]
        assert child in parent_instances
        assert child in child_instances


class TestNameable:
    """Tests for the Nameable mixin."""

    def test_nameable_explicit_name(self):
        """Nameable accepts explicit name."""
        obj = Nameable('myname')
        assert obj.name == 'myname'

    def test_nameable_auto_name(self):
        """Nameable with * suffix auto-generates name."""
        obj = Nameable('test*')
        assert obj.name.startswith('test')
        assert not obj.name.endswith('*')

    def test_nameable_unique_auto_names(self):
        """Multiple objects get unique auto-names."""
        obj1 = Nameable('auto*')
        obj2 = Nameable('auto*')
        assert obj1.name != obj2.name
        assert obj1.name == 'auto'
        assert obj2.name == 'auto_1'

    def test_nameable_has_id(self):
        """Nameable objects have unique IDs."""
        obj1 = Nameable('test1')
        obj2 = Nameable('test2')
        assert obj1.id is not None
        assert obj2.id is not None
        assert obj1.id != obj2.id

    def test_nameable_invalid_name_type(self):
        """Non-string name raises TypeError."""
        with pytest.raises(TypeError):
            Nameable(123)

    def test_nameable_invalid_name_format(self):
        """Invalid identifier name raises ValueError."""
        with pytest.raises(ValueError):
            Nameable('123invalid')
        with pytest.raises(ValueError):
            Nameable('has-dash')
        with pytest.raises(ValueError):
            Nameable('has space')

    def test_nameable_valid_names(self):
        """Various valid names are accepted."""
        Nameable('_private')
        Nameable('CamelCase')
        Nameable('with_underscore')
        Nameable('Name123')

    def test_nameable_name_is_readonly(self):
        """Name property cannot be set after creation."""
        obj = Nameable('readonly')
        with pytest.raises(AttributeError):
            obj.name = 'newname'


class TestSimObject:
    """Tests for the SimObject base class."""

    def setup_method(self):
        """Reset defaultclock before each test."""
        defaultclock.reset()
        defaultclock.dt = 0.1 * ms

    def test_simobject_default_name(self):
        """SimObject gets auto-generated name."""
        obj = SimObject()
        assert 'simobject' in obj.name

    def test_simobject_custom_name(self):
        """SimObject accepts custom name."""
        obj = SimObject(name='myobj')
        assert obj.name == 'myobj'

    def test_simobject_unique_names(self):
        """SimObjects get unique names."""
        obj1 = SimObject()
        obj2 = SimObject()
        assert obj1.name != obj2.name

    def test_simobject_uses_defaultclock(self):
        """SimObject uses defaultclock when no clock specified."""
        obj = SimObject()
        assert obj.clock is defaultclock

    def test_simobject_custom_clock(self):
        """SimObject accepts custom clock."""
        clock = Clock(dt=1 * ms, name='custom_clock')
        obj = SimObject(clock=clock)
        assert obj.clock is clock

    def test_simobject_dt_creates_clock(self):
        """SimObject with dt creates its own clock."""
        obj = SimObject(dt=2 * ms, name='dtobj')
        assert obj.clock is not defaultclock
        assert float(obj.clock.dt / ms) == 2.0

    def test_simobject_dt_and_clock_error(self):
        """Specifying both dt and clock raises error."""
        clock = Clock(dt=1 * ms)
        with pytest.raises(ValueError):
            SimObject(dt=1 * ms, clock=clock)

    def test_simobject_default_when(self):
        """SimObject defaults to 'start' scheduling phase."""
        obj = SimObject()
        assert obj.when == 'start'

    def test_simobject_custom_when(self):
        """SimObject accepts custom scheduling phase."""
        obj = SimObject(when='thresholds')
        assert obj.when == 'thresholds'

    def test_simobject_invalid_when(self):
        """Invalid scheduling phase raises error."""
        with pytest.raises(ValueError):
            SimObject(when='invalid')

    def test_simobject_when_setter(self):
        """Scheduling phase can be changed."""
        obj = SimObject(when='start')
        obj.when = 'end'
        assert obj.when == 'end'

    def test_simobject_when_setter_invalid(self):
        """Setting invalid scheduling phase raises error."""
        obj = SimObject()
        with pytest.raises(ValueError):
            obj.when = 'invalid'

    def test_simobject_default_order(self):
        """SimObject defaults to order 0."""
        obj = SimObject()
        assert obj.order == 0

    def test_simobject_custom_order(self):
        """SimObject accepts custom order."""
        obj = SimObject(order=5)
        assert obj.order == 5

    def test_simobject_order_setter(self):
        """Order can be changed."""
        obj = SimObject()
        obj.order = 10
        assert obj.order == 10

    def test_simobject_default_active(self):
        """SimObject defaults to active."""
        obj = SimObject()
        assert obj.active is True

    def test_simobject_active_setter(self):
        """Active can be toggled."""
        obj = SimObject()
        obj.active = False
        assert obj.active is False
        obj.active = True
        assert obj.active is True

    def test_simobject_has_id(self):
        """SimObject has unique ID."""
        obj1 = SimObject()
        obj2 = SimObject()
        assert obj1.id is not None
        assert obj1.id != obj2.id

    def test_simobject_contained_objects(self):
        """SimObject has contained_objects list."""
        obj = SimObject()
        assert obj.contained_objects == []

    def test_simobject_repr(self):
        """SimObject has informative repr."""
        obj = SimObject(when='groups', order=3, name='test')
        r = repr(obj)
        assert 'SimObject' in r
        assert 'test' in r
        assert 'groups' in r
        assert '3' in r

    def test_simobject_lifecycle_methods(self):
        """SimObject has lifecycle methods."""
        obj = SimObject()
        obj.before_run()
        obj.run()
        obj.after_run()


class TestSchedulePhases:
    """Tests for scheduling phases."""

    def test_schedule_phases_defined(self):
        """SCHEDULE_PHASES contains expected phases."""
        assert 'start' in SCHEDULE_PHASES
        assert 'groups' in SCHEDULE_PHASES
        assert 'thresholds' in SCHEDULE_PHASES
        assert 'synapses' in SCHEDULE_PHASES
        assert 'resets' in SCHEDULE_PHASES
        assert 'end' in SCHEDULE_PHASES

    def test_schedule_phases_order(self):
        """SCHEDULE_PHASES is in correct order."""
        phases = list(SCHEDULE_PHASES)
        assert phases.index('start') < phases.index('groups')
        assert phases.index('groups') < phases.index('thresholds')
        assert phases.index('thresholds') < phases.index('synapses')
        assert phases.index('synapses') < phases.index('resets')
        assert phases.index('resets') < phases.index('end')

    def test_all_phases_valid_for_simobject(self):
        """All phases can be used with SimObject."""
        for phase in SCHEDULE_PHASES:
            obj = SimObject(when=phase)
            assert obj.when == phase


class TestContainedObjects:
    """Tests for contained objects and cascading active state."""

    def test_active_cascades_to_contained(self):
        """Setting active cascades to contained objects."""
        parent = SimObject(name='parent')
        child1 = SimObject(name='child1')
        child2 = SimObject(name='child2')
        parent._contained_objects.extend([child1, child2])

        parent.active = False
        assert child1.active is False
        assert child2.active is False

        parent.active = True
        assert child1.active is True
        assert child2.active is True


class TestNameUniqueness:
    """Tests for name uniqueness within simulation context."""

    def test_auto_names_increment(self):
        """Auto-names increment to avoid collision."""
        objs = [Nameable('obj*') for _ in range(5)]
        names = [obj.name for obj in objs]
        assert len(names) == len(set(names))

    def test_explicit_names_allowed_duplicates(self):
        """Explicit names (without *) are allowed even if duplicate."""
        obj1 = Nameable('duplicate')
        obj2 = Nameable('duplicate')
        assert obj1.name == obj2.name == 'duplicate'

    def test_simobject_auto_names_unique(self):
        """SimObject auto-names are unique."""
        objs = [SimObject() for _ in range(5)]
        names = [obj.name for obj in objs]
        assert len(names) == len(set(names))
