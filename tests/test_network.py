"""
Tests for the Network class.
"""

import pytest
from numpy.testing import assert_allclose

from neurosim.core import Network, SimObject, Clock, defaultclock, SCHEDULE_PHASES
from neurosim.units import second
from neurosim.units.stdunits import ms


class CountingObject(SimObject):
    """Test object that counts how many times run() is called."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_count = 0
        self.before_run_called = False
        self.after_run_called = False

    def before_run(self, namespace=None):
        self.before_run_called = True

    def run(self):
        self.run_count += 1

    def after_run(self):
        self.after_run_called = True


class OrderTracker(SimObject):
    """Test object that tracks execution order."""

    execution_order = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        OrderTracker.execution_order.append(self.name)

    @classmethod
    def reset_order(cls):
        cls.execution_order = []


class TestNetworkCreation:
    """Tests for Network creation."""

    def test_network_default_name(self):
        """Network gets auto-generated name."""
        net = Network()
        assert 'network' in net.name

    def test_network_custom_name(self):
        """Network accepts custom name."""
        net = Network(name='mynet')
        assert net.name == 'mynet'

    def test_network_unique_names(self):
        """Networks get unique names."""
        net1 = Network()
        net2 = Network()
        assert net1.name != net2.name

    def test_network_initial_time(self):
        """Network starts at time 0."""
        net = Network()
        assert_allclose(float(net.t / second), 0.0)

    def test_network_with_objects(self):
        """Network accepts objects on creation."""
        obj1 = SimObject(name='obj1')
        obj2 = SimObject(name='obj2')
        net = Network(obj1, obj2)
        assert len(net) == 2


class TestNetworkAddRemove:
    """Tests for adding and removing objects."""

    def test_add_single_object(self):
        """Can add a single object."""
        net = Network()
        obj = SimObject(name='test')
        net.add(obj)
        assert 'test' in net

    def test_add_multiple_objects(self):
        """Can add multiple objects at once."""
        net = Network()
        obj1 = SimObject(name='obj1')
        obj2 = SimObject(name='obj2')
        net.add(obj1, obj2)
        assert len(net) == 2

    def test_add_list_of_objects(self):
        """Can add a list of objects."""
        net = Network()
        objs = [SimObject(name=f'obj{i}') for i in range(3)]
        net.add(objs)
        assert len(net) == 3

    def test_add_dict_values(self):
        """Adding dict adds its values."""
        net = Network()
        objs = {'a': SimObject(name='obja'), 'b': SimObject(name='objb')}
        net.add(objs)
        assert 'obja' in net
        assert 'objb' in net

    def test_add_invalid_type_raises(self):
        """Adding invalid type raises error."""
        net = Network()
        with pytest.raises(TypeError):
            net.add("not an object")

    def test_remove_object(self):
        """Can remove an object."""
        obj = SimObject(name='removeme')
        net = Network(obj)
        assert 'removeme' in net
        net.remove(obj)
        assert 'removeme' not in net

    def test_remove_list(self):
        """Can remove a list of objects."""
        objs = [SimObject(name=f'obj{i}') for i in range(3)]
        net = Network(*objs)
        net.remove(objs[:2])
        assert len(net) == 1


class TestNetworkRun:
    """Tests for Network.run()."""

    def setup_method(self):
        """Reset defaultclock before each test."""
        defaultclock.reset()
        defaultclock.dt = 0.1 * ms

    def test_run_calls_object_run(self):
        """Running network calls object run methods."""
        obj = CountingObject(name='counter')
        net = Network(obj)
        net.run(1 * ms)
        assert obj.run_count == 10

    def test_run_updates_time(self):
        """Running network updates simulation time."""
        net = Network(SimObject())
        net.run(5 * ms)
        assert_allclose(float(net.t / ms), 5.0)

    def test_run_accepts_float(self):
        """Run accepts float duration (seconds)."""
        obj = CountingObject(name='counter')
        net = Network(obj)
        net.run(0.001)  # 1 ms
        assert obj.run_count == 10

    def test_run_negative_duration_raises(self):
        """Negative duration raises error."""
        net = Network()
        with pytest.raises(ValueError):
            net.run(-1 * ms)

    def test_run_empty_network(self):
        """Running empty network does nothing."""
        net = Network()
        net.run(1 * ms)

    def test_run_calls_before_run(self):
        """Run calls before_run on objects."""
        obj = CountingObject(name='test')
        net = Network(obj)
        net.run(1 * ms)
        assert obj.before_run_called

    def test_run_calls_after_run(self):
        """Run calls after_run on objects."""
        obj = CountingObject(name='test')
        net = Network(obj)
        net.run(1 * ms)
        assert obj.after_run_called

    def test_inactive_objects_not_run(self):
        """Inactive objects are not updated."""
        obj = CountingObject(name='inactive')
        obj.active = False
        net = Network(obj)
        net.run(1 * ms)
        assert obj.run_count == 0


class TestNetworkScheduling:
    """Tests for object scheduling order."""

    def setup_method(self):
        """Reset execution order and clock."""
        OrderTracker.reset_order()
        defaultclock.reset()
        defaultclock.dt = 1 * ms

    def test_schedule_by_when(self):
        """Objects are scheduled by their when attribute."""
        obj_end = OrderTracker(when='end', name='end_obj')
        obj_start = OrderTracker(when='start', name='start_obj')
        obj_groups = OrderTracker(when='groups', name='groups_obj')

        net = Network(obj_end, obj_start, obj_groups)
        net.run(1 * ms)

        assert OrderTracker.execution_order == ['start_obj', 'groups_obj', 'end_obj']

    def test_schedule_by_order(self):
        """Within same phase, objects sorted by order."""
        obj_high = OrderTracker(when='start', order=10, name='high')
        obj_low = OrderTracker(when='start', order=1, name='low')
        obj_mid = OrderTracker(when='start', order=5, name='mid')

        net = Network(obj_high, obj_low, obj_mid)
        net.run(1 * ms)

        assert OrderTracker.execution_order == ['low', 'mid', 'high']

    def test_schedule_by_name(self):
        """Ties in order resolved by name."""
        obj_b = OrderTracker(when='start', order=0, name='bbb')
        obj_a = OrderTracker(when='start', order=0, name='aaa')
        obj_c = OrderTracker(when='start', order=0, name='ccc')

        net = Network(obj_b, obj_a, obj_c)
        net.run(1 * ms)

        assert OrderTracker.execution_order == ['aaa', 'bbb', 'ccc']

    def test_default_schedule(self):
        """Default schedule contains all phases."""
        net = Network()
        schedule = net.schedule
        for phase in SCHEDULE_PHASES:
            assert phase in schedule

    def test_custom_schedule(self):
        """Network accepts custom schedule."""
        net = Network()
        net.schedule = ['start', 'end']
        assert net.schedule == ['start', 'end']


class TestNetworkStop:
    """Tests for stopping simulation."""

    def setup_method(self):
        defaultclock.reset()
        defaultclock.dt = 1 * ms

    def test_stop_halts_simulation(self):
        """Calling stop() halts the simulation."""
        class StoppingObject(SimObject):
            def __init__(self, network, **kwargs):
                super().__init__(**kwargs)
                self.network = network
                self.count = 0

            def run(self):
                self.count += 1
                if self.count >= 5:
                    self.network.stop()

        net = Network()
        obj = StoppingObject(net, name='stopper')
        net.add(obj)
        net.run(100 * ms)

        assert obj.count == 5


class TestNetworkTimeTracking:
    """Tests for time tracking."""

    def setup_method(self):
        defaultclock.reset()
        defaultclock.dt = 1 * ms

    def test_time_has_units(self):
        """Network time is a Quantity."""
        net = Network()
        assert hasattr(net.t, 'dim')
        assert net.t.dim == second.dim

    def test_time_underscore(self):
        """Network t_ returns float seconds."""
        net = Network(SimObject())
        net.run(5 * ms)
        assert_allclose(net.t_, 0.005)

    def test_time_accumulates(self):
        """Time accumulates across multiple runs."""
        net = Network(SimObject())
        net.run(5 * ms)
        net.run(5 * ms)
        assert_allclose(float(net.t / ms), 10.0)


class TestNetworkAccess:
    """Tests for accessing objects in network."""

    def test_contains_by_name(self):
        """Can check if name is in network."""
        obj = SimObject(name='findme')
        net = Network(obj)
        assert 'findme' in net
        assert 'notfound' not in net

    def test_contains_by_object(self):
        """Can check if object is in network."""
        obj = SimObject(name='test')
        net = Network(obj)
        assert obj in net

    def test_getitem_by_name(self):
        """Can get object by name."""
        obj = SimObject(name='myobj')
        net = Network(obj)
        assert net['myobj'] is obj

    def test_getitem_missing_raises(self):
        """Getting missing object raises KeyError."""
        net = Network()
        with pytest.raises(KeyError):
            net['missing']

    def test_len(self):
        """len() returns number of objects."""
        objs = [SimObject(name=f'obj{i}') for i in range(5)]
        net = Network(*objs)
        assert len(net) == 5

    def test_iter(self):
        """Can iterate over objects."""
        objs = [SimObject(name=f'obj{i}') for i in range(3)]
        net = Network(*objs)
        net_objs = list(net)
        assert len(net_objs) == 3


class TestNetworkRepr:
    """Tests for Network repr."""

    def test_repr_contains_name(self):
        """Repr contains network name."""
        net = Network(name='testnet')
        r = repr(net)
        assert 'testnet' in r

    def test_repr_contains_time(self):
        """Repr contains current time."""
        net = Network()
        r = repr(net)
        assert 't=' in r


class TestMultipleClocks:
    """Tests for networks with multiple clocks."""

    def setup_method(self):
        defaultclock.reset()
        defaultclock.dt = 0.1 * ms

    def test_different_clocks_run_correctly(self):
        """Objects with different clocks run at correct rates."""
        clock1 = Clock(dt=1 * ms, name='clock1')
        clock2 = Clock(dt=2 * ms, name='clock2')

        obj1 = CountingObject(clock=clock1, name='obj1')
        obj2 = CountingObject(clock=clock2, name='obj2')

        net = Network(obj1, obj2)
        net.run(10 * ms)

        assert obj1.run_count == 10
        assert obj2.run_count == 5
