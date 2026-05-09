# Progress

## Round 1
**Task**: Task 1 — Implement the physical units and quantities system
**Files created**: neurosim/__init__.py, neurosim/units/__init__.py, neurosim/units/core.py, neurosim/units/baseunits.py, tests/test_units.py
**Commit**: Add a physical units system that tracks SI dimensions and enables dimension-checked arithmetic
**Acceptance**: 13/13 criteria met
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 2
**Task**: Task 2 — Implement standard units with SI prefixes and physical constants
**Files created**: neurosim/units/stdunits.py, neurosim/units/constants.py, tests/test_stdunits.py
**Commit**: Add standard units with SI prefixes commonly used in neuroscience
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for msecond), PASS on current state

## Round 3
**Task**: Task 3 — Implement the logging and preference system
**Files created**: neurosim/utils/__init__.py, neurosim/utils/logger.py, neurosim/utils/preferences.py, neurosim/core/__init__.py, tests/test_utils.py
**Commit**: Add a logging and preferences system for configuring simulator behavior
**Acceptance**: 6/6 criteria met
**Verification**: tests FAIL on previous state (ModuleNotFoundError for neurosim.utils), PASS on current state

## Round 4
**Task**: Task 4 — Implement simulation clocks
**Files created**: neurosim/core/clocks.py, tests/test_clocks.py
**Commit**: Add simulation clocks for managing time stepping during neural network simulation
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for Clock), PASS on current state

## Round 5
**Task**: Task 5 — Implement base classes for simulation objects
**Files created**: neurosim/core/base.py, tests/test_base.py
**Commit**: Add base classes for simulation objects with tracking, naming, and scheduling
**Acceptance**: 6/6 criteria met
**Verification**: tests FAIL on previous state (ImportError for Trackable/Nameable/SimObject), PASS on current state

## Round 6
**Task**: Task 6 — Implement the network that manages simulation runs
**Files created**: neurosim/core/network.py, tests/test_network.py
**Commit**: Add Network class for managing simulation runs with scheduling and time tracking
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for Network), PASS on current state

## Round 7
**Task**: Task 7 — Implement equation parsing
**Files created**: neurosim/equations/__init__.py, neurosim/equations/equations.py, tests/test_equations.py
**Commit**: Add equation parsing for differential equations, parameters, and subexpressions
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for equations module), PASS on current state

## Round 8
**Task**: Task 8 — Implement state variables system
**Files created**: neurosim/core/variables.py, tests/test_variables.py
**Commit**: Add state variables system for managing neuron and synapse state
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for variables), PASS on current state

## Round 9
**Task**: Task 9 — Implement the group base class
**Files created**: neurosim/groups/__init__.py, neurosim/groups/group.py, tests/test_group.py
**Commit**: Add Group base class with state variable access, attribute access, and indexing
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for groups), PASS on current state

## Round 10
**Task**: Task 10 — Implement NeuronGroup with differential equations and spiking
**Files created**: neurosim/groups/neurongroup.py, tests/test_neurongroup.py
**Commit**: Add NeuronGroup with differential equations, thresholds, resets, and refractory periods
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ImportError for NeuronGroup), PASS on current state

## Round 11
**Task**: Task 11 — Implement spike monitoring
**Files created**: neurosim/monitors/__init__.py, neurosim/monitors/spikemonitor.py, tests/test_spikemonitor.py
**Commit**: Add SpikeMonitor for recording neuron spikes
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state (ModuleNotFoundError for monitors), PASS on current state
