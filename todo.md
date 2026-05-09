# Todo

## Plan
Implement the neural simulator in dependency order. Start with the units system which is foundational, then clocks and simulation framework, then neuron groups with equations, then synapses and monitors, finally input generators. Each task delivers a complete feature with tests.

## Tasks
- [x] Task 1: Implement the physical units and quantities system with dimension tracking, arithmetic operations, and unit checking (units module + tests)
- [x] Task 2: Implement standard units like millivolt, millisecond, nanoamp and physical constants (standard units + tests)
- [x] Task 3: Implement the logging and preference system for configuration (utils + tests)
- [x] Task 4: Implement simulation clocks that manage time stepping with a default clock (clocks + tests)
- [x] Task 5: Implement base classes for simulation objects including tracking and naming (core base + tests)
- [x] Task 6: Implement the network that manages simulation runs with scheduling (network + tests)
- [>] Task 7: Implement equation parsing that converts user-defined differential equations into usable form (equations + tests)
- [ ] Task 8: Implement state variables system for neuron/synapse state storage (variables + tests)
- [ ] Task 9: Implement the group base class with state variable access (group + tests)
- [ ] Task 10: Implement neuron groups that define neuron dynamics with equations (neurongroup + tests)
- [ ] Task 11: Implement spike monitoring to record when neurons fire (spike monitor + tests)
- [ ] Task 12: Implement state monitoring to record neuron variables over time (state monitor + tests)
- [ ] Task 13: Implement synaptic connections between neuron groups (synapses + tests)
- [ ] Task 14: Implement spike generator groups that produce spikes at specified times (spike generator + tests)
- [ ] Task 15: Implement Poisson spike generators for random input (poisson group + tests)
- [ ] Task 16: Implement timed arrays for time-varying input signals (timed array + tests)
