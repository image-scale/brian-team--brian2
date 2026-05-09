# Acceptance Criteria

## Task 1: Physical Units and Quantities System

### Acceptance Criteria
- [x] Dimension class tracks 7 SI base dimensions (length, mass, time, current, temperature, amount, luminosity)
- [x] Quantity class stores numeric value with associated dimensions
- [x] Creating 5 * second gives a Quantity with value 5.0 and time dimension 1
- [x] 500 * millisecond equals 0.5 when expressed in seconds
- [x] Adding 1 * second + 500 * millisecond gives Quantity with value 1.5 seconds
- [x] Adding quantities with mismatched dimensions raises DimensionMismatchError
- [x] Multiplying quantities multiplies values and adds dimensions (e.g., amp * second = coulomb dimensions)
- [x] Dividing quantities divides values and subtracts dimensions
- [x] Quantities support numpy array operations (array of values with same dimension)
- [x] Unit class represents a unit with a scale factor and dimension
- [x] Dimensionless quantities work correctly (e.g., ratios)
- [x] Power operations work (e.g., meter**2 gives area dimensions)
- [x] Comparison operations work between same-dimension quantities

## Task 2: Standard Units with SI Prefixes

### Acceptance Criteria
- [x] SI prefixes are available for all base units (milli, micro, nano, pico, kilo, mega, giga, etc.)
- [x] Common neuroscience units available: mvolt (millivolt), msecond (millisecond), namp (nanoamp), nsiemens, etc.
- [x] Short aliases work: ms for millisecond, mV for millivolt, nA for nanoamp, Hz for hertz
- [x] 1000 * msecond equals 1 * second
- [x] 1 * mvolt equals 0.001 * volt
- [x] Physical constants available: electron charge, Boltzmann constant
- [x] Area and volume units work: cm (centimeter), um (micrometer), cm2, um2
- [x] All standard neuroscience prefixed units can be imported

## Task 3: Logging and Preference System

### Acceptance Criteria
- [x] Logger can be created with different levels (DEBUG, INFO, WARNING, ERROR)
- [x] Log messages include module name and timestamp
- [x] Preferences can be registered with defaults, types, and documentation
- [x] Preferences can be read and modified at runtime
- [x] Invalid preference values raise appropriate errors
- [x] Preference system supports nested namespaces (e.g., core.network.schedule)

## Task 4: Simulation Clocks

### Acceptance Criteria
- [x] Clock tracks current time and timestep number
- [x] Clock has configurable dt (time step size) defaulting to 0.1 ms
- [x] Clock can advance time by one dt and increment timestep
- [x] Clock can be set to run from start time to end time
- [x] Clock raises StopIteration when end time is reached
- [x] A global defaultclock is available for convenience
- [x] Multiple clocks can coexist with different dt values
- [x] Clock time has proper units (seconds)

## Task 5: Base Classes for Simulation Objects

### Acceptance Criteria
- [x] Nameable mixin provides automatic or custom naming for objects
- [x] Names are unique within a simulation context
- [x] SimObject base class provides common functionality for all simulation objects
- [x] Objects can be associated with a clock
- [x] Objects have unique identifiers for tracking
- [x] Objects can be scheduled at specific phases (start, groups, thresholds, resets, end)

## Task 6: Network for Simulation Runs

### Acceptance Criteria
- [x] Network can be created with a name and optional list of objects
- [x] Objects can be added and removed from the network
- [x] Network.run(duration) simulates for the specified duration
- [x] Objects are updated in order: schedule phase, then order, then name
- [x] Network tracks current simulation time as a Quantity
- [x] Network.stop() halts the simulation
- [x] before_run and after_run lifecycle methods are called on objects
- [x] Inactive objects are not updated during simulation

## Task 7: Equation Parsing

### Acceptance Criteria
- [x] Equations can be created from multi-line strings
- [x] Differential equations parsed: "dv/dt = -v/tau : volt"
- [x] Subexpressions (aliases) parsed: "v_total = v + v_rest : volt"
- [x] Parameters parsed: "tau : second"
- [x] Variables have associated units/dimensions
- [x] Expression objects store the right-hand side with identifiers
- [x] Equations container provides access by variable name
- [x] Invalid equation syntax raises EquationError

## Task 8: State Variables System

### Acceptance Criteria
- [x] Variable base class stores name, dimensions, dtype, and flags
- [x] Constant class stores immutable scalar values
- [x] ArrayVariable stores numpy arrays with get/set methods
- [x] Variables container manages multiple variables
- [x] Variables can be added by type (constant, array)
- [x] Variable dimensions match physical units
- [x] Scalar vs. vector variables are distinguished
- [x] Read-only variables reject set operations

## Task 9: Group Base Class

### Acceptance Criteria
- [x] Group inherits from SimObject and owns variables
- [x] Attribute access to state variables (group.v, group.v_)
- [x] Setting variables via attribute (group.v = value)
- [x] Indexing support (group[0], group[0:10])
- [x] Group has N property for size
- [x] Variables created from equations
- [x] Unit checking on variable assignment
- [x] Read-only variables cannot be set via attribute
