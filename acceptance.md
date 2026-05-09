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
- [ ] Clock tracks current time and timestep number
- [ ] Clock has configurable dt (time step size) defaulting to 0.1 ms
- [ ] Clock can advance time by one dt and increment timestep
- [ ] Clock can be set to run from start time to end time
- [ ] Clock raises StopIteration when end time is reached
- [ ] A global defaultclock is available for convenience
- [ ] Multiple clocks can coexist with different dt values
- [ ] Clock time has proper units (seconds)
