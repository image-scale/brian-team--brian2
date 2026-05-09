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
- [ ] Logger can be created with different levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Log messages include module name and timestamp
- [ ] Preferences can be registered with defaults, types, and documentation
- [ ] Preferences can be read and modified at runtime
- [ ] Invalid preference values raise appropriate errors
- [ ] Preference system supports nested namespaces (e.g., core.network.schedule)
