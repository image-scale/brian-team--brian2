# Acceptance Criteria

## Task 1: Physical Units and Quantities System

### Acceptance Criteria
- [ ] Dimension class tracks 7 SI base dimensions (length, mass, time, current, temperature, amount, luminosity)
- [ ] Quantity class stores numeric value with associated dimensions
- [ ] Creating 5 * second gives a Quantity with value 5.0 and time dimension 1
- [ ] 500 * millisecond equals 0.5 when expressed in seconds
- [ ] Adding 1 * second + 500 * millisecond gives Quantity with value 1.5 seconds
- [ ] Adding quantities with mismatched dimensions raises DimensionMismatchError
- [ ] Multiplying quantities multiplies values and adds dimensions (e.g., amp * second = coulomb dimensions)
- [ ] Dividing quantities divides values and subtracts dimensions
- [ ] Quantities support numpy array operations (array of values with same dimension)
- [ ] Unit class represents a unit with a scale factor and dimension
- [ ] Dimensionless quantities work correctly (e.g., ratios)
- [ ] Power operations work (e.g., meter**2 gives area dimensions)
- [ ] Comparison operations work between same-dimension quantities
