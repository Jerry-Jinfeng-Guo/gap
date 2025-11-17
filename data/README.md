# Example Data Files

This directory contains example Power Grid Model (PGM) format input files for testing and demonstrating the GAP solver.

## Files

### `pgm/network_1.json`
**4-bus network with transmission lines**
- Components: 4 nodes, 3 lines, 1 source, 2 loads
- Features: Basic line modeling with positive/zero sequence parameters
- Use case: Testing standard transmission line calculations
- Example: `./build/bin/gap_main -i data/pgm/network_1.json -o results.json`

### `pgm/network_2.json`
**Network with transformer**
- Components: Nodes + transformer elements
- Features: Transformer modeling with tap ratios and phase shifts
- Use case: Testing transformer power flow calculations
- Example: `./build/bin/gap_main -i data/pgm/network_2.json -o results.json -v`

### `pgm/network_3.json`
**Network with generic branches**
- Components: Nodes + generic branch elements
- Features: Generic branch modeling (flexible impedance definitions)
- Use case: Testing generalized branch calculations
- Example: `./build/bin/gap_main -i data/pgm/network_3.json -o results.json -v`

## Format

All files follow the [Power Grid Model](https://power-grid-model.github.io/power-grid-model/) JSON input format specification, which includes:
- Network topology (nodes, branches)
- Component parameters (impedances, ratings)
- Load and source definitions
- System base values

## Usage in Tests

These files are also used by:
- **Unit tests**: `tests/unit/test_pgm_io.cpp` validates PGM JSON parsing
- **Integration tests**: Ensures different component types are handled correctly
- **Documentation examples**: README.md uses these for usage demonstrations

## Adding New Examples

To add a new example file:
1. Create a valid PGM JSON input file
2. Place it in `data/pgm/`
3. Add description here
4. Optionally add a test case in `test_pgm_io.cpp`
