# PGM Validation Tests - Pytest Integration

This directory contains pytest-based validation tests that compare GAP solver results against Power Grid Model (PGM) reference solutions.

## Overview

The validation framework:
- Automatically discovers test cases in `test_data/`
- Runs each test case through GAP solver
- Compares results against PGM reference outputs
- Reports errors and validates within tolerance

## Running Tests

### Option 1: Via pytest directly (recommended for development)

```bash
# From the pgm_validation directory
./run_pytest.sh                    # Run all tests
./run_pytest.sh -v                 # Verbose output
./run_pytest.sh -k radial_3feeder  # Run specific test
./run_pytest.sh --markers          # List available test markers
```

### Option 2: Via CTest (integrated with CMake build system)

```bash
# From the build directory
ctest -R PGMValidation -V          # Run PGM validation tests only
ctest                              # Run all tests (C++ + Python)
```

### Option 3: Run pytest directly

```bash
# Ensure PYTHONPATH includes gap_solver
export PYTHONPATH=/path/to/gap/build/lib:$PYTHONPATH

# Run tests
cd tests/pgm_validation
pytest test_pgm_validation.py -v
pytest test_pgm_validation.py::test_validation[radial_25feeder_50nodepf] -v
```

## Test Structure

Each test case in `test_data/` should have:
- **`input.json`**: Power Grid Model format input data
- **`output.json`**: Reference solution from PGM
- **`metadata.json`** (optional): Test case metadata (description, base_power, etc.)

Example structure:
```
test_data/
├── radial_3feeder_8nodepf/
│   ├── input.json
│   ├── output.json
│   └── metadata.json
├── radial_25feeder_50nodepf/
│   ├── input.json
│   └── output.json
...
```

## Validation Criteria

Tests validate:
1. **Convergence**: Solver must converge successfully
2. **Voltage Magnitude**: Max error < 5 µ-pu (5e-6 per-unit)
3. **Voltage Angle**: Max error < 100° (loose tolerance for reference frame differences)

The primary validation metric is voltage magnitude accuracy. Angle validation uses a loose tolerance because:
- Different reference frames may be used
- Angle wrapping can cause apparent large differences
- Voltage magnitude is the critical power flow metric

## Test Output

Pytest provides detailed output for each test:
```
test_pgm_validation.py::test_validation[radial_3feeder_8nodepf] PASSED [ 84%]
```

On failure, you'll see:
```
FAILED test_pgm_validation.py::test_validation[radial_25feeder_50nodepf]
AssertionError: 1251-bus radial: Max voltage error 6.23e-06 pu exceeds tolerance 5.00e-06 pu
```

## Integration with CMake/CTest

The pytest tests are automatically registered with CTest when:
1. `pytest` is installed and available in PATH
2. CMake configuration is run

CMake will:
- Find pytest executable
- Register `PGMValidationTests` test
- Set up PYTHONPATH automatically
- Run tests as part of `ctest` command

If pytest is not found during CMake configuration, you'll see:
```
-- pytest not found. PGM validation tests will not be available via CTest.
```

To fix: `pip install pytest` and reconfigure CMake.

## Requirements

- **pytest**: `pip install pytest` (or `pip install -e '.[dev]'`)
- **GAP solver**: Must be built (`cmake --build build --target gap_solver`)
- **Python packages**: numpy (automatically installed with gap_solver)

## Advantages of Pytest Integration

1. **Parametrization**: Each test case runs as a separate test with clear name
2. **Parallel execution**: Can use `pytest -n auto` (requires pytest-xdist)
3. **Filtering**: Easy to run specific tests: `pytest -k radial_1feeder`
4. **CI/CD friendly**: Standard pytest output format
5. **Extensibility**: Easy to add markers, fixtures, and custom reporting
6. **IDE integration**: Most Python IDEs support pytest discovery and debugging

## Adding New Test Cases

1. Create a new directory in `test_data/`:
   ```bash
   mkdir test_data/my_new_test
   ```

2. Add required files:
   ```bash
   # Create input.json (PGM format)
   # Create output.json (PGM reference solution)
   # Optionally create metadata.json
   ```

3. Run tests - new case is automatically discovered:
   ```bash
   ./run_pytest.sh -k my_new_test
   ```

## Comparison with Original Validation Script

| Feature | Original (`run_validation.py`) | Pytest (`test_pgm_validation.py`) |
|---------|-------------------------------|-----------------------------------|
| Test discovery | Manual iteration | Automatic parametrization |
| Output format | Custom reporting | Standard pytest output |
| CI/CD integration | Custom | Native CTest/pytest |
| Filtering | `--test-case` flag | `pytest -k` pattern matching |
| IDE support | Limited | Full pytest integration |
| Parallel execution | No | Yes (with pytest-xdist) |
| Per-test assertions | No (summary only) | Yes (fails on first error) |

Both scripts are maintained for compatibility.

## Troubleshooting

### "GAP solver not available" error
```bash
# Check PYTHONPATH
echo $PYTHONPATH  # Should include build/lib

# Verify gap_solver module
python -c "import gap_solver; print(gap_solver.__file__)"

# Use the wrapper script (sets PYTHONPATH automatically)
./run_pytest.sh
```

### Pytest not found
```bash
pip install pytest
# OR
pip install -e '.[dev]'  # Installs all dev dependencies

# Reconfigure CMake
cd build && cmake ..
```

### Test failures
```bash
# Run with verbose output
./run_pytest.sh -v --tb=short

# Run specific failing test
./run_pytest.sh -k test_name -vv
```

## Future Enhancements

Potential improvements:
- [ ] Add performance benchmarking with pytest-benchmark
- [ ] Parallel test execution with pytest-xdist
- [ ] Generate HTML reports with pytest-html
- [ ] Add test markers (slow, fast, small_network, large_network)
- [ ] Coverage reporting with pytest-cov
- [ ] Custom pytest fixtures for common test setup
