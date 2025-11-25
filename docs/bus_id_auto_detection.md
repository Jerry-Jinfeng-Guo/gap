# Bus ID Auto-Detection Feature

## Overview

The GAP solver now automatically detects and handles multiple bus ID numbering schemes without requiring manual configuration. This eliminates recurring bugs caused by ID indexing mismatches.

## Supported ID Schemes

### 1. **0-based IDs** (e.g., 0, 1, 2, ...)
Common in modern software systems and JSON data formats.

```python
bus_data = [
    [0, 230000, 2, 1.05, 0, 0, 0],     # Bus 0: Slack
    [1, 230000, 0, 1.0, 0, -100e6, -50e6],   # Bus 1: Load
]
branch_data = [
    [1, 0, 1, 5.29, 52.9, 0],  # Branch from bus 0 to bus 1
]
```

### 2. **1-based IDs** (e.g., 1, 2, 3, ...)
Traditional in power system analysis (IEEE test cases, Power Grid Model).

```python
bus_data = [
    [1, 230000, 2, 1.05, 0, 0, 0],     # Bus 1: Slack
    [2, 230000, 0, 1.0, 0, -100e6, -50e6],   # Bus 2: Load
]
branch_data = [
    [1, 1, 2, 5.29, 52.9, 0],  # Branch from bus 1 to bus 2
]
```

### 3. **Non-sequential IDs** (e.g., 10, 20, 100, ...)
Used in large-scale systems with reserved ID ranges.

```python
bus_data = [
    [10, 230000, 2, 1.05, 0, 0, 0],    # Bus 10: Slack
    [20, 230000, 0, 1.0, 0, -100e6, -50e6],  # Bus 20: Load
]
branch_data = [
    [1, 10, 20, 5.29, 52.9, 0],  # Branch from bus 10 to bus 20
]
```

## Implementation Details

### Algorithm

The auto-detection is implemented in `src/admittance/cpu/cpu_admittance_matrix.cpp`:

1. **Scan Bus IDs**: Find minimum and maximum bus IDs
   ```cpp
   int min_bus_id = network_data.num_buses;
   int max_bus_id = -1;
   for (auto const& bus : network_data.buses) {
       min_bus_id = std::min(min_bus_id, bus.id);
       max_bus_id = std::max(max_bus_id, bus.id);
   }
   ```

2. **Create ID-to-Index Mapping**:
   ```cpp
   std::vector<int> id_to_idx(max_bus_id + 1, -1);  // Initialize with -1
   for (size_t idx = 0; idx < network_data.buses.size(); ++idx) {
       id_to_idx[network_data.buses[idx].id] = static_cast<int>(idx);
   }
   ```

3. **Map Branch Connections**:
   ```cpp
   int from_bus = id_to_idx[branch.from_bus];
   int to_bus = id_to_idx[branch.to_bus];
   ```

4. **Log Detection Results**:
   ```
   Bus ID range: [ 0 , 12 ], offset: 0   # 0-based IDs
   Bus ID range: [ 1 , 2 ], offset: 1     # 1-based IDs
   Bus ID range: [ 10 , 20 ], offset: 10  # Non-sequential IDs
   ```

### Python API

The Python binding (`solve_simple_power_flow`) no longer enforces ID validation:

```python
# All three ID schemes work seamlessly
result = gap_solver.solve_simple_power_flow(
    bus_data,      # Bus IDs auto-detected
    branch_data,   # Branch connections automatically mapped
    tolerance=1e-6,
    max_iterations=50
)
```

## Testing

Comprehensive tests in `tests/pgm_validation/test_id_auto_detection.py`:

```bash
$ python tests/pgm_validation/test_id_auto_detection.py

Test 1: 0-based Bus IDs (buses 0, 1)
  ✅ CONVERGED in 7 iterations
  Bus 0: 1.0500 pu, Bus 1: 0.9846 pu

Test 2: 1-based Bus IDs (buses 1, 2)
  ✅ CONVERGED in 7 iterations
  Bus 1: 1.0500 pu, Bus 2: 0.9846 pu

Test 3: Non-sequential Bus IDs (buses 10, 20)
  ✅ CONVERGED in 7 iterations
  Bus 10: 1.0500 pu, Bus 20: 0.9846 pu
```

## Benefits

1. **Eliminates Segfaults**: No more crashes from ID indexing mismatches
2. **Zero Configuration**: Works automatically without user intervention
3. **Multi-Source Compatibility**: Handles data from different systems seamlessly
4. **Debuggable**: Logs detected ID range for troubleshooting

## Migration Guide

### Before (Manual ID Handling)
```cpp
// Old: Hardcoded offset assumption
int from_idx = branch.from_bus - 1;  // Assumes 1-based IDs!
int to_idx = branch.to_bus - 1;
```

### After (Auto-Detection)
```cpp
// New: Robust ID mapping
int from_idx = id_to_idx[branch.from_bus];  // Works for any ID scheme
int to_idx = id_to_idx[branch.to_bus];
```

## Future Work

- **GPU Backend**: Implement same auto-detection for CUDA kernels
- **Appliances**: Extend ID mapping to appliance connections
- **Performance**: Optimize mapping for very large networks (>100k buses)

## Related Files

- Implementation: `src/admittance/cpu/cpu_admittance_matrix.cpp`
- Python Bindings: `bindings/solver_bindings.cpp`
- Tests: `tests/pgm_validation/test_id_auto_detection.py`
- C++ Debug Programs: `tests/validation/debug_simple_2bus.cpp`, `debug_radial_3feeder.cpp`
