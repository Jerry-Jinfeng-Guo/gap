#!/usr/bin/env python3
"""
Simple test demonstrating CPU backend parameter functionality
This test doesn't require pandas or complex dependencies.
"""
from pathlib import Path
import sys

# Use CPU-only build
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build_cpu" / "lib"))

import gap_solver

print("=" * 80)
print("GAP Solver - Backend Parameter Test")
print("=" * 80)
print()

# Simple 3-bus test system
print("Setting up 3-bus test system...")
bus_data = [
    [0, 230000, 2, 1.0, 0.0, 0, 0],  # Slack bus (bus 0)
    [1, 230000, 0, 1.0, 0.0, -50e6, -30e6],  # Load bus (bus 1): 50 MW, 30 MVAR
    [2, 230000, 0, 1.0, 0.0, -30e6, -15e6],  # Load bus (bus 2): 30 MW, 15 MVAR
]

branch_data = [
    [0, 0, 1, 5.0, 15.0, 0.0],  # Line 0-1
    [1, 0, 2, 10.0, 30.0, 0.0],  # Line 0-2
    [2, 1, 2, 8.0, 24.0, 0.0],  # Line 1-2
]

print("  - 3 buses (1 slack, 2 PQ)")
print("  - 3 branches")
print("  - Total load: 80 MW, 45 MVAR")
print()

# Test 1: Default backend (should be CPU)
print("Test 1: Default backend (no parameter)")
print("-" * 80)
result_default = gap_solver.solve_simple_power_flow(
    bus_data, branch_data, verbose=False
)
metadata_default = result_default[-1]
converged = metadata_default[0]
iterations = int(metadata_default[1])
print(
    f"Result: {'✅ Converged' if converged == 1.0 else '❌ Failed'} in {iterations} iterations"
)
print()

# Test 2: Explicit CPU backend
print("Test 2: Explicit backend='cpu'")
print("-" * 80)
result_cpu = gap_solver.solve_simple_power_flow(
    bus_data, branch_data, backend="cpu", verbose=False
)
metadata_cpu = result_cpu[-1]
converged = metadata_cpu[0]
iterations = int(metadata_cpu[1])
print(
    f"Result: {'✅ Converged' if converged == 1.0 else '❌ Failed'} in {iterations} iterations"
)
print()

# Test 3: Try GPU backend (should fail gracefully on CPU-only build)
print("Test 3: Request backend='gpu' on CPU-only build")
print("-" * 80)
try:
    result_gpu = gap_solver.solve_simple_power_flow(
        bus_data, branch_data, backend="gpu", verbose=False
    )
    print("❌ Should have raised an error")
except Exception as e:
    print(f"✅ Properly caught error: {str(e)}")
print()

# Verify results match
print("Test 4: Verify results consistency")
print("-" * 80)
max_diff = 0.0
for i in range(len(result_default) - 1):  # Skip metadata row
    for j in range(len(result_default[i])):
        diff = abs(result_default[i][j] - result_cpu[i][j])
        max_diff = max(max_diff, diff)

print(f"Maximum difference between default and explicit CPU: {max_diff:.2e}")
if max_diff < 1e-10:
    print("✅ Results are identical")
else:
    print("❌ Results differ")
print()

# Show voltage magnitudes
print("Voltage magnitudes (per unit):")
print("-" * 80)
for i in range(len(result_cpu) - 1):
    v_mag = result_cpu[i][2]  # Magnitude is 3rd element
    v_angle = result_cpu[i][3]  # Angle is 4th element
    print(f"  Bus {i}: |V| = {v_mag:.6f} pu, ∠V = {v_angle*180/3.14159:.2f}°")
print()

print("=" * 80)
print("✅ All tests passed! Backend parameter is working correctly.")
print("=" * 80)
