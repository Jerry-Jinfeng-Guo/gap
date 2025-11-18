#!/usr/bin/env python3
"""
Test CPU vs GPU backend comparison for GAP solver
"""
from pathlib import Path
import sys

# Add the lib directory to path
build_dir = Path(__file__).parent.parent.parent / "build_cuda" / "lib"
if not build_dir.exists():
    build_dir = Path(__file__).parent.parent.parent / "build" / "lib"

sys.path.insert(0, str(build_dir))

import numpy as np

try:
    import gap_solver

    print("✓ GAP solver imported successfully")
except ImportError as e:
    print(f"❌ Failed to import gap_solver: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("GAP CPU vs GPU Backend Comparison Test")
print("=" * 80)
print()

# System parameters
base_voltage = 230000.0  # 230 kV
base_power = 100e6  # 100 MVA
base_impedance = (base_voltage**2) / base_power  # 529 Ω

# Bus data: [id, u_rated, bus_type, u_pu, u_angle, p, q]
# bus_type: 0=PQ, 1=PV, 2=SLACK
bus_data = [
    [1, base_voltage, 2, 1.05, 0.0, 0.0, 0.0],  # Slack bus
    [2, base_voltage, 0, 1.0, 0.0, -50e6, -30e6],  # PQ load
    [3, base_voltage, 0, 1.0, 0.0, -40e6, -25e6],  # PQ load
]

# Branch data: [id, from, to, r_ohm, x_ohm, b_siemens]
r_pu = 0.02
x_pu = 0.15
r_ohm = r_pu * base_impedance
x_ohm = x_pu * base_impedance

branch_data = [
    [1, 1, 2, r_ohm, x_ohm, 0.0],
    [2, 2, 3, r_ohm, x_ohm, 0.0],
]

print("Test System:")
print("  Base: 100 MVA, 230 kV")
print("  Buses: 3 (1 slack, 2 PQ)")
print("  Branches: 2")
print("  Total Load: 90 MW + 55 MVAR")
print()

backends = ["cpu", "gpu"]
results = {}

for backend in backends:
    print(f"Testing {backend.upper()} backend...")

    try:
        result = gap_solver.solve_simple_power_flow(
            bus_data,
            branch_data,
            tolerance=1e-6,
            max_iterations=50,
            verbose=False,
            backend=backend,
        )

        # Parse results
        n_bus = len(result) - 1
        metadata = result[-1]
        converged = bool(metadata[0])
        iterations = int(metadata[1])
        final_mismatch = metadata[2]

        results[backend] = {
            "converged": converged,
            "iterations": iterations,
            "final_mismatch": final_mismatch,
            "voltages": [],
        }

        for i in range(n_bus):
            real, imag, magnitude, angle_rad = result[i]
            results[backend]["voltages"].append(
                {
                    "bus": i + 1,
                    "magnitude": magnitude,
                    "angle_deg": np.degrees(angle_rad),
                }
            )

        status = "✅ CONVERGED" if converged else "❌ FAILED"
        print(f"  Status: {status}")
        print(f"  Iterations: {iterations}")
        print(f"  Final mismatch: {final_mismatch:.2e}")
        print()

    except Exception as e:
        print(f"  ❌ Error: {e}")
        results[backend] = {"error": str(e)}
        print()

# Compare results
print("=" * 80)
print("Results Comparison")
print("=" * 80)
print()

if "cpu" in results and "gpu" in results:
    cpu_r = results["cpu"]
    gpu_r = results["gpu"]

    if "error" not in cpu_r and "error" not in gpu_r:
        if cpu_r["converged"] and gpu_r["converged"]:
            print(
                f"{'Bus':<6} {'CPU |V| (pu)':<15} {'GPU |V| (pu)':<15} {'Difference':<15}"
            )
            print("-" * 60)

            max_diff = 0.0
            for cpu_v, gpu_v in zip(cpu_r["voltages"], gpu_r["voltages"]):
                diff = abs(cpu_v["magnitude"] - gpu_v["magnitude"])
                max_diff = max(max_diff, diff)
                print(
                    f"{cpu_v['bus']:<6} {cpu_v['magnitude']:<15.8f} {gpu_v['magnitude']:<15.8f} {diff:<15.2e}"
                )

            print("-" * 60)
            print(f"Maximum voltage difference: {max_diff:.2e} pu")
            print()

            if max_diff < 1e-6:
                print("✅ PASS: CPU and GPU results match within tolerance!")
            else:
                print("⚠️  WARNING: Results differ by more than 1e-6")
        else:
            print("⚠️  Cannot compare: One or both solvers did not converge")
    else:
        print("⚠️  Cannot compare: One or both solvers encountered errors")
else:
    print("⚠️  Cannot compare: Missing results from one or both backends")

print()
print("=" * 80)
print("Test Complete")
print("=" * 80)
