"""
Simple test to verify GAP solver with bus power data
"""
from pathlib import Path
import sys

# Add the lib directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build" / "lib"))

import numpy as np

import gap_solver

print("=" * 70)
print("GAP Simple 2-Bus Test with Bug Fix Verification")
print("=" * 70)
print()

print("📋 Test Setup:")
print("  System: 100 MVA base, 230 kV")
print("  Bus 1: Slack at 1.05 pu")
print("  Bus 2: Load 100 MW + j50 MVAR")
print("  Line: 0.01 + j0.1 pu impedance")
print()

# System parameters
base_voltage = 230000.0  # 230 kV
base_power = 100e6  # 100 MVA
base_impedance = (base_voltage**2) / base_power  # 529 Ω

# Bus data: [id, u_rated, bus_type, u_pu, u_angle, p, q]
# bus_type: 0=PQ, 1=PV, 2=SLACK
bus_data = [
    [1, base_voltage, 2, 1.05, 0.0, 0.0, 0.0],  # Slack bus
    [
        2,
        base_voltage,
        0,
        1.0,
        0.0,
        -100e6,
        -50e6,
    ],  # PQ bus with load (negative = consumption)
]

# Branch data: [id, from, to, r_ohm, x_ohm, b_siemens]
r_pu = 0.01
x_pu = 0.1
r_ohm = r_pu * base_impedance  # 5.29 Ω
x_ohm = x_pu * base_impedance  # 52.9 Ω

branch_data = [[1, 1, 2, r_ohm, x_ohm, 0.0]]

print("🔧 Calling GAP Solver...")
print("  Tolerance: 1e-6")
print("  Max iterations: 50")
print()

result = gap_solver.solve_simple_power_flow(
    bus_data, branch_data, tolerance=1e-6, max_iterations=50, verbose=False
)

# Parse results
n_bus = len(result) - 1
metadata = result[-1]
converged = bool(metadata[0])
iterations = int(metadata[1])
final_mismatch = metadata[2]

print("📊 GAP Results:")
print(f"  Status: {'✅ CONVERGED' if converged else '❌ FAILED'}")
print(f"  Iterations: {iterations}")
print(f"  Final mismatch: {final_mismatch:.2e}")
print()

if converged or iterations == 50:
    print("Bus Voltages:")
    for i in range(n_bus):
        real, imag, magnitude, angle_rad = result[i]
        angle_deg = np.degrees(angle_rad)
        print(
            f"  Bus {i+1}: {magnitude:.4f}∠{angle_deg:7.3f}° pu  ({real:+.4f} {imag:+.4f}j)"
        )
    print()

    print("📊 PGM Reference Solution:")
    print("  Bus 1: 1.0439∠-0.53° pu")
    print("  Bus 2: 0.9780∠-5.87° pu")
    print()

    # Compare with PGM reference
    pgm_ref = {
        1: (1.0439, -0.53),
        2: (0.9780, -5.87),
    }

    print("📊 Comparison:")
    all_pass = True
    for i in range(n_bus):
        bus_id = i + 1
        _, _, gap_mag, gap_ang_rad = result[i]
        gap_ang_deg = np.degrees(gap_ang_rad)

        ref_mag, ref_ang = pgm_ref[bus_id]

        mag_error = abs(gap_mag - ref_mag)
        ang_error = abs(gap_ang_deg - ref_ang)

        mag_pct = (mag_error / ref_mag) * 100

        mag_pass = mag_pct < 1.0  # 1% tolerance
        ang_pass = ang_error < 1.0  # 1 degree tolerance

        pass_str = "✅ PASS" if (mag_pass and ang_pass) else "❌ FAIL"

        print(f"  Bus {bus_id}: {pass_str}")
        print(
            f"    Magnitude: GAP={gap_mag:.4f}, PGM={ref_mag:.4f}, Error={mag_pct:.2f}%"
        )
        print(
            f"    Angle:     GAP={gap_ang_deg:7.3f}°, PGM={ref_ang:7.3f}°, Error={ang_error:.2f}°"
        )

        if not (mag_pass and ang_pass):
            all_pass = False

    print()
    if all_pass:
        print("✅ ALL TESTS PASSED - GAP matches PGM reference!")
    else:
        print("❌ VALIDATION FAILED - Results differ from PGM reference")

else:
    print("❌ Solver did not converge")

print()
print("=" * 70)
