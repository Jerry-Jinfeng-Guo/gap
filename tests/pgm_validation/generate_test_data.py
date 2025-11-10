#!/usr/bin/env python3
"""
Generate test data for GAP solver validation.

This script:
1. Generates simple test networks using PGMGridGenerator
2. Saves network data to test_data/input.json
3. Uses PGM to generate reference solutions
4. Saves reference to test_data/output.json

This allows C++ debugging with known good reference data.
"""

import json
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid_generators.pgm_generator import PGMGridGenerator
from reference_solutions.pgm_reference import PGM_AVAILABLE, PGMReferenceSolver


def generate_simple_2bus_test():
    """Generate the simplest possible test case: 2-bus system."""
    print("📝 Generating Simple 2-Bus Test Case")
    print("=" * 50)

    # Create test data directory
    test_dir = Path(__file__).parent / "test_data" / "simple_2bus"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Manually create a simple 2-bus network in PGM format
    # This avoids any generator complexity

    base_power = 100e6  # 100 MVA
    base_voltage = 230000  # 230 kV

    # Create network manually
    network_data = {
        "version": "1.0",
        "type": "input",
        "is_batch": False,
        "attributes": {},
        "data": {
            "node": [
                {"id": 1, "u_rated": base_voltage},
                {"id": 2, "u_rated": base_voltage},
            ],
            "line": [
                {
                    "id": 3,
                    "from_node": 1,
                    "to_node": 2,
                    "from_status": 1,
                    "to_status": 1,
                    "r1": 0.01 * ((base_voltage**2) / base_power),  # 0.01 pu in ohms
                    "x1": 0.1 * ((base_voltage**2) / base_power),  # 0.1 pu in ohms
                    "c1": 0.0,
                    "tan1": 0.0,
                    "i_n": 1000.0,
                }
            ],
            "source": [
                {
                    "id": 4,
                    "node": 1,
                    "status": 1,
                    "u_ref": 1.05,
                    "u_ref_angle": 0.0,
                    "sk": 1e10,  # Very high short circuit power (stiff source)
                    "rx_ratio": 0.0,
                    "z01_ratio": 1.0,
                }
            ],
            "sym_load": [
                {
                    "id": 5,
                    "node": 2,
                    "status": 1,
                    "type": 0,  # const_power
                    "p_specified": 100e6,  # 100 MW
                    "q_specified": 50e6,  # 50 MVAR
                }
            ],
        },
    }

    # Save input file
    input_file = test_dir / "input.json"
    with open(input_file, "w") as f:
        json.dump(network_data, f, indent=2)

    print(f"✅ Input saved: {input_file}")
    print(f"   Network: 2 buses, 1 line")
    print(f"   Bus 1: Source, 1.05 pu")
    print(f"   Bus 2: Load, 100 MW + j50 MVAR")
    print(f"   Line: 0.01 + j0.1 pu")
    print()

    # Generate reference solution using PGM
    if PGM_AVAILABLE:
        print("🧮 Generating PGM reference solution...")
        try:
            solver = PGMReferenceSolver(calculation_method="newton_raphson")
            output_file = test_dir / "output.json"

            result = solver.generate_reference_solution(
                input_file=input_file,
                output_file=output_file,
                tolerance=1e-8,
                max_iterations=20,
            )

            print(f"✅ Reference saved: {output_file}")
            print(f"   Converged: {result['converged']}")
            print(f"   Time: {result['calculation_time']:.3f}s")
            print(f"   Nodes: {result['n_nodes']}")
            print()

            # Print voltage results from output file
            with open(output_file, "r") as f:
                output_data = json.load(f)

            print("📊 Reference Solution (PGM):")
            if "node" in output_data:
                for node in output_data["node"]:
                    u_pu = node["u_pu"]
                    u_angle = node["u_angle"]
                    print(f"   Node {node['id']}: {u_pu:.4f}∠{u_angle:.2f}° pu")

            return True

        except Exception as e:
            print(f"❌ Failed to generate reference: {e}")
            import traceback

            traceback.print_exc()
            return False
    else:
        print("⚠️  PGM not available - skipping reference generation")
        return False


def generate_radial_network_test():
    """Generate a small radial network test case."""
    print("\n📝 Generating Radial Network Test Case")
    print("=" * 50)

    if not PGM_AVAILABLE:
        print("⚠️  PGM not available - skipping")
        return False

    # Create test data directory
    test_dir = Path(__file__).parent / "test_data" / "radial_3feeder"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use grid generator
    print("🔧 Generating radial network with PGM grid generator...")
    generator = PGMGridGenerator(seed=42)

    network_data = generator.generate_symmetric_radial_grid(
        n_feeder=3,
        n_node_per_feeder=4,
        load_p_w_min=0.2e6,
        load_p_w_max=0.6e6,
        pf=0.95,
        n_step=1,
    )

    # Save to PGM format
    input_file = test_dir / "input.json"
    generator.export_to_pgm_format(network_data, test_dir)

    print(f"✅ Input saved: {input_file}")
    print(
        f"   Network: {network_data['metadata']['n_node']} nodes, "
        f"{network_data['metadata']['n_line']} lines"
    )
    print()

    # Generate reference solution
    print("🧮 Generating PGM reference solution...")
    try:
        solver = PGMReferenceSolver(calculation_method="newton_raphson")
        output_file = test_dir / "output.json"

        result = solver.generate_reference_solution(
            input_file=input_file,
            output_file=output_file,
            tolerance=1e-8,
            max_iterations=20,
        )

        print(f"✅ Reference saved: {output_file}")
        print(f"   Converged: {result['converged']}")
        print(f"   Time: {result['calculation_time']:.3f}s")
        print()

        return True

    except Exception as e:
        print(f"❌ Failed to generate reference: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("GAP Test Data Generator")
    print("🎯 Purpose: Generate test networks with PGM reference solutions")
    print("=" * 60)
    print()

    success_count = 0
    total_count = 2

    # Generate test cases
    if generate_simple_2bus_test():
        success_count += 1

    if generate_radial_network_test():
        success_count += 1

    # Summary
    print("=" * 60)
    print(f"✅ Generated {success_count}/{total_count} test cases")
    print()
    print("📁 Test data location: tests/pgm_validation/test_data/")
    print("   - simple_2bus/input.json & output.json")
    print("   - radial_3feeder/input.json & output.json")
    print()
    print("🔧 Next steps:")
    print("   1. Use these JSON files to test C++ solver directly")
    print(
        "   2. Debug with: ./build/gap_solver -i tests/pgm_validation/test_data/simple_2bus/input.json -o result.json"
    )
    print(
        "   3. Compare output with tests/pgm_validation/test_data/simple_2bus/output.json"
    )
