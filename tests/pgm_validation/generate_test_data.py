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


def generate_radial_network_test(n_feeder: int, n_node_per_feeder: int, seed: int = 42):
    """Generate a radial network test case with specified parameters."""
    test_name = f"radial_{n_feeder}feeder_{n_node_per_feeder}nodepf"

    print(f"\n📝 Generating {test_name}")
    print("=" * 70)

    if not PGM_AVAILABLE:
        print("⚠️  PGM not available - skipping")
        return False

    # Create test data directory
    test_dir = Path(__file__).parent / "test_data" / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use grid generator
    print(
        f"🔧 Generating radial network: {n_feeder} feeder(s) × {n_node_per_feeder} nodes/feeder"
    )
    generator = PGMGridGenerator(seed=seed)

    network_data = generator.generate_symmetric_radial_grid(
        n_feeder=n_feeder,
        n_node_per_feeder=n_node_per_feeder,
        load_p_w_min=0.15e6,  # 150 kW min
        load_p_w_max=0.20e6,  # 200 kW max
        pf=0.95,
        n_step=1,
    )

    # Extract metadata before export
    n_nodes = network_data["metadata"]["n_node"]
    n_lines = network_data["metadata"]["n_line"]
    n_loads = len(network_data["loads"])
    total_load_mw = network_data["loads"]["p_specified"].abs().sum() / 1e6

    # Save to PGM format
    input_file = test_dir / "input.json"
    generator.export_to_pgm_format(network_data, test_dir)

    print(f"✅ Input saved: {input_file}")
    print(f"   Network: {n_nodes} nodes, {n_lines} lines")
    print(f"   Total load: {total_load_mw:.2f} MW")

    # Create metadata file
    metadata = {
        "name": f"Radial {n_feeder}-Feeder Network ({n_node_per_feeder} nodes/feeder)",
        "description": f"{n_nodes}-bus radial distribution feeder with {n_feeder} main branch(es)",
        "network_type": "distribution",
        "topology": "radial",
        "n_node": n_nodes,
        "n_line": n_lines,
        "n_load": n_loads,
        "n_feeder": n_feeder,
        "n_node_per_feeder": n_node_per_feeder,
        "voltage_level_kv": 10.0,
        "base_voltage": 10000.0,
        "frequency": 50.0,
        "total_load_mw": round(total_load_mw, 2),
        "base_power_va": 10000000.0,
        "seed": seed,
        "notes": [
            "Automatically generated test case",
            "Loads are 150-200 kW per bus",
            "Power factor: 0.95 lagging",
        ],
        "validation_criteria": {
            "voltage_magnitude_tolerance_pu": 0.0001,
            "convergence_required": True,
        },
    }

    metadata_file = test_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved: {metadata_file}")
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

        # Update metadata with PGM solver info
        metadata["pgm_reference"] = {
            "converged": result["converged"],
            "calculation_time_s": result["calculation_time"],
            "iterations": result.get("n_iterations", None),
            "method": "newton_raphson",
            "tolerance": 1e-8,
            "max_iterations": 20,
        }

        # Re-save metadata with PGM results
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

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
    print("🎯 Purpose: Generate radial distribution network test cases")
    print("=" * 70)
    print()

    if not PGM_AVAILABLE:
        print("❌ Power Grid Model (PGM) is not available!")
        print("   Please install: pip install power-grid-model")
        sys.exit(1)

    # Define test case configurations
    # Format: (n_feeder, n_node_per_feeder)
    test_configs = [
        (1, 2),  # radial_1feeder_2nodepf
        (1, 4),  # radial_1feeder_4nodepf
        (1, 8),  # radial_1feeder_8nodepf
        (2, 2),  # radial_2feeder_2nodepf
        (2, 4),  # radial_2feeder_4nodepf
        (2, 8),  # radial_2feeder_8nodepf
        (3, 2),  # radial_3feeder_2nodepf
        (3, 4),  # radial_3feeder_4nodepf
        (3, 8),  # radial_3feeder_8nodepf
        (10, 10),  # radial_10feeder_10nodepf (101 buses)
        (25, 50),  # radial_25feeder_50nodepf (1251 buses - extreme large case)
    ]

    print(f"📋 Generating {len(test_configs)} test cases:")
    for n_feeder, n_nodepf in test_configs:
        test_name = f"radial_{n_feeder}feeder_{n_nodepf}nodepf"
        n_total = 1 + n_feeder * n_nodepf  # slack + feeder nodes
        print(f"   • {test_name:<30} ({n_total:2d} buses)")
    print()

    # Generate all test cases
    success_count = 0
    failed_cases = []

    for n_feeder, n_nodepf in test_configs:
        test_name = f"radial_{n_feeder}feeder_{n_nodepf}nodepf"
        try:
            if generate_radial_network_test(n_feeder, n_nodepf, seed=42):
                success_count += 1
            else:
                failed_cases.append(test_name)
        except Exception as e:
            print(f"❌ Error generating {test_name}: {e}")
            failed_cases.append(test_name)

    # Summary
    print("=" * 70)
    print(f"✅ Successfully generated {success_count}/{len(test_configs)} test cases")

    if failed_cases:
        print(f"\n❌ Failed cases:")
        for case in failed_cases:
            print(f"   • {case}")

    print()
    print("📁 Test data location: tests/pgm_validation/test_data/")
    print()
    print("🔧 Next steps:")
    print("   1. Run validation: python run_validation.py")
    print("   2. Run specific test: python validate.py radial_3feeder_4nodepf")
    print("   3. Check all tests: python run_validation.py --verbose")
    print()

    sys.exit(0 if success_count == len(test_configs) else 1)
