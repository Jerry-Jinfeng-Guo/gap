#!/usr/bin/env python3
"""
Generate test data for GAP solver validation.

This script:
1. Generates simple test networks using PGMGridGenerator
2. Saves network data to test_data/input.json
3. Generates load profile variations as test_data/update.json
4. Uses PGM to generate reference solutions for each scenario
5. Saves reference to test_data/output.json (base case)
6. Saves batch reference to test_data/batch_output.json (all scenarios)

This allows C++ batch calculation validation with known good reference data.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid_generators.pgm_generator import PGMGridGenerator
from reference_solutions.pgm_reference import PGM_AVAILABLE, PGMReferenceSolver


def generate_load_profile_scenarios(
    base_loads: dict, n_scenarios: int = 10, load_variation_range: tuple = (0.5, 1.5)
) -> list:
    """
    Generate load profile variations for batch calculation testing.

    Args:
        base_loads: Base load data from network generation
        n_scenarios: Number of load scenarios to generate
        load_variation_range: (min_factor, max_factor) for load scaling

    Returns:
        List of load update dictionaries for each scenario
    """
    scenarios = []
    load_factors = np.linspace(
        load_variation_range[0], load_variation_range[1], n_scenarios
    )

    for i, factor in enumerate(load_factors):
        scenario = {
            "scenario_id": i,
            "load_factor": float(factor),
            "description": f"Load profile at {factor*100:.1f}% of base case",
            "sym_load": [],
        }

        # Scale all loads by the factor
        for load_id, p_base, q_base in zip(
            base_loads["id"], base_loads["p_specified"], base_loads["q_specified"]
        ):
            scenario["sym_load"].append(
                {
                    "id": int(load_id),
                    "p_specified": float(p_base * factor),
                    "q_specified": float(q_base * factor),
                }
            )

        scenarios.append(scenario)

    return scenarios


def generate_radial_network_test(
    n_feeder: int, n_node_per_feeder: int, seed: int = 42, n_scenarios: int = 10
):
    """Generate a radial network test case with load profile scenarios."""
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

    # Generate load profile scenarios
    print(f"🔧 Generating {n_scenarios} load profile scenarios...")
    scenarios = generate_load_profile_scenarios(
        network_data["loads"],
        n_scenarios=n_scenarios,
        load_variation_range=(0.5, 1.5),  # 50% to 150% of base load
    )

    # Save update data in PGM-compatible batch format
    update_file = test_dir / "update.json"

    # Build arrays for PGM batch update format
    n_loads = len(scenarios[0]["sym_load"])
    load_ids = [load["id"] for load in scenarios[0]["sym_load"]]

    p_specified_matrix = []
    q_specified_matrix = []

    for scenario in scenarios:
        p_row = [load["p_specified"] for load in scenario["sym_load"]]
        q_row = [load["q_specified"] for load in scenario["sym_load"]]
        p_specified_matrix.append(p_row)
        q_specified_matrix.append(q_row)

    # PGM batch update format: id (1D), p_specified (2D), q_specified (2D)
    update_data = {
        "description": f"Load profile variations for {test_name}",
        "n_scenarios": n_scenarios,
        "load_variation_range": [0.5, 1.5],
        "sym_load": {
            "id": load_ids,
            "p_specified": p_specified_matrix,
            "q_specified": q_specified_matrix,
        },
        "scenarios_metadata": [
            {
                "scenario_id": i,
                "load_factor": scenarios[i]["load_factor"],
                "description": scenarios[i]["description"],
            }
            for i in range(n_scenarios)
        ],
    }

    with open(update_file, "w") as f:
        json.dump(update_data, f, indent=2)

    print(f"✅ Update data saved: {update_file}")
    print(f"   Scenarios: {n_scenarios} (50% to 150% load variation)")
    print(f"   Format: PGM-compatible batch update")

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
        "batch_scenarios": {
            "n_scenarios": n_scenarios,
            "load_variation_range": [0.5, 1.5],
            "description": "Load profiles from 50% to 150% of base case",
        },
        "notes": [
            "Automatically generated test case",
            "Loads are 150-200 kW per bus",
            "Power factor: 0.95 lagging",
            f"Includes {n_scenarios} load profile scenarios for batch validation",
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

    # Generate base case reference solution
    print("🧮 Generating PGM reference solution (base case)...")
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
            "base_case": {
                "converged": result["converged"],
                "calculation_time_s": result["calculation_time"],
                "iterations": result.get("n_iterations", None),
                "method": "newton_raphson",
                "tolerance": 1e-8,
                "max_iterations": 20,
            }
        }

        print(f"✅ Base case reference saved: {output_file}")
        print(f"   Converged: {result['converged']}")
        print(f"   Time: {result['calculation_time']:.4f}s")
        print()

    except Exception as e:
        print(f"❌ Failed to generate base case reference: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Generate batch reference solutions for all scenarios
    print(f"🧮 Generating PGM batch reference ({n_scenarios} scenarios)...")
    try:
        batch_start = time.time()

        # Run PGM batch calculation with update data
        batch_output_file = test_dir / "batch_output.json"

        # Load PGM input
        with open(input_file, "r") as f:
            pgm_input_json = json.load(f)

        # Prepare batch update in PGM format
        import power_grid_model as pgm

        # Convert input to PGM format
        data_to_convert = pgm_input_json.get("data", pgm_input_json)
        pgm_input = solver._convert_to_pgm_input(data_to_convert)

        # Create model
        model = pgm.PowerGridModel(pgm_input)

        # Prepare batch update data in PGM format
        # Format: id, p_specified, q_specified arrays with shape (n_scenarios, n_loads)
        n_loads = len(scenarios[0]["sym_load"])
        load_ids = [load["id"] for load in scenarios[0]["sym_load"]]

        # Build 2D arrays: rows = scenarios, cols = loads
        p_specified_matrix = np.zeros((n_scenarios, n_loads))
        q_specified_matrix = np.zeros((n_scenarios, n_loads))

        for i, scenario in enumerate(scenarios):
            for j, load in enumerate(scenario["sym_load"]):
                p_specified_matrix[i, j] = load["p_specified"]
                q_specified_matrix[i, j] = load["q_specified"]

        # Convert to PGM update format using the intermediate dict
        batch_update_dict = {
            "sym_load": {
                "id": load_ids,
                "p_specified": p_specified_matrix.tolist(),
                "q_specified": q_specified_matrix.tolist(),
            }
        }

        # Convert using PGM reference solver method
        batch_update = solver._convert_to_pgm_update(batch_update_dict)

        # Run batch power flow
        batch_results = model.calculate_power_flow(
            update_data=batch_update,
            calculation_method=pgm.CalculationMethod.newton_raphson,
            symmetric=True,
            error_tolerance=1e-8,
            max_iterations=20,
        )

        batch_time = time.time() - batch_start

        # Convert batch results to JSON-serializable format
        batch_output = {
            "description": f"Batch power flow results for {n_scenarios} load scenarios",
            "n_scenarios": n_scenarios,
            "total_calculation_time_s": batch_time,
            "avg_calculation_time_s": batch_time / n_scenarios,
            "scenarios": [],
        }

        for i in range(n_scenarios):
            scenario_result = {
                "scenario_id": i,
                "load_factor": scenarios[i]["load_factor"],
                "node": {
                    "id": batch_results["node"]["id"][i].tolist(),
                    "u_pu": batch_results["node"]["u_pu"][i].tolist(),
                    "u_angle": batch_results["node"]["u_angle"][i].tolist(),
                    "p": batch_results["node"]["p"][i].tolist(),
                    "q": batch_results["node"]["q"][i].tolist(),
                },
            }
            batch_output["scenarios"].append(scenario_result)

        # Save batch results
        with open(batch_output_file, "w") as f:
            json.dump(batch_output, f, indent=2)

        # Update metadata
        metadata["pgm_reference"]["batch"] = {
            "n_scenarios": n_scenarios,
            "total_time_s": batch_time,
            "avg_time_per_scenario_s": batch_time / n_scenarios,
            "speedup_vs_sequential": "N/A",  # Would need sequential timing
        }

        # Re-save metadata with batch results
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Batch reference saved: {batch_output_file}")
        print(f"   Scenarios: {n_scenarios}")
        print(f"   Total time: {batch_time:.4f}s")
        print(f"   Avg time per scenario: {batch_time/n_scenarios:.4f}s")
        print()

        return True

    except Exception as e:
        print(f"❌ Failed to generate batch reference: {e}")
        import traceback

        traceback.print_exc()

        # Still save metadata with base case only
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return False


if __name__ == "__main__":
    print("GAP Test Data Generator")
    print(
        "🎯 Purpose: Generate radial distribution network test cases with batch scenarios"
    )
    print("=" * 70)
    print()

    if not PGM_AVAILABLE:
        print("❌ Power Grid Model (PGM) is not available!")
        print("   Please install: pip install power-grid-model")
        sys.exit(1)

    # Define test case configurations
    # Format: (n_feeder, n_node_per_feeder, n_scenarios)
    test_configs = [
        (1, 2, 10),  # Small: 3 buses, 10 scenarios
        (1, 4, 10),  # Small: 5 buses, 10 scenarios
        (2, 4, 10),  # Medium: 9 buses, 10 scenarios
        (3, 4, 10),  # Medium: 13 buses, 10 scenarios
        (10, 10, 20),  # Large: 101 buses, 20 scenarios
    ]

    print(f"📋 Generating {len(test_configs)} test cases with batch scenarios:")
    for n_feeder, n_nodepf, n_scenarios in test_configs:
        test_name = f"radial_{n_feeder}feeder_{n_nodepf}nodepf"
        n_total = 1 + n_feeder * n_nodepf  # slack + feeder nodes
        print(f"   • {test_name:<30} ({n_total:4d} buses, {n_scenarios} scenarios)")
    print()

    # Generate all test cases
    success_count = 0
    failed_cases = []

    for n_feeder, n_nodepf, n_scenarios in test_configs:
        test_name = f"radial_{n_feeder}feeder_{n_nodepf}nodepf"
        try:
            if generate_radial_network_test(
                n_feeder, n_nodepf, seed=444, n_scenarios=n_scenarios
            ):
                success_count += 1
            else:
                failed_cases.append(test_name)
        except Exception as e:
            print(f"❌ Error generating {test_name}: {e}")
            import traceback

            traceback.print_exc()
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
    print("🔧 Generated files per test case:")
    print("   • input.json        - Network topology and base load")
    print("   • update.json       - Load profile scenarios for batch calculation")
    print("   • output.json       - PGM reference solution (base case)")
    print("   • batch_output.json - PGM batch reference (all scenarios)")
    print("   • metadata.json     - Test case metadata and PGM timing")
    print()
    print("🔧 Next steps:")
    print("   1. Run validation: python run_validation.py")
    print("   2. Test batch calc: Use update.json + batch_output.json in C++")
    print("   3. Compare timings: GAP batch vs PGM batch performance")
    print()

    sys.exit(0 if success_count == len(test_configs) else 1)
