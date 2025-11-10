"""
GAP NRPF Validation Pipeline - Clean Version

Main orchestration script for validating GAP Newton-Raphson Power Flow solver
against Power Grid Model benchmark using systematic test generation and comparison.

This file provides the structure for the validation pipeline. The GAP solver integration
is left as a TODO since the actual GAP solver imports are not available.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import asdict
import warnings

# Import our PGM benchmark integration modules
import sys
sys.path.append(str(Path(__file__).parent))

from grid_generators.pgm_generator import PGMGridGenerator
from json_io.gap_json_parser import PGMJSONParser, GAPNetworkData
from json_io.gap_json_serializer import GAPJSONSerializer, GAPPowerFlowResults
from reference_solutions.pgm_reference import PGMReferenceSolver, PGM_AVAILABLE


class ValidationPipeline:
    """
    Complete validation pipeline for GAP NRPF solver.
    
    Orchestrates the entire validation workflow:
    1. Generate test networks using PGM tools
    2. Create reference solutions with PGM solver
    3. Run GAP NRPF solver on same networks (TODO: implement)
    4. Compare results with detailed analysis
    5. Generate validation reports
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        base_power: float = 1e6,
        tolerance_settings: Optional[Dict[str, float]] = None
    ):
        """
        Initialize validation pipeline.
        
        Args:
            workspace_dir: Directory for validation workspace
            base_power: Base power for per-unit calculations (VA)
            tolerance_settings: Custom tolerance settings for comparisons
        """
        self.workspace_dir = Path(workspace_dir)
        self.base_power = base_power
        
        # Default tolerance settings
        self.tolerance = tolerance_settings or {
            'voltage_pu': 1e-6,
            'angle_deg': 1e-4,
            'power_mw': 1e-3,
            'power_mvar': 1e-3
        }
        
        # Create workspace structure
        self._setup_workspace()
        
        # Initialize components
        self.grid_generator = PGMGridGenerator(seed=42)
        self.json_parser = PGMJSONParser(base_power=base_power)
        self.json_serializer = GAPJSONSerializer()
        
        if PGM_AVAILABLE:
            self.reference_solver = PGMReferenceSolver()
        else:
            self.reference_solver = None
            warnings.warn("PGM reference solver not available")
            
    def _setup_workspace(self):
        """Create validation workspace directory structure."""
        dirs = [
            'test_networks',
            'reference_solutions', 
            'gap_solutions',
            'comparison_results',
            'reports',
            'logs'
        ]
        
        for dir_name in dirs:
            (self.workspace_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def run_single_test_demo(
        self,
        config: Dict[str, Any],
        case_name: str = "demo_case"
    ) -> Dict[str, Any]:
        """
        Run a single test case demonstration.
        
        This method shows the complete validation workflow without requiring
        the actual GAP solver to be available.
        
        Args:
            config: Grid generation configuration
            case_name: Name for the test case
            
        Returns:
            Test case results
        """
        print(f"Running validation demo: {case_name}")
        
        # 1. Generate test network
        print("  1. Generating test network...")
        network_data = self.grid_generator.generate_symmetric_radial_grid(**config)
        
        # 2. Save network to JSON
        test_dir = self.workspace_dir / 'test_networks' / case_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        input_file = test_dir / 'input.json'
        self.grid_generator.export_to_pgm_format(network_data, test_dir)
        print(f"     ✓ Network saved: {network_data['metadata']['n_node']} nodes, "
              f"{network_data['metadata']['n_line']} lines")
        
        # 3. Generate reference solution using PGM
        if self.reference_solver:
            print("  2. Generating PGM reference solution...")
            reference_file = self.workspace_dir / 'reference_solutions' / f'{case_name}_reference.json'
            reference_result = self.reference_solver.generate_reference_solution(
                input_file=input_file,
                output_file=reference_file
            )
            print(f"     ✓ Reference generated: {reference_result['calculation_time']:.3f}s")
        else:
            print("  2. PGM reference solver not available")
            reference_result = None
            reference_file = None
        
        # 4. Parse network for GAP solver
        print("  3. Parsing network for GAP solver...")
        gap_network = self.json_parser.parse_network(input_file)
        Y = self.json_parser.create_admittance_matrix(gap_network)
        S = self.json_parser.create_power_injection_vector(gap_network)
        print(f"     ✓ Parsed: Y matrix {Y.shape} ({Y.nnz} non-zeros)")
        
        # 5. Run GAP NRPF solver (synthetic for demonstration)
        print("  4. Running GAP NRPF solver (synthetic demo)...")
        gap_file = self.workspace_dir / 'gap_solutions' / f'{case_name}_gap.json'
        gap_result = self._run_gap_solver_demo(gap_network, gap_file)
        print(f"     ✓ GAP solver: {gap_result.converged}, {gap_result.n_iterations} iterations")
        
        # 6. Compare results
        if reference_file and reference_file.exists():
            print("  5. Comparing results...")
            comparison_file = self.workspace_dir / 'comparison_results' / f'{case_name}_comparison.json'
            comparison_result = self.json_serializer.create_comparison_summary(
                gap_results=gap_result,
                reference_file=reference_file,
                output_file=comparison_file,
                tolerance=self.tolerance
            )
            print(f"     ✓ Comparison: max voltage error {comparison_result['node_comparison']['max_voltage_error_pu']:.2e} p.u.")
        else:
            print("  5. Comparison skipped (no reference)")
            comparison_result = None
        
        return {
            'network_metadata': network_data['metadata'],
            'reference_info': reference_result,
            'gap_info': {
                'converged': gap_result.converged,
                'n_iterations': gap_result.n_iterations,
                'calculation_time': gap_result.calculation_time
            },
            'comparison': comparison_result
        }
    
    def _run_gap_solver_demo(
        self,
        network: GAPNetworkData,
        output_file: Path
    ) -> GAPPowerFlowResults:
        """
        Demonstration GAP solver runner with synthetic results.
        
        TODO: Replace this with actual GAP NRPF solver call when available.
        
        Args:
            network: Parsed network data
            output_file: Output file for results
            
        Returns:
            Synthetic GAP power flow results
        """
        
        # Synthetic calculation time
        start_time = time.time()
        calculation_time = 0.003  # 3 milliseconds
        
        # Create synthetic voltage solution with realistic voltage profile
        u_pu = np.ones(network.n_node)
        u_angle_deg = np.zeros(network.n_node)
        
        # Simple voltage drop simulation for demonstration
        for i in range(network.n_node):
            if i == 0:  # Source node
                u_pu[i] = 1.05  # Source voltage
                u_angle_deg[i] = 0.0
            else:
                # Voltage drop based on distance from source
                u_pu[i] = 1.05 - i * 0.005  # 0.5% drop per node
                u_angle_deg[i] = -i * 0.5   # Small angle lag
        
        u_volt = u_pu * network.base_voltage
        
        # Synthetic branch results
        n_branch = network.n_branch
        p_from_mw = np.linspace(1.5, 0.1, n_branch)  # Decreasing power along feeders
        q_from_mvar = np.linspace(0.3, 0.05, n_branch)
        p_to_mw = -p_from_mw * 0.98  # Small losses
        q_to_mvar = -q_from_mvar * 0.97
        
        base_current = network.base_power / (np.sqrt(3) * network.base_voltage)
        i_from_a = np.abs(p_from_mw + 1j * q_from_mvar) * 1e6 / (np.sqrt(3) * u_volt[network.from_node[:n_branch]])
        i_to_a = i_from_a * 0.99
        loading = i_from_a / base_current / 2.0  # Assume 2 p.u. current rating
        
        # Load and source results
        p_load_mw = network.p_load_pu * network.base_power / 1e6
        q_load_mvar = network.q_load_pu * network.base_power / 1e6
        
        total_load_p = np.sum(p_load_mw)
        total_load_q = np.sum(q_load_mvar)
        p_source_mw = np.array([total_load_p * 1.03])  # 3% losses
        q_source_mvar = np.array([total_load_q * 1.05])  # 5% reactive losses
        
        # Create GAP results
        gap_results = GAPPowerFlowResults(
            # Convergence
            converged=True,
            n_iterations=5,
            max_mismatch=1.2e-9,
            calculation_time=calculation_time,
            
            # Node results
            node_ids=network.node_ids,
            u_pu=u_pu,
            u_angle_deg=u_angle_deg,
            u_volt=u_volt,
            
            # Branch results
            branch_ids=network.branch_ids,
            p_from_mw=p_from_mw,
            q_from_mvar=q_from_mvar,
            p_to_mw=p_to_mw,
            q_to_mvar=q_to_mvar,
            i_from_a=i_from_a,
            i_to_a=i_to_a,
            loading=loading,
            
            # Load results
            load_ids=network.load_ids,
            p_load_mw=p_load_mw,
            q_load_mvar=q_load_mvar,
            
            # Source results
            source_ids=network.source_ids,
            p_source_mw=p_source_mw,
            q_source_mvar=q_source_mvar,
            
            # Base values
            base_power=network.base_power,
            base_voltage=network.base_voltage
        )
        
        # Serialize results
        self.json_serializer.serialize_results(gap_results, output_file)
        
        return gap_results


# Demonstration function
def run_validation_demo(
    workspace_dir: Path = Path("validation_demo_workspace")
) -> None:
    """
    Run a simple validation demonstration.
    
    This shows the complete validation workflow working with synthetic GAP results.
    When the actual GAP solver is available, this framework can be used directly
    by replacing the synthetic solver with real GAP NRPF calls.
    """
    
    print("GAP NRPF Validation Pipeline Demonstration")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            'n_feeder': 3,
            'n_node_per_feeder': 4,
            'load_p_w_min': 0.2e6,
            'load_p_w_max': 0.6e6,
            'pf': 0.95,
            'n_step': 1
        },
        {
            'n_feeder': 5,
            'n_node_per_feeder': 6,
            'load_p_w_min': 0.3e6,
            'load_p_w_max': 0.8e6,
            'pf': 0.92,
            'n_step': 1
        }
    ]
    
    # Initialize pipeline
    pipeline = ValidationPipeline(workspace_dir)
    
    # Run test cases
    results = []
    for i, config in enumerate(test_configs):
        case_name = f"demo_case_{i+1:02d}"
        result = pipeline.run_single_test_demo(config, case_name)
        results.append(result)
        print()
    
    print("=" * 50)
    print("VALIDATION DEMONSTRATION COMPLETE")
    print("=" * 50)
    
    # Summary
    for i, result in enumerate(results):
        print(f"Test Case {i+1}:")
        print(f"  Network: {result['network_metadata']['n_node']} nodes")
        print(f"  GAP Converged: {result['gap_info']['converged']}")
        print(f"  GAP Time: {result['gap_info']['calculation_time']:.3f}s")
        if result['comparison']:
            print(f"  Max Voltage Error: {result['comparison']['node_comparison']['max_voltage_error_pu']:.2e} p.u.")
        print()
    
    print(f"All files saved to: {workspace_dir.absolute()}")
    print("\nNext steps:")
    print("1. Replace synthetic GAP solver with actual GAP NRPF implementation")
    print("2. Run systematic validation across multiple network configurations")
    print("3. Analyze results and tune GAP solver parameters as needed")


if __name__ == "__main__":
    if not PGM_AVAILABLE:
        print("Power Grid Model not available. Install with: pip install power-grid-model")
        exit(1)
    
    print("Running GAP NRPF validation pipeline demonstration...")
    
    try:
        run_validation_demo()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()