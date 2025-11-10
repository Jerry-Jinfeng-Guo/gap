"""
PGM Reference Solution Generator

This module uses the actual Power Grid Model library to generate
reference solutions for validation against GAP NRPF solver.

Requires: pip install power-grid-model
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import warnings

try:
    import power_grid_model as pgm
    from power_grid_model.validation import validate_input_data, errors_to_string
    PGM_AVAILABLE = True
except ImportError:
    PGM_AVAILABLE = False
    warnings.warn(
        "Power Grid Model not available. Install with: pip install power-grid-model",
        UserWarning
    )


class PGMReferenceSolver:
    """
    Reference solution generator using Power Grid Model.
    
    Creates standardized reference solutions for benchmark validation
    using the proven PGM Newton-Raphson solver implementation.
    """
    
    def __init__(self, calculation_method: str = "newton_raphson"):
        """
        Initialize PGM reference solver.
        
        Args:
            calculation_method: PGM calculation method 
                               ("newton_raphson", "iterative_current", "linear")
        """
        if not PGM_AVAILABLE:
            raise ImportError("Power Grid Model not available")
            
        self.calculation_method = getattr(pgm.CalculationMethod, calculation_method)
        
    def generate_reference_solution(
        self,
        input_file: Path,
        output_file: Path,
        update_file: Optional[Path] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Generate reference solution using PGM solver.
        
        Args:
            input_file: Path to input.json file
            output_file: Path to save reference output.json
            update_file: Optional update data for time series
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Dictionary with solution info and metadata
        """
        # Load input data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
            
        # Convert to PGM format
        pgm_input = self._convert_to_pgm_input(input_data)
        
        # Validate input data
        validation_errors = validate_input_data(pgm_input)
        if validation_errors:
            raise ValueError(f"Input validation failed: {errors_to_string(validation_errors)}")
        
        # Create PGM model
        model = pgm.PowerGridModel(pgm_input)
        
        # Prepare calculation parameters
        calc_params = {
            'calculation_method': self.calculation_method,
            'symmetric': True,
            'error_tolerance': tolerance,
            'max_iterations': max_iterations
        }
        
        # Handle update data if provided
        update_data = None
        if update_file and update_file.exists():
            with open(update_file, 'r') as f:
                update_json = json.load(f)
            update_data = self._convert_to_pgm_update(update_json)
        
        # Run power flow calculation
        start_time = time.time()
        try:
            if update_data:
                # Batch calculation with updates
                result = model.calculate_power_flow(
                    update_data=update_data,
                    **calc_params
                )
                # Use first scenario if batch
                if isinstance(result, dict):
                    # Single scenario
                    pgm_result = result
                else:
                    # Batch result - take first
                    pgm_result = {key: value[0] for key, value in result.items()}
            else:
                # Single calculation
                pgm_result = model.calculate_power_flow(**calc_params)
                
        except Exception as e:
            raise RuntimeError(f"PGM calculation failed: {e}")
            
        calculation_time = time.time() - start_time
        
        # Debug: Check result structure
        print(f"PGM result type: {type(pgm_result)}")
        if hasattr(pgm_result, 'keys'):
            print(f"PGM result keys: {list(pgm_result.keys())}")
        elif isinstance(pgm_result, dict):
            print(f"PGM result keys: {list(pgm_result.keys())}")
        else:
            print(f"PGM result structure: {pgm_result}")
        
        # Convert results to standardized output format
        output_data = self._convert_pgm_output(pgm_result, calculation_time)
        
        # Save reference solution
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=self._json_serializer)
            
        # Return solution summary
        return {
            'converged': True,  # PGM throws exception if not converged
            'calculation_time': calculation_time,
            'n_nodes': len(pgm_result.get(pgm.ComponentType.node, [])),
            'n_lines': len(pgm_result.get(pgm.ComponentType.line, [])),
            'n_loads': len(pgm_result.get(pgm.ComponentType.sym_load, [])),
            'n_sources': len(pgm_result.get(pgm.ComponentType.source, [])),
            'solver': 'power_grid_model',
            'method': self.calculation_method.name,
            'output_file': str(output_file)
        }
    
    def _convert_to_pgm_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON input to PGM input format."""
        pgm_input = {}
        
        # Convert each component type
        if 'node' in input_data:
            pgm_input['node'] = self._convert_nodes(input_data['node'])
            
        if 'line' in input_data:
            pgm_input['line'] = self._convert_lines(input_data['line'])
            
        if 'sym_load' in input_data:
            pgm_input['sym_load'] = self._convert_loads(input_data['sym_load'])
            
        if 'source' in input_data:
            pgm_input['source'] = self._convert_sources(input_data['source'])
            
        return pgm_input
    
    def _convert_nodes(self, nodes_data: List[Dict]) -> np.ndarray:
        """Convert node data to PGM format."""
        n_node = len(nodes_data)
        nodes = pgm.initialize_array("input", "node", n_node)
        
        for i, node in enumerate(nodes_data):
            nodes['id'][i] = node['id']
            nodes['u_rated'][i] = node['u_rated']
            
        return nodes
    
    def _convert_lines(self, lines_data: List[Dict]) -> np.ndarray:
        """Convert line data to PGM format."""
        n_line = len(lines_data)
        lines = pgm.initialize_array("input", "line", n_line)
        
        for i, line in enumerate(lines_data):
            lines['id'][i] = line['id']
            lines['from_node'][i] = line['from_node']
            lines['to_node'][i] = line['to_node']
            lines['from_status'][i] = line.get('from_status', 1)
            lines['to_status'][i] = line.get('to_status', 1)
            lines['r1'][i] = line['r1']
            lines['x1'][i] = line['x1']
            lines['c1'][i] = line.get('c1', 0.0)
            lines['tan1'][i] = line.get('tan1', 0.0)
            lines['i_n'][i] = line.get('i_n', 1000.0)  # Default rating
            
        return lines
    
    def _convert_loads(self, loads_data: List[Dict]) -> np.ndarray:
        """Convert load data to PGM format."""
        n_load = len(loads_data)
        loads = pgm.initialize_array("input", "sym_load", n_load)
        
        for i, load in enumerate(loads_data):
            loads['id'][i] = load['id']
            loads['node'][i] = load['node']
            loads['status'][i] = load.get('status', 1)
            loads['type'][i] = getattr(pgm.LoadGenType, load.get('type', 'const_power'))
            loads['p_specified'][i] = load['p_specified']
            loads['q_specified'][i] = load['q_specified']
            
        return loads
    
    def _convert_sources(self, sources_data: List[Dict]) -> np.ndarray:
        """Convert source data to PGM format."""
        n_source = len(sources_data)
        sources = pgm.initialize_array("input", "source", n_source)
        
        for i, source in enumerate(sources_data):
            sources['id'][i] = source['id']
            sources['node'][i] = source['node']
            sources['status'][i] = source.get('status', 1)
            sources['u_ref'][i] = source['u_ref']
            sources['sk'][i] = source.get('sk', 1e20)
            sources['rx_ratio'][i] = source.get('rx_ratio', 0.1)
            sources['z01_ratio'][i] = source.get('z01_ratio', 1.0)
            
        return sources
    
    def _convert_to_pgm_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert update data to PGM format."""
        pgm_update = {}
        
        if 'sym_load' in update_data:
            load_update_data = update_data['sym_load']
            n_load = len(load_update_data['id'])
            
            # Handle both single and time series data
            p_specified = np.array(load_update_data['p_specified'])
            q_specified = np.array(load_update_data['q_specified'])
            
            if p_specified.ndim == 1:
                # Single scenario
                n_scenario = 1
                p_specified = p_specified.reshape(1, -1)
                q_specified = q_specified.reshape(1, -1)
            else:
                # Time series
                n_scenario = p_specified.shape[0]
            
            # Create PGM update array
            load_update = pgm.initialize_array("update", "sym_load", (n_scenario, n_load))
            
            for scenario in range(n_scenario):
                for i in range(n_load):
                    load_update['id'][scenario, i] = load_update_data['id'][i]
                    load_update['p_specified'][scenario, i] = p_specified[scenario, i]
                    load_update['q_specified'][scenario, i] = q_specified[scenario, i]
            
            pgm_update['sym_load'] = load_update
            
        return pgm_update
    
    def _convert_pgm_output(
        self, 
        pgm_result: Dict[pgm.ComponentType, np.ndarray],
        calculation_time: float
    ) -> Dict[str, Any]:
        """Convert PGM output to standardized JSON format."""
        output_data = {}
        
        # Convert node results
        if pgm.ComponentType.node in pgm_result:
            output_data['node'] = self._convert_node_output(pgm_result[pgm.ComponentType.node])
            
        # Convert line results
        if pgm.ComponentType.line in pgm_result:
            output_data['line'] = self._convert_line_output(pgm_result[pgm.ComponentType.line])
            
        # Convert load results
        if pgm.ComponentType.sym_load in pgm_result:
            output_data['sym_load'] = self._convert_load_output(pgm_result[pgm.ComponentType.sym_load])
            
        # Convert source results
        if pgm.ComponentType.source in pgm_result:
            output_data['source'] = self._convert_source_output(pgm_result[pgm.ComponentType.source])
            
        # Add metadata
        output_data['meta'] = {
            'solver': 'power_grid_model',
            'calculation_method': self.calculation_method.name,
            'calculation_time_s': calculation_time,
            'symmetric': True,
            'converged': True,  # PGM throws if not converged
            'base_power_va': 1.0,  # PGM uses 1 VA base internally
            'timestamp': time.time()
        }
        
        return output_data
    
    def _convert_node_output(self, node_result: np.ndarray) -> List[Dict[str, Any]]:
        """Convert PGM node results to JSON format."""
        results = []
        for i in range(len(node_result)):
            # Check for optional fields
            p = node_result['p'][i] if 'p' in node_result.dtype.names else 0.0
            q = node_result['q'][i] if 'q' in node_result.dtype.names else 0.0
            
            results.append({
                'id': int(node_result['id'][i]),
                'energized': bool(node_result['energized'][i]),
                'u_pu': float(node_result['u_pu'][i]),
                'u_angle': float(np.degrees(node_result['u_angle'][i])),  # Convert to degrees
                'u': float(node_result['u'][i]),
                'p': float(p),
                'q': float(q)
            })
        return results
    
    def _convert_line_output(self, line_result: np.ndarray) -> List[Dict[str, Any]]:
        """Convert PGM line results to JSON format."""
        results = []
        for i in range(len(line_result)):
            results.append({
                'id': int(line_result['id'][i]),
                'energized': bool(line_result['energized'][i]),
                'loading': float(line_result['loading'][i]),
                'p_from': float(line_result['p_from'][i]),
                'q_from': float(line_result['q_from'][i]),
                'i_from': float(line_result['i_from'][i]),
                's_from': float(line_result['s_from'][i]),
                'p_to': float(line_result['p_to'][i]),
                'q_to': float(line_result['q_to'][i]),
                'i_to': float(line_result['i_to'][i]),
                's_to': float(line_result['s_to'][i])
            })
        return results
    
    def _convert_load_output(self, load_result: np.ndarray) -> List[Dict[str, Any]]:
        """Convert PGM load results to JSON format."""
        results = []
        for i in range(len(load_result)):
            p = float(load_result['p'][i])
            q = float(load_result['q'][i])
            s = np.sqrt(p**2 + q**2)
            pf = p / max(s, 1e-12)  # Avoid division by zero
            
            results.append({
                'id': int(load_result['id'][i]),
                'energized': bool(load_result['energized'][i]),
                'p': p,
                'q': q,
                's': float(s),
                'pf': float(pf)
            })
        return results
    
    def _convert_source_output(self, source_result: np.ndarray) -> List[Dict[str, Any]]:
        """Convert PGM source results to JSON format."""
        results = []
        for i in range(len(source_result)):
            p = float(source_result['p'][i])
            q = float(source_result['q'][i])
            s = np.sqrt(p**2 + q**2)
            pf = p / max(s, 1e-12)  # Avoid division by zero
            
            # Check for optional fields
            current = source_result['i'][i] if 'i' in source_result.dtype.names else 0.0
            
            results.append({
                'id': int(source_result['id'][i]),
                'energized': bool(source_result['energized'][i]),
                'p': p,
                'q': q,
                's': float(s),
                'pf': float(pf),
                'i': float(current)
            })
        return results
    
    def generate_test_suite_references(
        self,
        test_cases_dir: Path,
        output_dir: Path,
        pattern: str = "input*.json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate reference solutions for a suite of test cases.
        
        Args:
            test_cases_dir: Directory containing test case input files
            output_dir: Directory to save reference outputs
            pattern: File pattern to match input files
            
        Returns:
            Dictionary of results for each test case
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Find all input files
        input_files = list(test_cases_dir.glob(pattern))
        
        for input_file in input_files:
            test_name = input_file.stem
            output_file = output_dir / f"reference_{test_name}.json"
            
            # Check for corresponding update file
            update_file = input_file.parent / f"update_{test_name.replace('input_', '')}.json"
            if not update_file.exists():
                update_file = None
                
            try:
                result = self.generate_reference_solution(
                    input_file=input_file,
                    output_file=output_file,
                    update_file=update_file
                )
                results[test_name] = result
                print(f"✓ Generated reference for {test_name}")
                
            except Exception as e:
                results[test_name] = {
                    'error': str(e),
                    'failed': True
                }
                print(f"✗ Failed to generate reference for {test_name}: {e}")
        
        # Save summary
        summary_file = output_dir / "reference_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
            
        return results
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Example usage and testing
if __name__ == "__main__":
    if not PGM_AVAILABLE:
        print("Power Grid Model not available. Install with: pip install power-grid-model")
        exit(1)
    
    # Test reference generation
    solver = PGMReferenceSolver()
    
    # Example: generate reference for test case
    test_dir = Path("test_data")
    if (test_dir / "input.json").exists():
        try:
            result = solver.generate_reference_solution(
                input_file=test_dir / "input.json",
                output_file=test_dir / "reference_output.json"
            )
            print("Successfully generated reference solution:")
            print(f"  Solver: {result['solver']}")
            print(f"  Method: {result['method']}")
            print(f"  Calculation time: {result['calculation_time']:.3f}s")
            print(f"  Nodes: {result['n_nodes']}")
            print(f"  Lines: {result['n_lines']}")
            print(f"  Loads: {result['n_loads']}")
            print(f"  Sources: {result['n_sources']}")
            
        except Exception as e:
            print(f"Error generating reference: {e}")
    else:
        print("No test input file found. Generate test data first using PGM grid generator.")