"""
Transportation Problem Solver using Standard Libraries
Comparison with custom VAM-MODI implementation
"""

import numpy as np
from scipy.optimize import linprog
import pulp
import json
from typing import List, Dict


class LibrarySolver:
    """Solve Transportation Problem using standard optimization libraries"""
    
    def __init__(self, cost_matrix: List[List[float]], 
                 supply: List[float], 
                 demand: List[float]):
        self.cost_matrix = cost_matrix
        self.supply = supply
        self.demand = demand
        self.m = len(supply)
        self.n = len(demand)
    
    def solve_with_scipy(self) -> Dict:
        """
        Solve using SciPy's Linear Programming solver
        
        Formulation:
        minimize: sum(c_ij * x_ij)
        subject to: 
            sum_j(x_ij) = supply_i for all i
            sum_i(x_ij) = demand_j for all j
            x_ij >= 0
        """
        # Flatten cost matrix into 1D array
        c = np.array(self.cost_matrix).flatten()
        
        # Equality constraints: Ax_eq = b_eq
        # Supply constraints: sum over j for each i
        A_eq_supply = np.zeros((self.m, self.m * self.n))
        for i in range(self.m):
            for j in range(self.n):
                A_eq_supply[i][i * self.n + j] = 1
        
        # Demand constraints: sum over i for each j
        A_eq_demand = np.zeros((self.n, self.m * self.n))
        for j in range(self.n):
            for i in range(self.m):
                A_eq_demand[j][i * self.n + j] = 1
        
        # Combine constraints
        A_eq = np.vstack([A_eq_supply, A_eq_demand])
        b_eq = np.array(self.supply + self.demand)
        
        # Bounds: x >= 0
        bounds = [(0, None) for _ in range(self.m * self.n)]
        
        # Solve
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            allocation = result.x.reshape((self.m, self.n))
            return {
                'method': 'SciPy linprog',
                'status': 'optimal',
                'allocation': allocation,
                'total_cost': result.fun,
                'message': result.message
            }
        else:
            return {
                'method': 'SciPy linprog',
                'status': 'failed',
                'message': result.message
            }
    
    def solve_with_pulp(self) -> Dict:
        """
        Solve using PuLP optimization library
        More intuitive formulation
        """
        # Create problem
        prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for i in range(self.m):
            for j in range(self.n):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        
        # Objective function
        prob += pulp.lpSum([self.cost_matrix[i][j] * x[i, j] 
                           for i in range(self.m) 
                           for j in range(self.n)])
        
        # Supply constraints
        for i in range(self.m):
            prob += pulp.lpSum([x[i, j] for j in range(self.n)]) == self.supply[i], \
                    f"Supply_{i}"
        
        # Demand constraints
        for j in range(self.n):
            prob += pulp.lpSum([x[i, j] for i in range(self.m)]) == self.demand[j], \
                    f"Demand_{j}"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        allocation = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                allocation[i][j] = x[i, j].varValue
        
        return {
            'method': 'PuLP',
            'status': pulp.LpStatus[prob.status],
            'allocation': allocation,
            'total_cost': pulp.value(prob.objective),
            'solver': 'CBC'
        }
    
    def compare_solutions(self, custom_solution: Dict) -> Dict:
        """Compare library solutions with custom VAM-MODI solution"""
        scipy_result = self.solve_with_scipy()
        pulp_result = self.solve_with_pulp()
        
        comparison = {
            'custom_vam_modi': {
                'cost': custom_solution['total_cost'],
                'allocation': custom_solution['allocation'].tolist()
            },
            'scipy': {
                'cost': scipy_result['total_cost'],
                'allocation': scipy_result['allocation'].tolist(),
                'difference': abs(scipy_result['total_cost'] - custom_solution['total_cost'])
            },
            'pulp': {
                'cost': pulp_result['total_cost'],
                'allocation': pulp_result['allocation'].tolist(),
                'difference': abs(pulp_result['total_cost'] - custom_solution['total_cost'])
            }
        }
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Print formatted comparison"""
        print("\n" + "="*70)
        print("SOLUTION COMPARISON: Custom VAM-MODI vs Library Methods")
        print("="*70)
        
        print(f"\n{'Method':<20} {'Total Cost':<15} {'Difference from Custom':<25}")
        print("-"*70)
        
        custom_cost = comparison['custom_vam_modi']['cost']
        print(f"{'VAM-MODI (Custom)':<20} {custom_cost:<15.2f} {'(baseline)':<25}")
        
        scipy_cost = comparison['scipy']['cost']
        scipy_diff = comparison['scipy']['difference']
        print(f"{'SciPy linprog':<20} {scipy_cost:<15.2f} {scipy_diff:<25.6f}")
        
        pulp_cost = comparison['pulp']['cost']
        pulp_diff = comparison['pulp']['difference']
        print(f"{'PuLP (CBC)':<20} {pulp_cost:<15.2f} {pulp_diff:<25.6f}")
        
        print("\n" + "="*70)
        
        # Check if all solutions agree
        tolerance = 0.01
        if scipy_diff < tolerance and pulp_diff < tolerance:
            print("✓ All methods produce consistent results (within tolerance)")
        else:
            print("⚠ Solutions differ - further investigation needed")
        
        print("="*70 + "\n")


def load_and_compare(problem_file: str):
    """Load custom solution and compare with library methods"""
    with open(problem_file, 'r') as f:
        data = json.load(f)
    
    # Extract problem data
    cost_matrix = data['cost_matrix']
    supply = data['supply']
    demand = data['demand']
    
    # Create library solver
    lib_solver = LibrarySolver(cost_matrix, supply, demand)
    
    # Prepare custom solution data
    custom_solution = {
        'total_cost': data['total_cost'],
        'allocation': np.array(data['allocation'])
    }
    
    # Compare
    comparison = lib_solver.compare_solutions(custom_solution)
    lib_solver.print_comparison(comparison)
    
    # Save comparison
    output_file = problem_file.replace('.json', '_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to {output_file}")
    
    return comparison


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LIBRARY-BASED SOLVER COMPARISON")
    print("="*70)
    
    # Wait for custom solver to generate solutions first
    print("\nNote: Run transportation_solver.py first to generate problem instances")
    print("\nAttempting to load and compare existing solutions...\n")
    
    try:
        print("\n--- Small Problem Comparison ---")
        compare_small = load_and_compare('solution_small.json')
        
        print("\n--- Large Problem Comparison ---")
        compare_large = load_and_compare('solution_large.json')
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run transportation_solver.py first to generate solutions.")
