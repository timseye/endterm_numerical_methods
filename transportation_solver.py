"""
Comprehensive Transportation Problem Solver
Implements VAM (Vogel's Approximation Method) and MODI (Modified Distribution Method)
For Numerical Methods Endterm Project
"""

import numpy as np
import copy
from typing import List, Tuple, Dict
import json


class TransportationProblem:
    """
    Solves Transportation Problem using VAM for initial solution
    and MODI method for optimization
    """
    
    def __init__(self, cost_matrix: List[List[float]], 
                 supply: List[float], 
                 demand: List[float],
                 problem_name: str = "Transportation Problem"):
        """
        Initialize the transportation problem
        
        Args:
            cost_matrix: m x n matrix of transportation costs
            supply: List of m supply values (sources)
            demand: List of n demand values (destinations)
            problem_name: Description of the problem
        """
        self.original_cost = cost_matrix
        self.original_supply = supply
        self.original_demand = demand
        self.problem_name = problem_name
        
        # For tracking solution steps
        self.solution_steps = []
        self.iterations = []
        
        # Balance the problem if needed
        self.cost_matrix, self.supply, self.demand = self._balance_problem(
            cost_matrix, supply, demand
        )
        
        self.m = len(self.supply)  # number of sources
        self.n = len(self.demand)  # number of destinations
        self.allocation = np.zeros((self.m, self.n))
        
    def _balance_problem(self, cost_matrix, supply, demand):
        """Balance the transportation problem by adding dummy source/destination"""
        total_supply = sum(supply)
        total_demand = sum(demand)
        
        cost = [row[:] for row in cost_matrix]
        sup = supply[:]
        dem = demand[:]
        
        if total_supply > total_demand:
            # Add dummy destination
            diff = total_supply - total_demand
            dem.append(diff)
            for row in cost:
                row.append(0)
            self.solution_steps.append(f"Added dummy destination with demand {diff}")
        elif total_demand > total_supply:
            # Add dummy source
            diff = total_demand - total_supply
            sup.append(diff)
            cost.append([0] * len(dem))
            self.solution_steps.append(f"Added dummy source with supply {diff}")
        else:
            self.solution_steps.append("Problem is balanced")
            
        return cost, sup, dem
    
    def _calculate_penalty(self, costs: List[float]) -> float:
        """Calculate penalty (difference between two smallest costs)"""
        if len(costs) == 0:
            return 0
        if len(costs) == 1:
            return costs[0]
        
        sorted_costs = sorted(costs)
        return sorted_costs[1] - sorted_costs[0]
    
    def vogels_approximation_method(self) -> np.ndarray:
        """
        Vogel's Approximation Method (VAM) for initial feasible solution
        
        Returns:
            Allocation matrix
        """
        self.solution_steps.append("\n=== VOGEL'S APPROXIMATION METHOD (VAM) ===\n")
        
        # Working copies
        supply_left = self.supply[:]
        demand_left = self.demand[:]
        allocation = np.zeros((self.m, self.n))
        
        step = 1
        
        while max(supply_left) > 0 and max(demand_left) > 0:
            # Calculate row penalties
            row_penalties = []
            for i in range(self.m):
                if supply_left[i] > 0:
                    costs = [self.cost_matrix[i][j] for j in range(self.n) 
                            if demand_left[j] > 0]
                    row_penalties.append((self._calculate_penalty(costs), i, 'row'))
                else:
                    row_penalties.append((-1, i, 'row'))
            
            # Calculate column penalties
            col_penalties = []
            for j in range(self.n):
                if demand_left[j] > 0:
                    costs = [self.cost_matrix[i][j] for i in range(self.m) 
                            if supply_left[i] > 0]
                    col_penalties.append((self._calculate_penalty(costs), j, 'col'))
                else:
                    col_penalties.append((-1, j, 'col'))
            
            # Find maximum penalty
            all_penalties = row_penalties + col_penalties
            all_penalties.sort(reverse=True)
            max_penalty, index, pen_type = all_penalties[0]
            
            if max_penalty == -1:
                break
            
            # Find minimum cost cell in the selected row/column
            if pen_type == 'row':
                i = index
                min_cost = float('inf')
                best_j = -1
                for j in range(self.n):
                    if demand_left[j] > 0 and self.cost_matrix[i][j] < min_cost:
                        min_cost = self.cost_matrix[i][j]
                        best_j = j
                j = best_j
            else:  # column
                j = index
                min_cost = float('inf')
                best_i = -1
                for i in range(self.m):
                    if supply_left[i] > 0 and self.cost_matrix[i][j] < min_cost:
                        min_cost = self.cost_matrix[i][j]
                        best_i = i
                i = best_i
            
            # Allocate
            allocated = min(supply_left[i], demand_left[j])
            allocation[i][j] = allocated
            supply_left[i] -= allocated
            demand_left[j] -= allocated
            
            self.solution_steps.append(
                f"Step {step}: Allocate {allocated} to cell ({i+1},{j+1}) "
                f"with cost {self.cost_matrix[i][j]}"
            )
            step += 1
        
        self.allocation = allocation
        total_cost = self.calculate_total_cost(allocation)
        self.solution_steps.append(f"\nInitial VAM Solution Cost: {total_cost}")
        
        return allocation
    
    def _find_loop(self, allocation: np.ndarray, start_i: int, start_j: int) -> List[Tuple[int, int]]:
        """Find a closed loop for MODI method starting from (start_i, start_j)"""
        # Get all allocated cells
        allocated_cells = set()
        for i in range(self.m):
            for j in range(self.n):
                if allocation[i][j] > 0 or (i == start_i and j == start_j):
                    allocated_cells.add((i, j))
        
        # Try to find loop using backtracking
        def find_path(current, visited, direction):
            if len(visited) > 2 and current == (start_i, start_j):
                return visited
            
            i, j = current
            
            # Alternate between horizontal and vertical moves
            if direction == 'horizontal':
                # Try all columns in same row
                for next_j in range(self.n):
                    if next_j != j and (i, next_j) in allocated_cells and (i, next_j) not in visited:
                        result = find_path((i, next_j), visited + [(i, next_j)], 'vertical')
                        if result:
                            return result
            else:  # vertical
                # Try all rows in same column
                for next_i in range(self.m):
                    if next_i != i and (next_i, j) in allocated_cells and (next_i, j) not in visited:
                        result = find_path((next_i, j), visited + [(next_i, j)], 'horizontal')
                        if result:
                            return result
            
            return None
        
        # Start with horizontal move
        path = find_path((start_i, start_j), [(start_i, start_j)], 'horizontal')
        return path if path else []
    
    def modi_method(self, max_iterations: int = 100) -> np.ndarray:
        """
        Modified Distribution (MODI) Method for optimization
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimized allocation matrix
        """
        self.solution_steps.append("\n=== MODIFIED DISTRIBUTION (MODI) METHOD ===\n")
        
        allocation = self.allocation.copy()
        
        for iteration in range(max_iterations):
            # Check if solution is degenerate
            num_allocations = np.count_nonzero(allocation)
            expected = self.m + self.n - 1
            
            if num_allocations < expected:
                # Add epsilon allocations to make non-degenerate
                epsilon = 1e-10
                for i in range(self.m):
                    for j in range(self.n):
                        if allocation[i][j] == 0:
                            allocation[i][j] = epsilon
                            num_allocations += 1
                            if num_allocations >= expected:
                                break
                    if num_allocations >= expected:
                        break
            
            # Calculate u_i and v_j values
            u = [None] * self.m
            v = [None] * self.n
            u[0] = 0  # Set first u to 0
            
            # Iteratively calculate u and v using allocated cells
            changed = True
            iterations_calc = 0
            while changed and iterations_calc < 100:
                changed = False
                iterations_calc += 1
                for i in range(self.m):
                    for j in range(self.n):
                        if allocation[i][j] > 0:
                            if u[i] is not None and v[j] is None:
                                v[j] = self.cost_matrix[i][j] - u[i]
                                changed = True
                            elif v[j] is not None and u[i] is None:
                                u[i] = self.cost_matrix[i][j] - v[j]
                                changed = True
            
            # Calculate opportunity costs for unallocated cells
            opportunity_costs = {}
            max_opportunity = -float('inf')
            best_cell = None
            
            for i in range(self.m):
                for j in range(self.n):
                    if allocation[i][j] == 0 or allocation[i][j] < 1e-9:
                        if u[i] is not None and v[j] is not None:
                            opp_cost = u[i] + v[j] - self.cost_matrix[i][j]
                            opportunity_costs[(i, j)] = opp_cost
                            if opp_cost > max_opportunity:
                                max_opportunity = opp_cost
                                best_cell = (i, j)
            
            # Check optimality
            if max_opportunity <= 0:
                self.solution_steps.append(
                    f"Iteration {iteration + 1}: Optimal solution found! "
                    f"All opportunity costs ≤ 0"
                )
                break
            
            # Solution not optimal, improve it
            self.solution_steps.append(
                f"\nIteration {iteration + 1}: "
                f"Best cell {best_cell} with opportunity cost {max_opportunity:.2f}"
            )
            
            # Find loop and adjust allocation
            loop = self._find_loop(allocation, best_cell[0], best_cell[1])
            
            if not loop or len(loop) < 4:
                # If loop not found, solution is optimal
                self.solution_steps.append("Could not find improvement loop. Solution is optimal.")
                break
            
            # Find minimum allocation in negative positions of loop
            min_alloc = float('inf')
            for idx, (i, j) in enumerate(loop):
                if idx % 2 == 1:  # Negative positions
                    if allocation[i][j] < min_alloc:
                        min_alloc = allocation[i][j]
            
            # Adjust allocations along the loop
            for idx, (i, j) in enumerate(loop):
                if idx % 2 == 0:  # Positive positions
                    allocation[i][j] += min_alloc
                else:  # Negative positions
                    allocation[i][j] -= min_alloc
            
            # Store iteration info
            self.iterations.append({
                'iteration': iteration + 1,
                'allocation': allocation.copy(),
                'cost': self.calculate_total_cost(allocation)
            })
        
        self.allocation = allocation
        final_cost = self.calculate_total_cost(allocation)
        self.solution_steps.append(f"\n=== FINAL OPTIMAL SOLUTION ===")
        self.solution_steps.append(f"Total Cost: {final_cost}")
        
        return allocation
    
    def calculate_total_cost(self, allocation: np.ndarray = None) -> float:
        """Calculate total transportation cost"""
        if allocation is None:
            allocation = self.allocation
        
        total = 0
        for i in range(self.m):
            for j in range(self.n):
                total += allocation[i][j] * self.cost_matrix[i][j]
        return total
    
    def solve(self) -> Dict:
        """
        Complete solution: VAM + MODI
        
        Returns:
            Dictionary with solution details
        """
        # Step 1: Initial solution with VAM
        self.vogels_approximation_method()
        initial_cost = self.calculate_total_cost()
        
        # Step 2: Optimize with MODI
        self.modi_method()
        final_cost = self.calculate_total_cost()
        
        return {
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'allocation': self.allocation,
            'improvement': initial_cost - final_cost,
            'steps': self.solution_steps
        }
    
    def print_solution(self):
        """Print formatted solution"""
        print(f"\n{'='*70}")
        print(f"{self.problem_name}")
        print(f"{'='*70}")
        
        print("\nCost Matrix:")
        print("       ", end="")
        for j in range(self.n):
            print(f"  D{j+1:2}", end="")
        print("  Supply")
        
        for i in range(self.m):
            print(f"  S{i+1:2}  ", end="")
            for j in range(self.n):
                print(f" {self.cost_matrix[i][j]:4.0f}", end="")
            print(f"   {self.supply[i]:5.0f}")
        
        print("       ", end="")
        for j in range(self.n):
            print(f" {self.demand[j]:4.0f}", end="")
        print()
        
        print("\n" + "\n".join(self.solution_steps))
        
        print("\n\nFinal Allocation Matrix:")
        print("       ", end="")
        for j in range(self.n):
            print(f"  D{j+1:2}", end="")
        print("  Supply")
        
        for i in range(self.m):
            print(f"  S{i+1:2}  ", end="")
            for j in range(self.n):
                if self.allocation[i][j] > 1e-9:
                    print(f" {self.allocation[i][j]:4.0f}", end="")
                else:
                    print(f"    -", end="")
            print(f"   {self.supply[i]:5.0f}")
        
        print("       ", end="")
        for j in range(self.n):
            print(f" {self.demand[j]:4.0f}", end="")
        print()
        
        print(f"\nTotal Transportation Cost: {self.calculate_total_cost():.2f}")
        print(f"{'='*70}\n")
    
    def export_solution(self, filename: str):
        """Export solution to JSON file"""
        solution_data = {
            'problem_name': self.problem_name,
            'cost_matrix': self.cost_matrix,
            'supply': self.supply,
            'demand': self.demand,
            'allocation': self.allocation.tolist(),
            'total_cost': float(self.calculate_total_cost()),
            'steps': self.solution_steps,
            'iterations': self.iterations
        }
        
        with open(filename, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        print(f"Solution exported to {filename}")


if __name__ == "__main__":
    # Small problem for manual verification (3x4)
    print("\n" + "="*70)
    print("PROBLEM 1: Small Transportation Problem (3 sources × 4 destinations)")
    print("="*70)
    
    cost_matrix_small = [
        [2, 3, 11, 7],
        [1, 0, 6, 1],
        [5, 8, 15, 9]
    ]
    supply_small = [6, 1, 10]
    demand_small = [7, 5, 3, 2]
    
    tp_small = TransportationProblem(
        cost_matrix_small, 
        supply_small, 
        demand_small,
        "Small TP: Factory to Project Sites"
    )
    result_small = tp_small.solve()
    tp_small.print_solution()
    tp_small.export_solution('solution_small.json')
    
    # Large problem for demonstration (6x8)
    print("\n" + "="*70)
    print("PROBLEM 2: Large Transportation Problem (6 sources × 8 destinations)")
    print("="*70)
    
    np.random.seed(42)
    cost_matrix_large = np.random.randint(5, 50, size=(6, 8)).tolist()
    supply_large = [150, 200, 180, 220, 170, 180]
    demand_large = [130, 140, 120, 150, 160, 110, 140, 150]
    
    tp_large = TransportationProblem(
        cost_matrix_large,
        supply_large,
        demand_large,
        "Large TP: Regional Distribution Network"
    )
    result_large = tp_large.solve()
    tp_large.print_solution()
    tp_large.export_solution('solution_large.json')
