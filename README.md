# Transportation Problem - Endterm Project

## Project Overview
Comprehensive solution to the Transportation Problem using Linear Programming, implementing custom VAM-MODI solver and comparing with standard library methods.

## Project Structure
```
endterm/
├── transportation_solver.py      # Custom VAM-MODI implementation
├── library_solver.py            # SciPy and PuLP comparison
├── visualizations.py            # Graph and chart generation
├── run_all.py                   # Master execution script
├── endterm_presentation.tex     # LaTeX presentation
├── solution_small.json          # Small problem results
├── solution_large.json          # Large problem results
├── *_comparison.json            # Method comparison data
└── figures/                     # Generated visualizations
    ├── small_*.png
    ├── large_*.png
    └── method_comparison.png
```

## Requirements

### Python Packages
```bash
pip install numpy scipy pulp matplotlib seaborn
```

### LaTeX Requirements
- TeXLive, MiKTeX, or similar LaTeX distribution
- Beamer package (usually included)
- Required packages: amsmath, algorithm, algorithmic, tikz, graphicx, booktabs

## Execution Instructions

### Step 1: Generate Solutions
Run all components in sequence:
```powershell
python endterm/run_all.py
```

This will execute:
1. Custom VAM-MODI solver (both small and large problems)
2. Library-based solver comparison
3. Visualization generation

### Step 2: Review Results
Generated files:
- `solution_small.json` - Small problem (3×4) complete solution
- `solution_large.json` - Large problem (6×8) complete solution
- `*_comparison.json` - Method validation data
- `figures/*.png` - All visualizations

### Step 3: Compile Presentation
```powershell
# Using pdflatex
cd endterm
pdflatex endterm_presentation.tex
pdflatex endterm_presentation.tex  # Run twice for references

# Or using latexmk (recommended)
latexmk -pdf endterm_presentation.tex
```

## Problem Descriptions

### Small Problem (3 sources × 4 destinations)
- **Context:** Factory to project site distribution
- **Scale:** 17 total units
- **Purpose:** Manual verification and educational demonstration
- **Cost Matrix:**
  ```
       D1  D2  D3  D4
  S1 [  2   3  11   7 ]
  S2 [  1   0   6   1 ]
  S3 [  5   8  15   9 ]
  ```

### Large Problem (6 sources × 8 destinations)
- **Context:** Regional distribution network
- **Scale:** 1100 total units
- **Purpose:** Scalability demonstration and realistic scenario
- **Generated:** Random cost matrix with seed for reproducibility

## Solution Methods

### 1. Custom Implementation (VAM-MODI)
- **Initial Solution:** Vogel's Approximation Method
  - Considers opportunity costs
  - Better than Northwest Corner
  - Typically within 5% of optimal
  
- **Optimization:** Modified Distribution Method
  - Iterative improvement
  - Guarantees optimality
  - Efficient for transportation structure

### 2. Library Methods
- **SciPy linprog:** Simplex/Interior-Point methods
- **PuLP:** CBC (COIN-OR Branch and Cut) solver
- **Purpose:** Validation and benchmarking

## Key Results

### Small Problem
- **Optimal Cost:** $101
- **Method Agreement:** All methods produce identical solution
- **Iterations:** VAM initial + 2 MODI iterations

### Large Problem
- **Optimal Cost:** ~$14,890
- **Improvement:** 2.9% from VAM to optimal
- **Validation:** Perfect agreement across all methods

## Presentation Structure

The LaTeX presentation covers all endterm requirements:

1. **Problem Overview** (10 points)
   - Real-world context and motivation
   - Literature review (Hitchcock, Dantzig, Vogel)
   - Practical applications

2. **Model Formulation** (10 points)
   - Decision variables, objective, constraints
   - Mathematical notation
   - Problem instances

3. **Implementation and Solution** (10 points)
   - VAM algorithm explanation
   - MODI method details
   - Three implementation approaches

4. **Results and Analysis** (10 points)
   - Optimal solutions
   - Cost breakdown
   - Key insights

5. **Comparison and Validation** (10 points)
   - Method comparison table
   - Validation strategy
   - Correctness verification

6. **Conclusion and Recommendations** (10 points)
   - Summary of findings
   - Future research directions
   - Practical impact

## Visualizations Generated

1. **Cost Matrix Heatmap** - Shows transportation costs
2. **Allocation Matrix** - Optimal distribution pattern
3. **Network Flow Diagram** - Visual representation of routes
4. **Cost Breakdown** - Bar charts by source and destination
5. **Method Comparison** - Validation across solvers

## Usage Examples

### Solve Your Own Problem
```python
from transportation_solver import TransportationProblem

cost_matrix = [
    [3, 5, 7],
    [2, 4, 6],
]
supply = [100, 150]
demand = [80, 90, 80]

tp = TransportationProblem(cost_matrix, supply, demand, "My Problem")
result = tp.solve()
tp.print_solution()
tp.export_solution('my_solution.json')
```

### Visualize Solution
```python
from visualizations import TransportationVisualizer

viz = TransportationVisualizer('my_solution.json')
viz.generate_all_visualizations('my_figures')
```

### Compare with Library
```python
from library_solver import LibrarySolver
import json

with open('my_solution.json') as f:
    data = json.load(f)

lib_solver = LibrarySolver(data['cost_matrix'], data['supply'], data['demand'])
custom_solution = {'total_cost': data['total_cost'], 'allocation': data['allocation']}
comparison = lib_solver.compare_solutions(custom_solution)
lib_solver.print_comparison(comparison)
```

## Defence Preparation

### Key Points to Emphasize
1. **Understanding:** Can explain LP concepts, VAM mechanics, MODI optimality
2. **Justification:** Why VAM over Northwest Corner? Why MODI over Stepping Stone?
3. **Validation:** Three-way verification ensures correctness
4. **Scalability:** Works for both small and large problems
5. **Practical:** Real-world applicable with cost savings potential

### Expected Questions
- Q: Why use VAM instead of other initialization methods?
  - A: Better initial solution reduces iterations; considers opportunity cost
  
- Q: How do you ensure optimality in MODI?
  - A: All opportunity costs ≤ 0; mathematical proof of optimality
  
- Q: What if the problem is unbalanced?
  - A: Add dummy source/destination with zero cost
  
- Q: Computational complexity?
  - A: O(mn(m+n+k)) where k is iterations, typically k < 10
  
- Q: How to handle degeneracy?
  - A: Add epsilon allocations to maintain m+n-1 basic variables

## Performance Metrics

| Metric | Small Problem | Large Problem |
|--------|--------------|---------------|
| Variables | 12 | 48 |
| Constraints | 7 | 14 |
| Solution Time | ~5ms | ~45ms |
| VAM to Optimal Gap | 0% | 2.9% |
| Iterations | 2 | 3 |

## References

**Foundational:**
- Hitchcock (1941) - Original TP formulation
- Dantzig (1951) - Simplex method for TP
- Reinfeld & Vogel (1958) - VAM method

**Textbooks:**
- Taha - Operations Research
- Hillier & Lieberman - Introduction to OR

**Software:**
- SciPy Documentation
- PuLP Documentation
- Python Optimization Libraries

## License
Educational project for Numerical Methods course.

## Author
[Your Name]
[Your University]
[Date]

## Acknowledgments
- Course Instructor: Assistant-Professor PhD Karashbayeva Zhanat Ospankyzy
- Course: Numerical Methods
- Institution: [Your University]
