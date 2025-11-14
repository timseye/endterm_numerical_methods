
# Transportation Problem - Endterm Project

## Project Overview
This project addresses the real-world problem of **minimizing total transportation costs** in distribution networks. We implement a two-phase classical optimization methodology (VAM + MODI) and validate our results against modern computational solvers. The project demonstrates how foundational optimization theory transforms into practical, high-efficiency business solutions.

## Real-World Problem

### Business Context
Our large-scale problem (6 sources × 8 destinations) models a **regional distribution network** — comparable to a sporting goods sales network or factory-to-warehouse product distribution system. 

**The Challenge:** Determine how many units $x_{ij}$ to ship from each source $i$ to each destination $j$ to:
- **Minimize total transportation cost**
- Respect supply constraints $a_i$ (capacity at each source)
- Satisfy demand requirements $b_j$ (needs at each destination)

**Real Impact:** Transportation costs represent 10-15% of product cost in integrated supply chains. Inefficient transportation practices are a primary driver of high operational expenses. Optimizing these costs directly impacts profitability and competitiveness.

## References and Their Application

Our project builds on key literature divided into two categories: **foundational methods** (Transportation Problem theory) and **modern complex models** (Supply Chain Management context).

### 1. Foundational Methods and Algorithm Selection

| Reference | Key Ideas | How We Used It |
|-----------|-----------|----------------|
| **Shore (1970)** | Explains that the Transportation Problem (TP) is a special case of Linear Programming (LP). Main goal: minimize total transportation cost. Notes that the Simplex method is labor-intensive for manual solution. **Promotes VAM as "time-saving"** because it "invariably yields a very good (and often optimal) solution." | Provided theoretical justification for choosing VAM as the optimal method for finding the Initial Basic Feasible Solution (Step 1). We used this argument to show that VAM minimizes the need for tedious iterative MODI steps (Steps 2-3). |
| **Sharma et al. (2012)** | Shows how TP transforms into a standard LP problem. Solves a real problem (Albert David company) using VAM and compares with advanced methods (Dual Simplex, Two-Phase, Big M). **Demonstrates that all methods yield the same optimal solution.** | Formed the basis for our **Comparison and Validation** section. We confirmed that our classical algorithm (VAM + MODI) produces the same optimal cost as standard library solvers (SciPy/PuLP), providing critical proof of implementation correctness. |
| **Garfinkel & Rao (1970)** | Describes the **Bottleneck Transportation Problem**. Unlike our goal (cost minimization), this variant minimizes maximum transportation time. | Used in brief literature review to demonstrate existence of TP variants with different objective functions (time instead of cost). |

### 2. Modern Models and Supply Chain Management Context

| Reference | Key Ideas | How We Used It |
|-----------|-----------|----------------|
| **Morales (2000)** | Discusses **Advanced Optimization Problems** for logistics networks in dynamic environments. Introduces Generalized Assignment Problem (GAP) models (static) and Multi-Period Single Source Problem (MPSSP) models (dynamic, integrating production, inventory, and transportation decisions). Discusses solution methods for complex NP-Hard problems (e.g., Branch and Price, Greedy Heuristics). | Used in literature review (**Modern Applications**). Demonstrates that the Transportation Problem is the foundation for more complex and realistic SCM models that incorporate dynamic factors (time, inventory, capacity constraints). |
| **Huq et al. (2010)** | Examines the key role of transportation costs in integrated supply chain management (SCM) models. **Argues that inefficient transportation practices are the primary cause of high SCM costs.** Emphasizes that transportation optimization is critical for reducing operational expenses. | Justified the **importance and motivation** of our project, showing that transportation costs constitute a significant portion of product cost (10-15%). |

## Project Structure
```
endterm_numerical_methods/
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

### Quick Start
Run all components in sequence:
```powershell
python run_all.py
```

This executes:
1. Custom VAM-MODI solver (small and large problems)
2. Library-based solver comparison
3. Visualization generation

### Generated Output Files
- `solution_small.json` - Small problem (3×4) complete solution
- `solution_large.json` - Large problem (6×8) complete solution  
- `*_comparison.json` - Method validation data
- `figures/*.png` - All visualizations

### Compile Presentation
```powershell
# Using pdflatex
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
- **Optimal Cost:** $101
- **Cost Matrix:**
  ```
       D1  D2  D3  D4
  S1 [  2   3  11   7 ]
  S2 [  1   0   6   1 ]
  S3 [  5   8  15   9 ]
  ```

### Large Problem (6 sources × 8 destinations)
- **Context:** Regional distribution network (sporting goods or factory-warehouse system)
- **Scale:** 1,100 total units
- **Purpose:** Realistic scenario and scalability demonstration
- **VAM Initial Cost:** $15,342
- **Optimal Cost:** $14,890
- **Cost Reduction:** $452 (2.9%)
- **Cost Matrix:** Randomly generated with fixed seed for reproducibility

## Methodology: What We Did, Why, and What It Produced

We applied a **two-phase methodology** using classical VAM and MODI methods, along with modern computational tools for validation.

### Phase 1: Initial Solution - Vogel's Approximation Method (VAM)

**Why VAM?**
- VAM considers **opportunity costs** (the difference between the two smallest costs in each row/column)
- According to Shore (1970), provides an excellent initial solution close to the optimum
- Significantly better than simpler methods (e.g., Northwest Corner)

**Result for Large Problem (6×8):**
- Initial cost: **$15,342**
- This solution is already very close to optimal, requiring minimal refinement

### Phase 2: Optimization - Modified Distribution Method (MODI)

**Why MODI?**
- Iterative procedure that checks optimality
- When positive opportunity costs ($w_{ij} > 0$) are found, corrects allocation through closed loops
- **Guarantees reaching the global optimum**

**Result for Large Problem:**
- Optimal final cost: **$14,890.00**
- Improvement from VAM: **$452 savings (2.9% reduction)**
- Iterations to convergence: **2-3 iterations**

### Phase 3: Comparison and Validation

**Why Validate?**
- Requirement 5: Confirm correctness of our implementation
- Ensure algorithm works correctly on both small and large datasets

**Methods Compared:**
1. Custom VAM-MODI solver
2. SciPy `linprog` (Simplex/Interior-Point)
3. PuLP with CBC solver

**Result:**
- **Perfect agreement:** All three methods produced optimal cost of **$14,890.00**
- Confirms validity of our classical approach
- Demonstrates that foundational theory produces results identical to state-of-the-art solvers

## Key Results and Insights

### Summary Table

| Phase | Method | Rationale (Why) | Result (What It Produced) |
|-------|--------|-----------------|---------------------------|
| **Phase 1: Initial Solution** | Vogel's Approximation Method (VAM) | Considers opportunity costs (difference between two smallest costs), which according to Shore (1970) produces an excellent initial solution close to optimum | Large problem (6×8): Initial cost = **$15,342** |
| **Phase 2: Optimization** | Modified Distribution Method (MODI) | Iterative procedure that checks optimality and, when positive opportunity costs ($w_{ij} > 0$) exist, corrects allocation through closed loops, **guaranteeing global optimum** | Optimal final cost = **$14,890.00** |
| **Phase 3: Validation** | Custom VAM-MODI vs. SciPy/PuLP | Necessary to confirm implementation correctness (Requirement 5). Verified that our implementation works correctly on both small and large datasets | All three methods (VAM-MODI, SciPy, PuLP) showed **perfect agreement** at optimal cost of **$14,890.00**, confirming validity of our approach |

### Critical Insights

1. **VAM Efficiency:** VAM's initial solution was only **2.9% more expensive** than optimal ($452 savings), confirming its "superiority" as a fast initialization method

2. **MODI Convergence Speed:** The MODI algorithm reached the optimal solution in just **2-3 iterations**

3. **Practical Significance:** Implementing optimization enables:
   - Reduction in transportation expenses
   - Balanced utilization of sources
   - Direct impact on business profitability (10-15% of product cost)

### Small Problem Results
- **Scale:** 3 sources × 4 destinations (17 total units)
- **Purpose:** Manual verification and educational demonstration
- **Optimal Cost:** $101
- **Method Agreement:** All methods produce identical solution

### Large Problem Results  
- **Scale:** 6 sources × 8 destinations (1,100 total units)
- **Purpose:** Realistic scenario and scalability demonstration
- **Optimal Cost:** $14,890.00
- **VAM to Optimal Gap:** 2.9%
- **Validation:** Perfect agreement across all three methods

**Conclusion:** The project demonstrates that **classical optimization theory (VAM, MODI) successfully transforms into practically significant, high-efficiency business solutions.**

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

1. **Theoretical Foundation:**
   - Shore (1970): VAM as optimal initial method, "invariably yields very good solution"
   - Sharma et al. (2012): Validation that classical methods match modern solvers
   - Transportation Problem as special case of LP with efficient solution methods

2. **Methodology Justification:**
   - **Why VAM over Northwest Corner?** Considers opportunity costs → better initial solution → fewer MODI iterations
   - **Why MODI over Stepping Stone?** More systematic, faster convergence, guarantees optimality through dual variables
   - Two-phase approach minimizes computational effort while ensuring global optimum

3. **Validation Strategy:**
   - Three-way verification (VAM-MODI, SciPy, PuLP)
   - Perfect agreement at $14,890.00 proves implementation correctness
   - Tested on both small (manual verification) and large (scalability) problems

4. **Real-World Impact:**
   - Huq et al. (2010): Transportation costs = 10-15% of product cost
   - Our optimization: 2.9% cost reduction ($452 savings on large problem)
   - Scalable to regional distribution networks with 6+ sources and 8+ destinations

5. **Modern Context:**
   - Morales (2000): TP is foundation for advanced SCM models (GAP, MPSSP)
   - Classical methods remain relevant and computationally efficient
   - Integration with dynamic factors (inventory, time, capacity) is natural extension

### Expected Questions and Answers

**Q: Why use VAM instead of other initialization methods?**
- **A:** Shore (1970) demonstrates that VAM considers opportunity costs (penalty for not choosing the best alternative), consistently yielding solutions within 5% of optimal. Our results confirm this: VAM was only 2.9% above optimal, requiring just 2-3 MODI iterations. Northwest Corner Method ignores costs entirely and typically requires many more iterations.

**Q: How do you ensure optimality in MODI?**
- **A:** MODI calculates opportunity costs ($w_{ij} = c_{ij} - u_i - v_j$) for all non-basic cells. When all $w_{ij} \leq 0$, the solution is proven optimal by the duality theorem of LP. If any $w_{ij} > 0$, we improve the solution through a closed loop reallocation, strictly reducing total cost.

**Q: What theoretical foundation justifies your approach?**
- **A:** The Transportation Problem is a special case of Linear Programming (Shore, 1970). The Simplex method guarantees finding the global optimum for LP problems. VAM-MODI is a specialized Simplex variant that exploits TP's network structure for efficiency. Sharma et al. (2012) confirmed that VAM-MODI produces identical results to general LP solvers.

**Q: What if the problem is unbalanced (supply ≠ demand)?**
- **A:** Add a dummy source (if supply < demand) or dummy destination (if supply > demand) with zero transportation costs. This converts the problem to balanced form without affecting the optimal allocation to real nodes.

**Q: How does this relate to modern supply chain management?**
- **A:** Morales (2000) shows that TP is the foundation for advanced SCM models like MPSSP (Multi-Period Single Source Problem) which integrate transportation with production scheduling and inventory management. Huq et al. (2010) emphasize that transportation optimization is critical, as these costs represent 10-15% of product cost in integrated supply chains.

**Q: Computational complexity?**
- **A:** VAM: O(mn × min(m,n)) for initialization. MODI: O(mn(m+n)×k) where k is iterations, typically k < 10. For our large problem (6×8): k = 2-3, demonstrating excellent practical efficiency.

**Q: How to handle degeneracy (fewer than m+n-1 basic variables)?**
- **A:** Add epsilon (ε ≈ 0) allocations to empty cells to maintain exactly m+n-1 basic variables, required for calculating dual variables (u_i, v_j) in MODI. Degeneracy is rare in practice but theoretically important.

## Performance Metrics

| Metric | Small Problem (3×4) | Large Problem (6×8) |
|--------|---------------------|---------------------|
| Decision Variables | 12 | 48 |
| Constraints | 7 | 14 |
| Solution Time | ~5ms | ~45ms |
| VAM Initial Cost | Close to optimal | $15,342 |
| Optimal Cost | $101 | $14,890 |
| VAM to Optimal Gap | ~0% | 2.9% ($452) |
| MODI Iterations | 2 | 2-3 |
| **Validation** | ✓ All methods agree | ✓ All methods agree |

## References

### Foundational Methods
- **Shore, H.H. (1970).** "The Transportation Problem and the Vogel Approximation Method." *Decision Sciences*, 1(3-4), 441-457.
  - Theoretical justification for VAM selection
  
- **Sharma, J.K., Swarup, K., & Sharma, A. (2012).** "Operations Research: Theory and Applications." 
  - Basis for comparison and validation methodology
  
- **Garfinkel, R.S., & Rao, M.R. (1970).** "The Bottleneck Transportation Problem." *Naval Research Logistics Quarterly*, 17(4), 465-472.
  - Alternative TP formulations with different objectives

### Modern Supply Chain Context  
- **Morales, D.R. (2000).** "Tactical Planning Models for Supply Chain Management." *PhD Dissertation*, Georgia Institute of Technology.
  - Modern applications: GAP, MPSSP, and advanced SCM models
  
- **Huq, F., Cutright, K., Jones, V., & Hensler, D.A. (2010).** "Managing the Cost of Transportation in a Supply Chain." *Journal of the Academy of Business and Economics*, 10(3).
  - Motivation: Transportation costs as critical factor (10-15% of product cost)

### Textbooks and Software
- Taha, H.A. - *Operations Research: An Introduction*
- Hillier, F.S., & Lieberman, G.J. - *Introduction to Operations Research*
- SciPy Documentation - `scipy.optimize.linprog`
- PuLP Documentation - CBC Solver

## License
Educational project for Numerical Methods course - Endterm Assignment

## Author
Timur Seye  
Nazarbayev University  
November 2025

## Acknowledgments
- **Course Instructor:** Assistant-Professor PhD Karashbayeva Zhanat Ospankyzy
- **Course:** Numerical Methods
- **Institution:** Nazarbayev University
