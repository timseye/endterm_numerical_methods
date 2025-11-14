"""
Visualization and Analysis Tools for Transportation Problem
Generates graphs, tables, and comparative analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import json
import seaborn as sns
from typing import Dict, List


class TransportationVisualizer:
    """Create visualizations for transportation problem solutions"""
    
    def __init__(self, solution_file: str):
        """Load solution from JSON file"""
        with open(solution_file, 'r') as f:
            self.data = json.load(f)
        
        self.problem_name = self.data['problem_name']
        self.cost_matrix = np.array(self.data['cost_matrix'])
        self.supply = np.array(self.data['supply'])
        self.demand = np.array(self.data['demand'])
        self.allocation = np.array(self.data['allocation'])
        self.total_cost = self.data['total_cost']
        
        self.m = len(self.supply)
        self.n = len(self.demand)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_cost_matrix_heatmap(self, save_path: str = None):
        """Visualize cost matrix as heatmap"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(self.cost_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(self.n))
        ax.set_yticks(np.arange(self.m))
        ax.set_xticklabels([f'D{j+1}' for j in range(self.n)])
        ax.set_yticklabels([f'S{i+1}' for i in range(self.m)])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transportation Cost', rotation=270, labelpad=20)
        
        # Annotate cells with costs
        for i in range(self.m):
            for j in range(self.n):
                text = ax.text(j, i, f'{self.cost_matrix[i][j]:.0f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'Cost Matrix Heatmap\n{self.problem_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Destinations', fontsize=12)
        ax.set_ylabel('Sources', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost matrix heatmap saved to {save_path}")
        
        return fig
    
    def plot_allocation_matrix(self, save_path: str = None):
        """Visualize allocation matrix"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create binary mask for allocated cells
        mask = self.allocation > 1e-9
        
        # Plot allocation amounts
        im = ax.imshow(self.allocation, cmap='Greens', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(self.n))
        ax.set_yticks(np.arange(self.m))
        ax.set_xticklabels([f'D{j+1}' for j in range(self.n)])
        ax.set_yticklabels([f'S{i+1}' for i in range(self.m)])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Allocation Amount', rotation=270, labelpad=20)
        
        # Annotate cells
        for i in range(self.m):
            for j in range(self.n):
                if self.allocation[i][j] > 1e-9:
                    text = ax.text(j, i, f'{self.allocation[i][j]:.0f}',
                                 ha="center", va="center", 
                                 color="black", fontweight='bold', fontsize=10)
                else:
                    text = ax.text(j, i, '-',
                                 ha="center", va="center", 
                                 color="gray", fontsize=8)
        
        ax.set_title(f'Optimal Allocation Matrix\n{self.problem_name}\nTotal Cost: {self.total_cost:.2f}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Destinations', fontsize=12)
        ax.set_ylabel('Sources', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Allocation matrix saved to {save_path}")
        
        return fig
    
    def plot_network_diagram(self, save_path: str = None):
        """Create network flow diagram"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Position sources on left, destinations on right
        source_y = np.linspace(0, 1, self.m)
        dest_y = np.linspace(0, 1, self.n)
        
        source_x = 0.2
        dest_x = 0.8
        
        # Draw sources
        for i, y in enumerate(source_y):
            circle = plt.Circle((source_x, y), 0.03, color='lightblue', ec='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(source_x - 0.08, y, f'S{i+1}\n({self.supply[i]:.0f})', 
                   ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Draw destinations
        for j, y in enumerate(dest_y):
            circle = plt.Circle((dest_x, y), 0.03, color='lightcoral', ec='red', linewidth=2)
            ax.add_patch(circle)
            ax.text(dest_x + 0.08, y, f'D{j+1}\n({self.demand[j]:.0f})', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Draw allocation arrows
        max_allocation = np.max(self.allocation)
        for i in range(self.m):
            for j in range(self.n):
                if self.allocation[i][j] > 1e-9:
                    # Line thickness proportional to allocation
                    linewidth = 0.5 + 4 * (self.allocation[i][j] / max_allocation)
                    
                    # Draw arrow
                    ax.annotate('', xy=(dest_x - 0.04, dest_y[j]), 
                               xytext=(source_x + 0.04, source_y[i]),
                               arrowprops=dict(arrowstyle='->', lw=linewidth, 
                                             color='green', alpha=0.6))
                    
                    # Add label
                    mid_x = (source_x + dest_x) / 2
                    mid_y = (source_y[i] + dest_y[j]) / 2
                    label = f'{self.allocation[i][j]:.0f}\n(${self.cost_matrix[i][j]:.0f})'
                    ax.text(mid_x, mid_y, label, ha='center', va='center',
                           fontsize=7, bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='yellow', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        ax.set_title(f'Transportation Network Flow Diagram\n{self.problem_name}\nTotal Cost: {self.total_cost:.2f}',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        source_patch = mpatches.Patch(color='lightblue', label='Sources (Supply)')
        dest_patch = mpatches.Patch(color='lightcoral', label='Destinations (Demand)')
        flow_patch = mpatches.Patch(color='green', label='Allocation Flow')
        ax.legend(handles=[source_patch, dest_patch, flow_patch], loc='upper center',
                 bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network diagram saved to {save_path}")
        
        return fig
    
    def plot_cost_breakdown(self, save_path: str = None):
        """Bar chart showing cost contribution from each source"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cost by source
        source_costs = []
        for i in range(self.m):
            cost = sum(self.allocation[i][j] * self.cost_matrix[i][j] 
                      for j in range(self.n))
            source_costs.append(cost)
        
        colors_source = plt.cm.Blues(np.linspace(0.4, 0.8, self.m))
        bars1 = ax1.bar([f'S{i+1}' for i in range(self.m)], source_costs, color=colors_source)
        ax1.set_title('Total Cost by Source', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Source', fontsize=10)
        ax1.set_ylabel('Total Cost ($)', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Cost by destination
        dest_costs = []
        for j in range(self.n):
            cost = sum(self.allocation[i][j] * self.cost_matrix[i][j] 
                      for i in range(self.m))
            dest_costs.append(cost)
        
        colors_dest = plt.cm.Reds(np.linspace(0.4, 0.8, self.n))
        bars2 = ax2.bar([f'D{j+1}' for j in range(self.n)], dest_costs, color=colors_dest)
        ax2.set_title('Total Cost by Destination', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Destination', fontsize=10)
        ax2.set_ylabel('Total Cost ($)', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom', fontsize=8)
        
        fig.suptitle(f'Cost Breakdown Analysis\n{self.problem_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost breakdown saved to {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, output_dir: str = 'figures'):
        """Generate all visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine prefix based on problem size
        prefix = 'small' if self.m <= 4 else 'large'
        
        print(f"\nGenerating visualizations for {self.problem_name}...")
        
        self.plot_cost_matrix_heatmap(f'{output_dir}/{prefix}_cost_heatmap.png')
        self.plot_allocation_matrix(f'{output_dir}/{prefix}_allocation.png')
        self.plot_network_diagram(f'{output_dir}/{prefix}_network.png')
        self.plot_cost_breakdown(f'{output_dir}/{prefix}_cost_breakdown.png')
        
        print(f"All visualizations generated in {output_dir}/\n")


def create_comparison_plot(small_comparison_file: str, large_comparison_file: str, 
                          save_path: str = 'figures/method_comparison.png'):
    """Create comparison plot of different methods"""
    # Load comparison data
    with open(small_comparison_file, 'r') as f:
        small_comp = json.load(f)
    
    with open(large_comparison_file, 'r') as f:
        large_comp = json.load(f)
    
    # Extract costs
    methods = ['VAM-MODI', 'SciPy', 'PuLP']
    
    small_costs = [
        small_comp['custom_vam_modi']['cost'],
        small_comp['scipy']['cost'],
        small_comp['pulp']['cost']
    ]
    
    large_costs = [
        large_comp['custom_vam_modi']['cost'],
        large_comp['scipy']['cost'],
        large_comp['pulp']['cost']
    ]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Small problem
    bars1 = ax1.bar(x, small_costs, width, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_title('Small Problem (3×4)\nMethod Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Cost ($)', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Large problem
    bars2 = ax2.bar(x, large_costs, width, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_title('Large Problem (6×8)\nMethod Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Cost ($)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Transportation Problem: Method Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Method comparison plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VISUALIZATION GENERATOR")
    print("="*70)
    
    try:
        # Generate visualizations for small problem
        print("\n--- Small Problem Visualizations ---")
        viz_small = TransportationVisualizer('solution_small.json')
        viz_small.generate_all_visualizations()
        
        # Generate visualizations for large problem
        print("\n--- Large Problem Visualizations ---")
        viz_large = TransportationVisualizer('solution_large.json')
        viz_large.generate_all_visualizations()
        
        # Create method comparison plot
        print("\n--- Method Comparison Plot ---")
        create_comparison_plot(
            'solution_small_comparison.json',
            'solution_large_comparison.json'
        )
        
        print("\n" + "="*70)
        print("All visualizations generated successfully!")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the following in order:")
        print("1. transportation_solver.py")
        print("2. library_solver.py")
        print("3. visualizations.py (this file)")
