"""
Master Script - Run All Components of the Endterm Project
Executes: VAM-MODI Solver → Library Comparison → Visualizations
"""

import subprocess
import sys
import os


def run_script(script_path: str, description: str):
    """Run a Python script and report results"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running {description}: {str(e)}")
        return False


def main():
    """Execute all project components in sequence"""
    print("\n" + "="*70)
    print("ENDTERM PROJECT - TRANSPORTATION PROBLEM")
    print("Comprehensive Solution Pipeline")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        (os.path.join(base_dir, "transportation_solver.py"), "Custom VAM-MODI Solver"),
        (os.path.join(base_dir, "library_solver.py"), "Library Method Comparison"),
        (os.path.join(base_dir, "visualizations.py"), "Visualization Generation")
    ]
    
    results = []
    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70 + "\n")
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status:12} - {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "="*70)
        print("ALL COMPONENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  • endterm/solution_small.json")
        print("  • endterm/solution_large.json")
        print("  • endterm/solution_small_comparison.json")
        print("  • endterm/solution_large_comparison.json")
        print("  • endterm/figures/*.png (all visualizations)")
        print("\nNext steps:")
        print("  1. Review generated solutions and visualizations")
        print("  2. Compile the LaTeX presentation (endterm_presentation.tex)")
        print("  3. Prepare for project defence")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("SOME COMPONENTS FAILED")
        print("Please check error messages above")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
