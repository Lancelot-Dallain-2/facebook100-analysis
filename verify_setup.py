"""
Setup Verification Script

Checks that all requirements are met before running the analysis.
"""

import os
import sys


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  [FAIL] Python {version.major}.{version.minor}.{version.micro} (need >=3.8)")
        return False


def check_dependencies():
    """Check required packages."""
    print("\nChecking dependencies...")
    
    required = [
        'numpy',
        'pandas',
        'scipy',
        'networkx',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n[WARN] Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_data_directory():
    """Check if data directory exists."""
    print("\nChecking data directory...")
    
    possible_paths = [
        "fb100/data",
        "../fb100/data",
        "../../fb100/data"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Count .gml files
            gml_files = [f for f in os.listdir(path) if f.endswith('.gml')]
            if gml_files:
                print(f"  [OK] Found data directory: {path}")
                print(f"  [OK] Contains {len(gml_files)} .gml files")
                return True
    
    print("  [FAIL] Data directory not found")
    print("  Expected: fb100/data/ with .gml network files")
    print("\n  Please ensure the Facebook100 dataset is in the correct location.")
    return False


def check_output_directories():
    """Check/create output directories."""
    print("\nChecking output directories...")
    
    dirs = [
        'results',
        'results/q2',
        'results/q3',
        'results/q4',
        'results/q5',
        'results/q6'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"  [OK] Created: {d}")
        else:
            print(f"  [OK] Exists: {d}")
    
    return True


def check_scripts():
    """Check that all question scripts exist."""
    print("\nChecking question scripts...")
    
    scripts = [
        'q2_analysis.py',
        'q3_assortativity.py',
        'q4_link_prediction.py',
        'q5_label_propagation.py',
        'q6_community.py',
        'main.py'
    ]
    
    missing = []
    
    for script in scripts:
        if os.path.exists(script):
            print(f"  [OK] {script}")
        else:
            print(f"  [FAIL] {script} (MISSING)")
            missing.append(script)
    
    if missing:
        print(f"\n[WARN] Missing scripts: {', '.join(missing)}")
        return False
    
    return True


def main():
    """Run all checks."""
    print("="*80)
    print(" "*25 + "SETUP VERIFICATION")
    print("="*80 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Directory", check_data_directory),
        ("Output Directories", check_output_directories),
        ("Question Scripts", check_scripts)
    ]
    
    results = {}
    
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80 + "\n")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "="*80)
    
    if all_passed:
        print("\n[OK] All checks passed! You can run the analysis with:")
        print("\n    python main.py --all")
        print("\nor individual questions with:")
        print("\n    python main.py --q2")
        print("\n" + "="*80 + "\n")
        return 0
    else:
        print("\n[FAIL] Some checks failed. Please fix the issues above.")
        print("="*80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
