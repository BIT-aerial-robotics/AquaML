#!/usr/bin/env python3
"""
AquaML Test Runner

This script provides a convenient way to run AquaML tests with various options.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(test_type="unit", verbose=False, coverage=False, parallel=False):
    """Run tests with specified options"""
    
    # Base command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Add test type marker
    if test_type != "all":
        cmd.extend(["-m", test_type])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=AquaML",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add colored output
    cmd.append("--color=yes")
    
    print(f"🚀 Running AquaML tests: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=test_dir.parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run AquaML tests")
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "legacy", "all"],
        default="unit",
        help="Type of tests to run (default: unit)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run tests quickly (exclude slow tests)"
    )
    
    args = parser.parse_args()
    
    # Adjust test type for quick mode
    if args.quick:
        test_type = f"{args.type} and not slow"
    else:
        test_type = args.type
    
    # Run tests
    success = run_tests(
        test_type=test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 