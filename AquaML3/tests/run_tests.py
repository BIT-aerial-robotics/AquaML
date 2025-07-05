"""
Test runner script for AquaML data collectors.

This script provides convenient commands to run different types of tests
with appropriate configurations and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run AquaML data collector tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests only
  python run_tests.py --integration             # Run integration tests only
  python run_tests.py --performance             # Run performance tests only
  python run_tests.py --all                     # Run all tests
  python run_tests.py --fast                    # Run fast tests only
  python run_tests.py --coverage                # Run with coverage report
  python run_tests.py --verbose                 # Run with verbose output
        """
    )
    
    # Test selection arguments
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--fast', action='store_true',
                       help='Run fast tests only (excludes slow tests)')
    
    # Configuration arguments
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Run with verbose output')
    parser.add_argument('--parallel', '-n', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--output', choices=['normal', 'junit', 'html'],
                       default='normal', help='Output format')
    
    # Specific test arguments
    parser.add_argument('--test', type=str,
                       help='Run specific test (e.g., test_base_collector.py::TestBaseCollector::test_initialization)')
    parser.add_argument('--keyword', '-k', type=str,
                       help='Run tests matching keyword expression')
    
    args = parser.parse_args()
    
    # Set up base directory
    test_dir = Path(__file__).parent
    base_cmd = ['python', '-m', 'pytest']
    
    # Build command based on arguments
    if args.unit:
        base_cmd.extend([str(test_dir / 'unit')])
        description = "Unit Tests"
    elif args.integration:
        base_cmd.extend([str(test_dir / 'integration')])
        description = "Integration Tests"
    elif args.performance:
        base_cmd.extend([str(test_dir / 'performance')])
        description = "Performance Tests"
    elif args.all:
        base_cmd.extend([str(test_dir)])
        description = "All Tests"
    elif args.fast:
        base_cmd.extend([str(test_dir), '-m', 'not slow'])
        description = "Fast Tests"
    elif args.test:
        base_cmd.extend([str(test_dir / args.test)])
        description = f"Specific Test: {args.test}"
    else:
        # Default to unit tests
        base_cmd.extend([str(test_dir / 'unit')])
        description = "Unit Tests (default)"
    
    # Add configuration options
    if args.verbose:
        base_cmd.append('-v')
    
    if args.coverage:
        base_cmd.extend([
            '--cov=AquaML.data.collectors',
            '--cov-report=html',
            '--cov-report=term-missing'
        ])
    
    if args.parallel > 1:
        base_cmd.extend(['-n', str(args.parallel)])
    
    if args.keyword:
        base_cmd.extend(['-k', args.keyword])
    
    # Set output format
    if args.output == 'junit':
        base_cmd.extend(['--junit-xml=test-results.xml'])
    elif args.output == 'html':
        base_cmd.extend(['--html=test-report.html', '--self-contained-html'])
    
    # Add common options
    base_cmd.extend([
        '--tb=short',  # Short traceback format
        '--strict-markers',  # Strict marker validation
        '--disable-warnings'  # Disable warnings for cleaner output
    ])
    
    # Run the tests
    success = run_command(base_cmd, description)
    
    # Print summary
    if success:
        print(f"\n{'='*60}")
        print(f"✅ {description} completed successfully!")
        print(f"{'='*60}")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - HTML report: htmlcov/index.html")
            print("  - Terminal report shown above")
    else:
        print(f"\n{'='*60}")
        print(f"❌ {description} failed!")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == '__main__':
    main() 