#!/usr/bin/env python3
"""
Ramanujan Test Runner

Simple test runner script for the Ramanujan ML framework tests.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from test_driver import RamanujanTestDriver


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Run Ramanujan ML Framework Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests with default settings
  python run_tests.py --work-dir /tmp   # Use custom work directory
  python run_tests.py --verbose         # Verbose output
        """
    )
    
    parser.add_argument("--work-dir", help="Custom work directory for test artifacts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested without running")
    
    args = parser.parse_args()
    
    # Display configuration
    print("RAMANUJAN ML FRAMEWORK TEST RUNNER")
    print("=" * 50)
    print(f"Work directory: {args.work_dir or 'Default (auto-generated)'}")
    print(f"Verbose: {args.verbose}")
    
    if args.dry_run:
        print("\nDRY RUN - Tests that would be executed:")
        test_categories = [
            "Framework Initialization",
            "Model Creation", 
            "Model Training",
            "Model Prediction",
            "Model Persistence",
            "Real Market Data Integration",
            "AutoML Functionality",
            "Model Comparison"
        ]
        for category in test_categories:
            print(f"  [OK] {category}")
        return 0
    
    try:
        # Initialize and run tests
        driver = RamanujanTestDriver(work_dir=args.work_dir)
        
        print(f"\nStarting tests at {datetime.now()}")
        driver.run_all_tests()
        
        # Return appropriate exit code
        return 1 if driver.test_results["failed"] > 0 else 0
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())