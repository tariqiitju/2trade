#!/usr/bin/env python3
"""
Odin's Eye Test Runner

Simple test runner script with configuration support and command-line options.
"""

import sys
import os
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from test_driver import OdinsEyeTestDriver


def load_test_config(config_path: str = None) -> dict:
    """Load test configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent / "test_config.yml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get("test_config", {})
    except Exception as e:
        print(f"Warning: Could not load test config: {e}")
        return {}


def save_test_results(results: dict, output_file: str):
    """Save test results to JSON file"""
    try:
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_tests": results["passed"] + results["failed"],
                "passed": results["passed"],
                "failed": results["failed"],
                "warnings": len(results["warnings"]),
                "success_rate": (results["passed"] / (results["passed"] + results["failed"]) * 100) if (results["passed"] + results["failed"]) > 0 else 0
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"✓ Test results saved to: {output_file}")
        
    except Exception as e:
        print(f"Warning: Could not save test results: {e}")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Run Odin's Eye Library Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests with default config
  python run_tests.py --quick           # Run quick test suite
  python run_tests.py --data-root /data # Use custom data directory
  python run_tests.py --config my.yml   # Use custom config file
  python run_tests.py --verbose         # Verbose output
        """
    )
    
    parser.add_argument("--config", "-c", help="Path to test configuration file")
    parser.add_argument("--data-root", help="Override data root directory")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--save-results", help="Save results to specified file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested without running")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_test_config(args.config)
    
    # Override config with command line arguments
    if args.data_root:
        config["data_root"] = args.data_root
    if args.quick:
        config["quick_mode"] = True
    if args.verbose:
        config["verbose"] = True
    
    # Display configuration
    print("ODIN'S EYE TEST RUNNER")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Data root: {config.get('data_root', 'Default (W:/market-data)')}")
    print(f"  Quick mode: {config.get('quick_mode', False)}")
    print(f"  Verbose: {config.get('verbose', False)}")
    print(f"  Save results: {bool(args.save_results or config.get('output', {}).get('save_results'))}")
    
    if args.dry_run:
        print("\nDRY RUN - Tests that would be executed:")
        test_categories = config.get('test_categories', {})
        for category, enabled in test_categories.items():
            status = "✓ ENABLED" if enabled else "✗ DISABLED"
            print(f"  {category:15} {status}")
        return 0
    
    try:
        # Initialize and run tests
        driver = OdinsEyeTestDriver(data_root=config.get('data_root'))
        
        print(f"\nStarting tests at {datetime.now()}")
        driver.run_all_tests()
        
        # Save results if requested
        results_file = args.save_results or (config.get('output', {}).get('save_results') and config.get('output', {}).get('results_file'))
        if results_file:
            save_test_results(driver.test_results, results_file)
        
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