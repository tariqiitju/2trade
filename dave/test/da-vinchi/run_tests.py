#!/usr/bin/env python3
"""
Da Vinchi Test Runner

CLI interface for running Da Vinchi feature engineering pipeline tests.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from test_driver import DaVinchiTestDriver


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Run Da Vinchi Feature Engineering Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                                 # Run all tests
  python run_tests.py --verbose                       # Verbose output
  python run_tests.py --stage 0                       # Run only Stage 0 tests
  python run_tests.py --stage 1                       # Run only Stage 1 tests
  python run_tests.py --instruments AAPL,MSFT         # Custom instruments
  python run_tests.py --start-date 2023-01-01         # Custom date range
  python run_tests.py --data-root /custom/path        # Custom data root
        """
    )
    
    # Test execution options
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with detailed logging")
    parser.add_argument("--stage", type=int, choices=[0, 1],
                       help="Run tests for specific stage only")
    parser.add_argument("--model-performance", action="store_true",
                       help="Run model performance evaluation tests")
    parser.add_argument("--config", help="Path to custom test config file")
    
    # Data options
    parser.add_argument("--data-root", help="Override data root directory")
    parser.add_argument("--instruments", help="Comma-separated list of instruments to test")
    parser.add_argument("--start-date", help="Start date for test data (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for test data (YYYY-MM-DD)")
    
    # Output options
    parser.add_argument("--save-results", action="store_true",
                       help="Save test results to JSON file")
    parser.add_argument("--work-dir", help="Custom work directory for test outputs")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DA VINCHI FEATURE ENGINEERING TEST RUNNER")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    try:
        # Initialize test driver
        driver = DaVinchiTestDriver(
            config_path=args.config,
            verbose=args.verbose
        )
        
        # Apply command line overrides
        if args.data_root:
            driver.data_root = args.data_root
            print(f"Using data root: {args.data_root}")
            
        if args.instruments:
            driver.test_instruments = args.instruments.split(',')
            print(f"Using instruments: {driver.test_instruments}")
            
        if args.start_date or args.end_date:
            date_range_config = driver.config.get('test_date_range', {})
            if args.start_date:
                date_range_config['start_date'] = args.start_date
            if args.end_date:
                date_range_config['end_date'] = args.end_date
            driver.config['test_date_range'] = date_range_config
            print(f"Using date range: {date_range_config.get('start_date')} to {date_range_config.get('end_date')}")
            
        if args.save_results:
            driver.config.setdefault('test_settings', {})['save_results'] = True
            
        if args.work_dir:
            driver.workspace_dir = Path(args.work_dir)
            driver.workspace_dir.mkdir(parents=True, exist_ok=True)
            print(f"Using work directory: {args.work_dir}")
        
        # Run specific stage tests, model performance tests, or all tests
        if args.stage is not None:
            print(f"Running Stage {args.stage} tests only")
            success = run_stage_tests(driver, args.stage)
        elif args.model_performance:
            print("Running model performance evaluation tests")
            success = run_model_performance_tests(driver)
        else:
            print("Running all tests")
            success = driver.run_all_tests()
        
        print(f"\nEnd time: {datetime.now()}")
        print("=" * 80)
        
        if success:
            print("ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("SOME TESTS FAILED")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_stage_tests(driver: DaVinchiTestDriver, stage: int) -> bool:
    """Run tests for a specific stage only"""
    print(f"\nRunning Stage {stage} tests...")
    
    try:
        if stage == 0:
            driver._test_stage0_data_validator()
        elif stage == 1:
            driver._test_stage1_base_features()
        else:
            raise ValueError(f"Invalid stage: {stage}")
            
        # Print results
        driver._print_test_summary()
        
        return len(driver.failed_tests) == 0
        
    except Exception as e:
        print(f"Error running Stage {stage} tests: {e}")
        return False


def run_model_performance_tests(driver: DaVinchiTestDriver) -> bool:
    """Run model performance evaluation tests"""
    print(f"\nRunning model performance evaluation tests...")
    
    try:
        driver._test_model_performance()
            
        # Print results
        driver._print_test_summary()
        
        return len(driver.failed_tests) == 0
        
    except Exception as e:
        print(f"Error running model performance tests: {e}")
        return False


if __name__ == "__main__":
    main()