#!/usr/bin/env python3
"""
Standalone Model Performance Test Runner

Runs comprehensive model performance evaluation to test how well Da Vinchi features
can predict stock prices using various ML models with rolling window validation.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from test_model_performance import ModelPerformanceTester, ModelPerformanceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main entry point for model performance testing"""
    parser = argparse.ArgumentParser(
        description="Test model performance with Da Vinchi generated features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with limited data
  python run_model_performance_test.py --quick
  
  # Comprehensive test with full 120-day windows
  python run_model_performance_test.py --comprehensive
  
  # Custom configuration
  python run_model_performance_test.py --instruments AAPL,MSFT,GOOGL --training-window 60 --models linear_regression,random_forest
  
  # Test with 30-day prediction horizon
  python run_model_performance_test.py --prediction-horizon 30 --training-window 120
        """
    )
    
    # Test configuration
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with limited data")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive test with full parameters")
    
    # Data configuration
    parser.add_argument("--instruments", default="AAPL",
                       help="Comma-separated list of instruments to test")
    parser.add_argument("--start-date", default="2023-01-01",
                       help="Start date for evaluation (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-12-31",
                       help="End date for evaluation (YYYY-MM-DD)")
    parser.add_argument("--data-root", 
                       help="Custom data root directory")
    
    # Model configuration
    parser.add_argument("--training-windows", default="60,30,15",
                       help="Comma-separated list of training window sizes in days")
    parser.add_argument("--prediction-horizon", type=int, default=1,
                       help="Number of days ahead to predict")
    parser.add_argument("--models", default="linear_regression,ridge_regression,random_forest",
                       help="Comma-separated list of models to test")
    parser.add_argument("--max-features", type=int, 
                       help="Maximum number of features to use")
    
    # Output configuration
    parser.add_argument("--output-file",
                       help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.quick:
        # Quick test configuration
        config = ModelPerformanceConfig(
            training_windows=[30, 15],
            prediction_horizon=1,
            min_samples_required=50,
            test_models=['linear_regression', 'random_forest'],
            max_features=10
        )
        instruments = ['AAPL']
        start_date = '2023-06-01'
        end_date = '2023-09-30'
        
    elif args.comprehensive:
        # Comprehensive test configuration
        config = ModelPerformanceConfig(
            training_windows=[120, 60, 30],
            prediction_horizon=1,
            min_samples_required=200,
            test_models=['linear_regression', 'ridge_regression', 'random_forest', 'xgboost', 'lightgbm'],
            max_features=30
        )
        instruments = ['AAPL', 'MSFT', 'GOOGL']
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
    else:
        # Custom configuration
        training_windows = [int(w) for w in args.training_windows.split(',')]
        config = ModelPerformanceConfig(
            training_windows=training_windows,
            prediction_horizon=args.prediction_horizon,
            min_samples_required=max(min(training_windows) * 2, 100),
            test_models=args.models.split(','),
            max_features=args.max_features
        )
        instruments = args.instruments.split(',')
        start_date = args.start_date
        end_date = args.end_date
    
    print("=" * 80)
    print("DA VINCHI MODEL PERFORMANCE EVALUATION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Instruments: {instruments}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Training windows: {config.training_windows} days")
    print(f"Prediction horizon: {config.prediction_horizon} day(s)")
    print(f"Models to test: {config.test_models}")
    if config.max_features:
        print(f"Max features: {config.max_features}")
    print()
    
    try:
        # Initialize tester
        tester = ModelPerformanceTester(config, data_root=args.data_root)
        
        print("Starting feature generation and model evaluation...")
        print("This may take several minutes depending on the configuration.")
        print()
        
        # Run evaluation
        performance_summary = tester.run_rolling_window_evaluation(
            instruments=instruments,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print results
        tester.print_performance_report()
        
        # Save results if requested
        if args.output_file:
            output_path = args.output_file
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"model_performance_results_{timestamp}.json"
        
        tester.save_results(output_path)
        print(f"\nDetailed results saved to: {output_path}")
        
        print(f"\nCompleted successfully at {datetime.now()}")
        
        # Summary statistics
        total_windows = performance_summary.get('total_windows', 0)
        models_tested = len(performance_summary.get('models_tested', []))
        
        print(f"\nSummary:")
        print(f"- Total prediction windows: {total_windows}")
        print(f"- Models successfully tested: {models_tested}")
        print(f"- Features used: Generated from Stage 0 + Stage 1 pipeline")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()