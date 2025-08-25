#!/usr/bin/env python3
"""
Ramanujan ML Framework Test Driver

Comprehensive test suite for the Ramanujan machine learning framework.
Tests all model types, training configurations, and prediction capabilities.
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add Ramanujan to Python path
# From dave/test/ramanujan/ go up to project root (2trade/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Added to Python path: {project_root}")

try:
    from ramanujan import ModelFramework
    from ramanujan.config import ModelConfig, TrainingConfig, ModelType
    from ramanujan.exceptions import RamanujanError, ModelError, TrainingError
    from odins_eye import OdinsEye, DateRange  # For real data
except ImportError as e:
    print(f"ERROR: Could not import required libraries: {e}")
    print("Make sure Ramanujan and Odin's Eye are properly installed and in the Python path.")
    sys.exit(1)


class RamanujanTestDriver:
    """Comprehensive test driver for Ramanujan ML framework"""
    
    def __init__(self, work_dir: Optional[str] = None):
        """Initialize test driver"""
        self.work_dir = work_dir or str(Path(__file__).parent / "test_workspace")
        self.framework = ModelFramework(self.work_dir)
        
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": []
        }
        
        # Initialize data access
        try:
            self.eye = OdinsEye()
            print(f"[OK] Initialized Odin's Eye for real market data")
        except Exception as e:
            self.eye = None
            self._add_warning(f"Could not initialize Odin's Eye: {e}")
    
    def _add_error(self, test_name: str, error: str):
        """Add error to test results"""
        self.test_results["failed"] += 1
        self.test_results["errors"].append(f"{test_name}: {error}")
    
    def _add_warning(self, warning: str):
        """Add warning to test results"""
        self.test_results["warnings"].append(warning)
    
    def _pass_test(self, test_name: str):
        """Mark test as passed"""
        self.test_results["passed"] += 1
        print(f"[OK] {test_name}")
    
    def _fail_test(self, test_name: str, error: str):
        """Mark test as failed"""
        self._add_error(test_name, error)
        print(f"[FAIL] {test_name}: {error}")
    
    def generate_sample_data(self, n_samples: int = 1000, n_features: int = 10) -> tuple:
        """Generate sample financial-like data for testing"""
        np.random.seed(42)
        
        # Generate features that simulate financial indicators
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create correlated features (like financial data)
        base_data = np.random.randn(n_samples, n_features)
        
        # Add some correlations and trends
        for i in range(1, n_features):
            base_data[:, i] += 0.3 * base_data[:, i-1]  # Correlation with previous feature
        
        # Add time trend
        time_trend = np.linspace(0, 1, n_samples)
        for i in range(n_features):
            base_data[:, i] += 0.2 * time_trend * ((-1) ** i)  # Alternating trends
        
        X = pd.DataFrame(base_data, columns=feature_names)
        
        # Generate target variable (like price or return prediction)
        y = (0.5 * X['feature_0'] + 
             0.3 * X['feature_1'] - 
             0.2 * X['feature_2'] + 
             0.1 * np.random.randn(n_samples))
        
        y = pd.Series(y, name='target')
        
        return X, y
    
    def get_real_market_data(self, symbols: List[str] = None, limit: int = 500) -> tuple:
        """Get real market data if available"""
        if self.eye is None:
            return None, None
        
        try:
            if symbols is None:
                symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            # Get market data for the first symbol
            data = self.eye.get_market_data(symbols[0])
            
            if data.empty:
                return None, None
            
            # Limit data size for testing
            data = data.tail(limit)
            
            # Select features and target
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'sma_20' in data.columns:
                feature_cols.extend(['sma_20', 'price_change', 'volatility_20'])
            
            # Only use columns that exist
            available_cols = [col for col in feature_cols if col in data.columns]
            
            X = data[available_cols].dropna()
            
            # Create target (next day return)
            y = data['close'].pct_change().shift(-1).dropna()
            
            # Align X and y
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            return X, y
            
        except Exception as e:
            self._add_warning(f"Failed to get real market data: {e}")
            return None, None
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        print("\n" + "="*60)
        print("TESTING: Framework Initialization")
        print("="*60)
        
        try:
            # Test basic initialization
            framework = ModelFramework()
            self._pass_test("Basic framework initialization")
            
            # Test custom work directory
            custom_dir = Path(self.work_dir) / "custom_test"
            framework = ModelFramework(custom_dir)
            
            if custom_dir.exists():
                self._pass_test("Custom work directory initialization")
            else:
                self._fail_test("Custom work directory initialization", "Directory not created")
            
            # Test directory structure
            required_dirs = ["models", "configs", "results", "logs"]
            missing_dirs = []
            for dir_name in required_dirs:
                if not (custom_dir / dir_name).exists():
                    missing_dirs.append(dir_name)
            
            if not missing_dirs:
                self._pass_test("Framework directory structure creation")
            else:
                self._fail_test("Framework directory structure creation", f"Missing directories: {missing_dirs}")
                
        except Exception as e:
            self._fail_test("Framework initialization", str(e))
    
    def test_model_creation(self):
        """Test model creation for all model types"""
        print("\n" + "="*60)
        print("TESTING: Model Creation")
        print("="*60)
        
        # Test prediction models
        prediction_models = [
            'xgboost', 'lightgbm', 'random_forest', 
            'linear_regression', 'ridge_regression'
        ]
        
        for model_type in prediction_models:
            try:
                model = self.framework.create_model(model_type, f"test_{model_type}")
                if model is not None:
                    self._pass_test(f"Create {model_type} model")
                else:
                    self._fail_test(f"Create {model_type} model", "Model is None")
            except Exception as e:
                # Some models might not be installed (like XGBoost)
                if "not installed" in str(e):
                    self._add_warning(f"{model_type}: {e}")
                else:
                    self._fail_test(f"Create {model_type} model", str(e))
        
        # Test clustering models
        clustering_models = ['kmeans', 'gmm']
        
        for model_type in clustering_models:
            try:
                model = self.framework.create_model(model_type, f"test_{model_type}")
                if model is not None:
                    self._pass_test(f"Create {model_type} model")
                else:
                    self._fail_test(f"Create {model_type} model", "Model is None")
            except Exception as e:
                if "not installed" in str(e):
                    self._add_warning(f"{model_type}: {e}")
                else:
                    self._fail_test(f"Create {model_type} model", str(e))
        
        # Test correlation models
        correlation_models = ['pearson', 'spearman', 'kendall', 'mutual_information']
        
        for model_type in correlation_models:
            try:
                model = self.framework.create_model(model_type, f"test_{model_type}")
                if model is not None:
                    self._pass_test(f"Create {model_type} model")
                else:
                    self._fail_test(f"Create {model_type} model", "Model is None")
            except Exception as e:
                self._fail_test(f"Create {model_type} model", str(e))
    
    def test_model_training(self):
        """Test model training with sample data"""
        print("\n" + "="*60)
        print("TESTING: Model Training")
        print("="*60)
        
        # Generate sample data
        X, y = self.generate_sample_data(200, 5)  # Smaller dataset for faster testing
        
        print(f"Using sample data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test supervised models
        supervised_models = ['linear_regression', 'ridge_regression', 'random_forest']
        
        for model_type in supervised_models:
            try:
                model = self.framework.create_model(model_type, f"train_test_{model_type}")
                
                # Quick training configuration
                training_config = TrainingConfig(
                    train_test_split=0.8,
                    cross_validation_folds=3,
                    verbose=False
                )
                
                results = self.framework.train_model(model, X, y, training_config)
                
                if results and 'train_rmse' in results:
                    self._pass_test(f"Train {model_type} model")
                    print(f"   Train RMSE: {results['train_rmse']:.4f}, Test RMSE: {results.get('test_rmse', 'N/A')}")
                else:
                    self._fail_test(f"Train {model_type} model", "No valid results returned")
                    
            except Exception as e:
                if "not installed" in str(e):
                    self._add_warning(f"{model_type}: {e}")
                else:
                    self._fail_test(f"Train {model_type} model", str(e))
        
        # Test unsupervised models
        unsupervised_models = ['kmeans']
        
        for model_type in unsupervised_models:
            try:
                model = self.framework.create_model(model_type, f"cluster_test_{model_type}")
                
                results = self.framework.train_model(model, X)  # No target for unsupervised
                
                if results and 'n_clusters' in results:
                    self._pass_test(f"Train {model_type} model")
                    print(f"   Clusters: {results['n_clusters']}, Silhouette Score: {results.get('silhouette_score', 'N/A'):.4f}")
                else:
                    self._fail_test(f"Train {model_type} model", "No valid results returned")
                    
            except Exception as e:
                self._fail_test(f"Train {model_type} model", str(e))
        
        # Test correlation models
        correlation_models = ['pearson', 'spearman']
        
        for model_type in correlation_models:
            try:
                model = self.framework.create_model(model_type, f"corr_test_{model_type}")
                
                results = self.framework.train_model(model, X)  # No target for correlation
                
                if results and 'mean_correlation' in results:
                    self._pass_test(f"Train {model_type} model")
                    print(f"   Mean Correlation: {results['mean_correlation']:.4f}")
                else:
                    self._fail_test(f"Train {model_type} model", "No valid results returned")
                    
            except Exception as e:
                self._fail_test(f"Train {model_type} model", str(e))
    
    def test_model_prediction(self):
        """Test model prediction"""
        print("\n" + "="*60)
        print("TESTING: Model Prediction")
        print("="*60)
        
        X, y = self.generate_sample_data(100, 5)
        
        # Test with pre-trained models
        model_names = [name for name in self.framework.list_models().keys() 
                      if 'train_test' in name or 'cluster_test' in name or 'corr_test' in name]
        
        for model_name in model_names[:3]:  # Test first 3 models
            try:
                predictions = self.framework.predict(model_name, X)
                
                if predictions is not None and len(predictions) > 0:
                    self._pass_test(f"Predict with {model_name}")
                    print(f"   Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
                else:
                    self._fail_test(f"Predict with {model_name}", "No predictions returned")
                    
            except Exception as e:
                self._fail_test(f"Predict with {model_name}", str(e))
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        print("\n" + "="*60)
        print("TESTING: Model Persistence")
        print("="*60)
        
        # Get a trained model
        trained_models = [name for name, info in self.framework.list_models().items() 
                         if info['is_trained']]
        
        if not trained_models:
            self._add_warning("No trained models available for persistence testing")
            return
        
        model_name = trained_models[0]
        
        try:
            # Save model
            saved_path = self.framework.save_model(model_name)
            
            if saved_path.exists():
                self._pass_test(f"Save model {model_name}")
                print(f"   Saved to: {saved_path}")
            else:
                self._fail_test(f"Save model {model_name}", "File not created")
                return
            
            # Load model
            loaded_model = self.framework.load_model(saved_path, f"loaded_{model_name}")
            
            if loaded_model is not None and loaded_model.is_trained:
                self._pass_test(f"Load model {model_name}")
            else:
                self._fail_test(f"Load model {model_name}", "Model not properly loaded")
                
        except Exception as e:
            self._fail_test(f"Model persistence for {model_name}", str(e))
    
    def test_real_market_data(self):
        """Test with real market data if available"""
        print("\n" + "="*60)
        print("TESTING: Real Market Data Integration")
        print("="*60)
        
        X, y = self.get_real_market_data()
        
        if X is None or y is None:
            self._add_warning("Real market data not available, skipping real data tests")
            return
        
        print(f"Using real market data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Features: {list(X.columns)}")
        
        try:
            # Test with linear regression on real data
            model = self.framework.create_model('linear_regression', 'real_data_test')
            
            training_config = TrainingConfig(
                train_test_split=0.8,
                cross_validation_folds=3,
                verbose=False
            )
            
            results = self.framework.train_model(model, X, y, training_config)
            
            if results and 'train_rmse' in results:
                self._pass_test("Train model with real market data")
                print(f"   Train RMSE: {results['train_rmse']:.4f}")
                print(f"   Test RMSE: {results.get('test_rmse', 'N/A')}")
                
                # Test prediction
                predictions = self.framework.predict(model, X.head(10))
                if predictions is not None:
                    self._pass_test("Predict with real market data")
                    print(f"   Sample predictions: {predictions[:5]}")
                else:
                    self._fail_test("Predict with real market data", "No predictions")
            else:
                self._fail_test("Train model with real market data", "No valid results")
                
        except Exception as e:
            self._fail_test("Real market data integration", str(e))
    
    def test_auto_ml(self):
        """Test AutoML functionality"""
        print("\n" + "="*60)
        print("TESTING: AutoML Functionality")
        print("="*60)
        
        X, y = self.generate_sample_data(150, 4)  # Small dataset for speed
        
        try:
            # Test AutoML with basic models
            model_types = ['linear_regression', 'ridge_regression']  # Use simple models
            
            automl_results = self.framework.auto_ml(
                X, y, 
                model_types=model_types,
                optimization_trials=5  # Small number for speed
            )
            
            if automl_results and 'best_model' in automl_results:
                self._pass_test("AutoML execution")
                print(f"   Best model: {automl_results['best_model']}")
                print(f"   Best score: {automl_results['best_score']:.4f}")
                
                # Check if all requested models were tested
                tested_models = list(automl_results['model_results'].keys())
                if set(tested_models) == set(model_types):
                    self._pass_test("AutoML model coverage")
                else:
                    self._fail_test("AutoML model coverage", f"Expected {model_types}, got {tested_models}")
            else:
                self._fail_test("AutoML execution", "No valid results returned")
                
        except Exception as e:
            self._fail_test("AutoML functionality", str(e))
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        print("\n" + "="*60)
        print("TESTING: Model Comparison")
        print("="*60)
        
        # Get trained models
        trained_models = [name for name, info in self.framework.list_models().items() 
                         if info['is_trained'] and 'train_test' in name]
        
        if len(trained_models) < 2:
            self._add_warning("Need at least 2 trained models for comparison testing")
            return
        
        X, y = self.generate_sample_data(100, 5)
        
        try:
            comparison_results = self.framework.compare_models(
                trained_models[:3], X, y, metrics=['rmse', 'r2']
            )
            
            if comparison_results is not None and not comparison_results.empty:
                self._pass_test("Model comparison")
                print(f"   Compared {len(comparison_results)} models")
                print(f"   Metrics: {list(comparison_results.columns)}")
            else:
                self._fail_test("Model comparison", "No comparison results")
                
        except Exception as e:
            self._fail_test("Model comparison", str(e))
    
    def run_all_tests(self):
        """Run all tests"""
        print("RAMANUJAN ML FRAMEWORK TEST DRIVER")
        print("=" * 80)
        print(f"Start time: {datetime.now()}")
        print(f"Work directory: {self.work_dir}")
        
        # Run test suites
        test_suites = [
            self.test_framework_initialization,
            self.test_model_creation,
            self.test_model_training,
            self.test_model_prediction,
            self.test_model_persistence,
            self.test_real_market_data,
            self.test_auto_ml,
            self.test_model_comparison
        ]
        
        for test_suite in test_suites:
            try:
                test_suite()
            except Exception as e:
                self._add_error(test_suite.__name__, f"Test suite failed: {e}")
                traceback.print_exc()
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print test results summary"""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        print(f"Total Tests: {total_tests}")
        print(f"[OK] Passed: {self.test_results['passed']}")
        print(f"[FAIL] Failed: {self.test_results['failed']}")
        print(f"[WARN] Warnings: {len(self.test_results['warnings'])}")
        
        if self.test_results["errors"]:
            print("\nERRORS:")
            for error in self.test_results["errors"]:
                print(f"  [FAIL] {error}")
        
        if self.test_results["warnings"]:
            print("\nWARNINGS:")
            for warning in self.test_results["warnings"]:
                print(f"  [WARN] {warning}")
        
        # Overall status
        if self.test_results["failed"] == 0:
            print(f"\n[SUCCESS] ALL TESTS PASSED! ({self.test_results['passed']} tests)")
        else:
            success_rate = (self.test_results["passed"] / total_tests * 100) if total_tests > 0 else 0
            print(f"\n[INFO] Test Success Rate: {success_rate:.1f}%")
        
        print(f"\nCompleted: {datetime.now()}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ramanujan ML Framework Test Driver")
    parser.add_argument("--work-dir", help="Custom work directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize test driver
        driver = RamanujanTestDriver(work_dir=args.work_dir)
        
        # Run tests
        driver.run_all_tests()
        
        # Exit with appropriate code
        exit_code = 1 if driver.test_results["failed"] > 0 else 0
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()