# Ramanujan ML Framework Test Suite

Comprehensive test suite for the Ramanujan machine learning framework.

## Overview

This test suite validates all functionality of the Ramanujan ML framework including:
- **Prediction Models**: XGBoost, LightGBM, CatBoost, Random Forest, Linear/Ridge Regression, LSTM, GARCH
- **Clustering Models**: K-Means, Gaussian Mixture Models, Hidden Markov Models
- **Correlation Models**: Pearson, Spearman, Kendall correlations, Tail Dependence, Mutual Information
- Training configurations and optimization
- Model persistence (save/load)
- Real market data integration with Odin's Eye
- AutoML functionality
- Model comparison and evaluation

## Test Structure

```
dave/test/ramanujan/
├── test_driver.py      # Main test driver with all test cases
├── run_tests.py        # Test runner script
└── README.md          # This file
```

## Usage

### Basic Usage

```bash
# Run all tests
cd dave/test/ramanujan
python test_driver.py

# Or use the test runner
python run_tests.py
```

### Command Line Options

```bash
# Use custom work directory
python run_tests.py --work-dir /tmp/ramanujan_tests

# Verbose output
python run_tests.py --verbose

# Dry run (show what would be tested)
python run_tests.py --dry-run
```

## Test Categories

### 1. Framework Initialization
- Basic framework setup
- Work directory creation
- Directory structure validation
- Logging initialization

### 2. Model Creation
- All prediction model types (supervised learning)
- All clustering model types (unsupervised learning)
- All correlation model types
- Parameter validation
- Model registry functionality

### 3. Model Training
- Supervised model training with synthetic data
- Unsupervised model training
- Correlation analysis
- Training configuration handling
- Cross-validation and metrics

### 4. Model Prediction
- Prediction with trained models
- Batch prediction
- Different prediction configurations
- Error handling for untrained models

### 5. Model Persistence
- Model saving (joblib/pickle formats)
- Model loading
- Metadata preservation
- Configuration restoration

### 6. Real Market Data Integration
- Integration with Odin's Eye data access
- Training on real financial data
- Feature engineering validation
- Time series data handling

### 7. AutoML Functionality
- Automated model selection
- Hyperparameter optimization
- Model comparison
- Best model identification

### 8. Model Comparison
- Performance comparison across models
- Metric calculation
- Results visualization preparation
- Statistical significance testing

## Expected Output

### Successful Run
```
RAMANUJAN ML FRAMEWORK TEST DRIVER
================================================================================
Start time: 2024-08-23 15:30:00.123456
Work directory: /path/to/test_workspace

============================================================
TESTING: Framework Initialization
============================================================
[OK] Basic framework initialization
[OK] Custom work directory initialization
[OK] Framework directory structure creation

============================================================
TESTING: Model Creation
============================================================
[OK] Create xgboost model
[OK] Create lightgbm model
[OK] Create random_forest model
[OK] Create linear_regression model
[OK] Create kmeans model
[OK] Create pearson model

============================================================
TESTING: Model Training
============================================================
Using sample data: 200 samples, 5 features
[OK] Train linear_regression model
   Train RMSE: 0.8234, Test RMSE: 0.9123
[OK] Train random_forest model
   Train RMSE: 0.1234, Test RMSE: 0.7891
[OK] Train kmeans model
   Clusters: 8, Silhouette Score: 0.4567

... [more test output] ...

================================================================================
TEST RESULTS SUMMARY
================================================================================
Total Tests: 45
[OK] Passed: 42
[FAIL] Failed: 0
[WARN] Warnings: 3

WARNINGS:
  [WARN] catboost: catboost not installed. Install with: pip install catboost
  [WARN] lstm: TensorFlow not installed. Install with: pip install tensorflow
  [WARN] Could not initialize Odin's Eye: Data root directory does not exist

[SUCCESS] ALL TESTS PASSED! (42 tests)

Completed: 2024-08-23 15:35:15.789012
```

### Failed Tests
```
================================================================================
TEST RESULTS SUMMARY
================================================================================
Total Tests: 45
[OK] Passed: 38
[FAIL] Failed: 4
[WARN] Warnings: 3

ERRORS:
  [FAIL] Train xgboost model: XGBoost not installed. Install with: pip install xgboost
  [FAIL] AutoML functionality: No valid models available for AutoML
  [FAIL] Model persistence for test_model: Permission denied writing to directory

[INFO] Test Success Rate: 84.4%
```

## Dependencies

The test suite will automatically detect and skip tests for optional ML libraries that aren't installed:

### Core Dependencies (Required)
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- scipy>=1.7.0

### Optional ML Libraries
- xgboost>=1.6.0
- lightgbm>=3.3.0
- catboost>=1.0.0
- tensorflow>=2.8.0
- arch>=5.3.0 (for GARCH models)
- hmmlearn>=0.2.7 (for HMM models)

### Install Dependencies
```bash
# Install core dependencies
pip install -r consuela/config/requirements.txt

# Install Ramanujan ML dependencies
pip install -r ramanujan/requirements.txt

# Or install specific libraries as needed:
pip install xgboost lightgbm catboost tensorflow
```

## Integration with Odin's Eye

The test suite integrates with the Odin's Eye data access library to:
- Test models with real financial market data
- Validate feature engineering on real data
- Test time series prediction scenarios
- Ensure compatibility between frameworks

If Odin's Eye data is not available, the tests gracefully fall back to synthetic data.

## Troubleshooting

### Common Issues

1. **Missing ML Libraries**
   ```
   WARNING: xgboost not installed. Install with: pip install xgboost
   ```
   - Install the missing library or ignore if not needed
   - The test suite will skip unavailable models

2. **Data Directory Issues**
   ```
   FAIL: Real market data integration: Data root directory does not exist
   ```
   - Ensure Odin's Eye data is available
   - Tests will use synthetic data if real data unavailable

3. **Permission Issues**
   ```
   FAIL: Model persistence: Permission denied writing to directory
   ```
   - Check write permissions for the work directory
   - Use `--work-dir` to specify a writable location

4. **Memory Issues**
   ```
   FAIL: Train lstm model: Out of memory
   ```
   - Reduce dataset size in test configuration
   - Use smaller model parameters

### Debugging

Enable verbose mode for detailed output:
```bash
python run_tests.py --verbose
```

Use dry-run to see which tests would execute:
```bash
python run_tests.py --dry-run
```

## Performance Notes

- Test suite typically takes 2-5 minutes to complete
- Time varies based on available ML libraries
- LSTM and deep learning models take longer
- Real data integration adds time if large datasets available

## Contributing

When adding new models to Ramanujan:

1. Add corresponding tests to `test_driver.py`
2. Update model creation tests
3. Add training tests with appropriate data
4. Test prediction functionality
5. Verify persistence works correctly
6. Update this README if new test categories are added