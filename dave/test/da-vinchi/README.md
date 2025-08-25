# Da Vinchi Feature Engineering Tests

This directory contains comprehensive tests for the Da Vinchi feature engineering pipeline.

## Test Structure

- `test_driver.py` - Main test driver with all test cases
- `run_tests.py` - Test runner with CLI interface  
- `test_config.yml` - Test configuration settings

## Test Coverage

### Stage 0: Data Validator
- Data loading and validation using Odin's Eye
- OHLCV relationship validation
- Winsorization and outlier handling
- Trading calendar alignment
- Data quality reporting

### Stage 1: Base Features
- Returns and volatility features
- Trend and momentum indicators
- Range and band calculations
- Liquidity and microstructure features

### Model Performance Testing
- Rolling window prediction evaluation
- Multiple ML model comparison (Linear Regression, Random Forest, XGBoost, etc.)
- Feature quality assessment via predictive performance
- Comprehensive performance metrics (RMSE, MAE, Directional Accuracy, etc.)
- 120-day training windows with configurable prediction horizons

## Running Tests

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run specific stage tests
python run_tests.py --stage 0
python run_tests.py --stage 1

# Use custom instruments
python run_tests.py --instruments AAPL,MSFT,TSLA

# Use custom date range  
python run_tests.py --start-date 2023-01-01 --end-date 2023-12-31

# Run model performance evaluation
python run_tests.py --model-performance --verbose

# Run standalone model performance test
python run_model_performance_test.py --comprehensive

# Quick model performance test
python run_model_performance_test.py --quick

# Custom model performance test
python run_model_performance_test.py --instruments AAPL,MSFT --training-window 60 --models random_forest,xgboost
```

## Test Data

Tests use real market data from Odin's Eye with instruments from:
- `consuela/config/instrument-list/favorites_instruments.yml`
- `consuela/config/instrument-list/popular_instruments.yml`