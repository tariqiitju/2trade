# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2Trade is a financial trading system with a modular architecture consisting of:

- **Odin's Eye**: Financial data access library providing unified access to market data, economic indicators, news sentiment, and Google Trends data
- **Consuela**: Configuration management system containing instrument lists and system requirements
- **Da Vinchi**: Feature engineering pipeline for financial trading with 8 stages (Stage 0-1 implemented)
- **Dave**: Testing framework and utilities
- **Ramanujan**: Comprehensive machine learning framework with 15+ prediction, clustering, and correlation models

## Core Architecture

### Data Storage Structure
The system operates on a data lake stored at `W:/market-data` (configurable via `DATA_ROOT` environment variable) with the following structure:
- `market_data/{interval}/` - OHLCV data in Parquet format organized by timeframes (daily, hourly, 30min, 15min, 5min, 1min)
- `economic_data/` - FRED economic indicators in Parquet format
- `news_data/` - Financial news with sentiment analysis in JSON format
- `trends_data/` - Google Trends search interest data in JSON format
- `earnings_data/` - Corporate earnings data including calendars, estimates, and historical reports in JSON format

### Configuration System
- **Data Specification**: `odins_eye/config/data-spec.yml` defines the complete data structure, schemas, and retention policies
- **Data Configuration**: `odins_eye/config/data-config.yml` contains data source configurations, storage settings, and operational parameters
- **Instrument Lists**: `consuela/config/instrument-list/` contains favorites and popular trading instruments in YAML format
- **API Keys**: `consuela/config/api_keys.yml` contains API keys for data sources (Finnhub, Alpha Vantage, FMP, etc.)
- **Data Quality**: `consuela/config/data-quality-config.yml` defines validation rules and quality thresholds for all data types

## Common Commands

### Dependencies
```bash
# Install required dependencies
pip install -r consuela/config/requirements.txt

# Additional dependencies for data collection
pip install pytrends  # For Google Trends data collection
```

### Data Collection Systems
```bash
# Comprehensive Earnings Data Collection
python scripts/download_comprehensive_earnings_data.py --upcoming-only --days-ahead 7
python scripts/download_comprehensive_earnings_data.py --symbols AAPL MSFT --analyze

# Enhanced Trends Data Collection with Tariff Focus
python scripts/download_comprehensive_trends_data.py --tariff-only
python scripts/download_comprehensive_trends_data.py --category trade_tariffs
python scripts/download_comprehensive_trends_data.py --analyze

# Data Quality Evaluation (includes earnings and trends validation)
python consuela/scripts/data_quality_evaluator.py
```

### Testing Odin's Eye Library
```bash
# Run complete test suite
cd dave/test/odins-eye
python run_tests.py

# Run tests with verbose output
python run_tests.py --verbose

# Run quick test subset
python run_tests.py --quick

# Save test results
python run_tests.py --save-results results.json

# Use custom data directory
python run_tests.py --data-root /path/to/data

# Run tests directly without configuration
python test_driver.py --verbose
```

### Testing Ramanujan ML Framework
```bash
# Run complete ML test suite
cd dave/test/ramanujan
python run_tests.py

# Run tests with verbose output
python run_tests.py --verbose

# Run tests with custom work directory
python run_tests.py --work-dir /tmp/ramanujan_tests

# Dry run to see available tests
python run_tests.py --dry-run

# Run tests directly
python test_driver.py
```

### Using Odin's Eye Library
```python
# Import and initialize
from odins_eye import OdinsEye, DateRange, MarketDataInterval

# Initialize with default data root (W:/market-data)
eye = OdinsEye()

# Or with custom data root
eye = OdinsEye(data_root="/custom/path")

# Get market data
data = eye.get_market_data("AAPL")
data = eye.get_market_data(["AAPL", "MSFT"], interval=MarketDataInterval.MIN_1)

# Get economic data
economic = eye.get_economic_data("UNRATE")

# Use date filtering
from datetime import datetime
date_range = DateRange(start_date="2023-01-01", end_date="2023-12-31")
filtered_data = eye.get_market_data("AAPL", date_range=date_range)
```

### Using Ramanujan ML Framework
```python
# Import framework and models
from ramanujan import ModelFramework
from ramanujan.config import ModelConfig, TrainingConfig

# Initialize framework
framework = ModelFramework()

# Create and train a prediction model
config = ModelConfig(
    model_type="xgboost",
    hyperparameters={"n_estimators": 100, "max_depth": 6}
)
training_config = TrainingConfig(cv_folds=5, test_size=0.2)

model_id = framework.create_model(config)
framework.train_model(model_id, X_train, y_train, training_config)

# Make predictions
predictions = framework.predict(model_id, X_test)

# AutoML functionality
best_models = framework.auto_ml(X_train, y_train, model_types=["xgboost", "lightgbm", "random_forest"])

# Model comparison
comparison = framework.compare_models(model_ids, X_test, y_test)
```

## Key Design Patterns

### Data Access Layer
Odin's Eye provides a unified interface over heterogeneous data sources:
- **Core Module** (`odins_eye/core.py`): Main `OdinsEye` class with data access methods
- **Filters Module** (`odins_eye/filters.py`): Comprehensive filtering system with enums, dataclasses for date ranges, instruments, and data types
- **Exceptions Module** (`odins_eye/exceptions.py`): Custom exception hierarchy for error handling

### Configuration-Driven Design
The system heavily relies on YAML configuration files that define:
- Data schemas and validation rules
- Storage formats and retention policies
- Data source configurations and API settings
- File naming conventions and directory structures

### Timezone Handling
Market data contains timezone-aware datetime columns (e.g., `America/New_York`). The filtering system automatically handles timezone synchronization between data and filter criteria using pandas' `tz_localize()` method.

### Testing Strategy
Comprehensive test coverage using real instrument data:
- Tests use actual instrument lists from `consuela/config/instrument-list/`
- Test configuration in `dave/test/odins-eye/test_config.yml`
- Both integration and unit tests covering all data types and filtering scenarios

### Machine Learning Framework
Ramanujan provides a unified interface over multiple ML model types:
- **Core Framework** (`ramanujan/core.py`): Main `ModelFramework` class with model lifecycle management
- **Base Architecture** (`ramanujan/base.py`): Abstract base classes defining common interfaces for all model types
- **Model Categories**: Prediction (supervised), Clustering (unsupervised), and Correlation analysis models
- **Configuration System** (`ramanujan/config.py`): Dataclass-based configuration with YAML serialization support
- **AutoML Pipeline**: Automated model selection, hyperparameter optimization, and ensemble creation

## Data Integration Points

### Market Data
- Supports multiple intervals from 1-minute to daily
- Includes technical indicators (SMA, VWAP, volatility, volume ratios)
- File format: `{symbol}.parquet` in interval-specific directories

### Economic Data
- FRED economic indicators with category and importance metadata
- Includes unemployment, interest rates, inflation, GDP, and market indicators
- File format: `{FRED_symbol}.parquet`

### Earnings Data (New)
- Comprehensive earnings data collection via multiple APIs (Finnhub, Alpha Vantage, FMP)
- Earnings calendars with upcoming and historical earnings events
- EPS estimates, actuals, and surprises analysis
- Historical quarterly and annual earnings reports
- File formats: JSON with date-based naming (`earnings_calendar_*.json`, `{symbol}_earnings.json`)

### Enhanced Trends Data
- Google Trends data with search interest scores and related queries
- **Tariff and Trade Policy Focus**: 39+ keywords including trade wars, section 232/301 tariffs
- **8 Trend Categories**: Financial stocks, trade tariffs, supply chain, geopolitical trade, etc.
- **Multi-timeframe Analysis**: 5-year policy trends, 12-month market trends
- File formats: JSON with category-based naming (`{category}_trends_{date}.json`)

### News Data
- Financial news with sentiment analysis and symbol association
- Multi-source aggregation with quality scoring
- File formats: JSON with date-based naming conventions

## Ramanujan ML Models

### Prediction Models (Supervised Learning)
- **XGBoostModel**: Gradient boosting with XGBoost library
- **LightGBMModel**: Fast gradient boosting with LightGBM
- **CatBoostModel**: Categorical boosting for datasets with categorical features
- **RandomForestModel**: Ensemble of decision trees with bagging
- **LinearRegressionModel**: Basic linear regression with regularization options
- **RidgeRegressionModel**: Ridge regression with L2 regularization
- **LSTMModel**: Long Short-Term Memory networks for time series (TensorFlow/Keras)
- **GARCHModel**: Generalized Autoregressive Conditional Heteroskedasticity for volatility modeling

### Clustering Models (Unsupervised Learning)
- **KMeansModel**: K-means clustering with silhouette score optimization
- **GMMModel**: Gaussian Mixture Model with AIC/BIC model selection
- **HMMModel**: Hidden Markov Model for sequential pattern discovery

### Correlation Models
- **PearsonCorrelationModel**: Linear correlation analysis
- **SpearmanCorrelationModel**: Rank-based correlation (monotonic relationships)
- **KendallCorrelationModel**: Tau correlation for ordinal data
- **TailDependenceModel**: Extreme value correlation using copulas
- **MutualInformationModel**: Information-theoretic dependence measure

## Development Notes

### Virtual Environment
The project uses a virtual environment at `venv/` for dependency isolation.

### Instrument Management
Trading instruments are centrally managed in `consuela/config/instrument-list/`:
- `favorites_instruments.yml`: Personal trading watchlist (78+ instruments)
- `popular_instruments.yml`: Popular trading instruments (98+ instruments)
- Includes stocks, ETFs, commodity ETFs with detailed metadata (exchange, sector, expense ratios)

### Error Handling
Custom exception hierarchy provides specific error types:
- `OdinsEyeError`: Base exception class
- `DataNotFoundError`: For missing data scenarios
- `ConfigurationError`: For setup and configuration issues
- `InvalidFilterError`: For invalid query parameters

### Module Structure
Each major component is self-contained:
- Independent configuration systems
- Separate test suites
- Modular import structure allowing selective usage

### ML Framework Integration
Ramanujan integrates seamlessly with Odin's Eye for financial data access:
- Automatic data loading from Odin's Eye for model training and prediction
- Support for time series prediction on market data
- Feature engineering on economic indicators and news sentiment
- Real-time model evaluation with live market data

### ML Testing Framework
Comprehensive test coverage in `dave/test/ramanujan/`:
- **Test Driver** (`test_driver.py`): 30+ test cases covering all framework functionality
- **Test Runner** (`run_tests.py`): CLI interface with verbose and dry-run options
- **Test Categories**: Framework initialization, model creation, training, prediction, persistence, AutoML, model comparison
- **Data Integration**: Tests with both synthetic data and real market data from Odin's Eye
- **Performance Validation**: Automatic metric calculation and model comparison

### ML Dependencies
Core ML libraries managed in `consuela/config/requirements.txt`:
- **Required**: pandas, numpy, scikit-learn, scipy, joblib
- **Optional**: xgboost, lightgbm, catboost, tensorflow, arch, hmmlearn, optuna
- **Graceful Degradation**: Framework automatically detects and skips unavailable model types

## Da Vinchi Feature Engineering Pipeline

### Overview
Da Vinchi is a comprehensive feature engineering system that transforms raw OHLCV market data into predictive features through an 8-stage pipeline. Currently, Stage 0 (Data Validator) and Stage 1 (Base Features) are implemented and tested.

### Architecture
- **Stage 0: Data Validator** (`da_vinchi/core/stage_0_data_validator.py`): Data validation, winsorization, quality scoring using Odin's Eye
- **Stage 1: Base Features** (`da_vinchi/core/stage_1_base_features.py`): 60+ technical indicators including returns, volatility, momentum, bands, and liquidity features
- **Pipeline Configuration**: YAML-based configuration system for feature selection and parameters
- **Testing Framework**: Comprehensive test suite in `dave/test/da-vinchi/` with model performance evaluation

### Key Features Generated (Stage 1)
- **Returns & Volatility**: Parkinson, Garman-Klass estimators, rolling volatility (5d, 20d, 60d)
- **Trend & Momentum**: RSI, MACD, Stochastic oscillators, ADX, ±DI indicators
- **Range & Bands**: ATR, Bollinger Bands, price position relative to bands
- **Liquidity & Microstructure**: Amihud illiquidity, Roll spread estimators, volume ratios

### Testing Da Vinchi Pipeline
```bash
cd dave/test/da-vinchi

# Run complete Da Vinchi test suite
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run specific stage tests
python run_tests.py --stage 0  # Data validator only
python run_tests.py --stage 1  # Base features only

# Run model performance evaluation
python run_tests.py --model-performance --verbose

# Standalone model performance testing
python run_model_performance_test.py --comprehensive
python run_model_performance_test.py --quick
```

### Model Performance Testing Framework
Comprehensive ML model evaluation system for testing Da Vinchi features:

```bash
# Multi-window training comparison
python run_model_performance_test.py --training-windows 60,30,15 --models random_forest,xgboost,ridge_regression

# Multi-instrument testing
python run_model_performance_test.py --instruments AAPL,GOOGL,MSFT --training-windows 60,30,15

# Custom configuration
python run_model_performance_test.py --instruments AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31 --max-features 20
```

### Model Performance Insights
Based on comprehensive testing across multiple instruments and training windows:

**Best Performing Models by Target:**
- **Price Prediction**: Ridge Regression (RMSE: 3.03, 96.12% directional accuracy)
- **Volatility Forecasting**: GARCH models (specialized for volatility clustering)  
- **Volume Prediction**: XGBoost (complex microstructure patterns)
- **Range Prediction**: Random Forest (non-linear support/resistance levels)

**Training Window Optimization:**
- **60-day windows**: Best for Ridge Regression, XGBoost, Linear models
- **30-day windows**: Optimal for Random Forest, balanced performance
- **15-day windows**: Poor for gradient boosting, acceptable for linear models

**Key Metrics Interpretation:**
- **RMSE/MAE**: Absolute error in dollars (scale-dependent, higher with mixed instruments)
- **MAPE**: Normalized percentage error (scale-independent, best for multi-instrument comparison)
- **Directional Accuracy**: Binary up/down prediction accuracy (critical for trading strategies)

**Feature Importance Patterns:**
- **Moving Averages**: EMA_12, EMA_50, SMA_20 dominate across models
- **Technical Indicators**: RSI, MACD, Bollinger Band position highly predictive
- **Volatility Features**: Garman-Klass, Parkinson estimators provide consistent signal

### Multi-Target Prediction Strategy
Da Vinchi supports specialized model selection for different prediction targets:
- **Close Price**: Ridge Regression (linear relationships with regularization)
- **High/Low Ranges**: Random Forest (non-linear pattern recognition)  
- **Volatility**: GARCH/LSTM (time series volatility clustering)
- **Volume**: XGBoost (complex market microstructure)
- **Opening Gaps**: LightGBM (overnight news/event processing)