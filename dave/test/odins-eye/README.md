# Odin's Eye Test Suite

Comprehensive test suite for the Odin's Eye financial data access library.

## Overview

This test suite validates all functionality of the Odin's Eye library including:
- Market data access across different time intervals
- Economic data retrieval from FRED indicators
- News data with sentiment analysis
- Google Trends data access
- Date range and instrument filtering
- Custom queries with complex filters
- Error handling and edge cases

## Test Structure

```
dave/test/odins-eye/
├── test_driver.py      # Main test driver with all test cases
├── run_tests.py        # Test runner script with configuration
├── test_config.yml     # Test configuration settings  
└── README.md          # This file
```

## Usage

### Basic Usage

```bash
# Run all tests with default settings
cd dave/test/odins-eye
python test_driver.py

# Or use the test runner
python run_tests.py
```

### Command Line Options

```bash
# Use custom data directory
python test_driver.py --data-root /path/to/data

# Run quick tests only  
python run_tests.py --quick

# Verbose output
python run_tests.py --verbose

# Save results to file
python run_tests.py --save-results results.json

# Use custom config
python run_tests.py --config my_config.yml

# Dry run (show what would be tested)
python run_tests.py --dry-run
```

## Test Categories

### 1. Basic Initialization
- Library import and initialization
- Configuration loading
- Data root validation
- Data availability information

### 2. Market Data Access
- Single symbol daily data retrieval
- Multiple symbols batch access
- Different time intervals (daily, hourly, minute-level)
- Technical indicators inclusion
- Volume and price filtering

### 3. Date Range Filtering
- Recent data (last 30 days)
- Specific date ranges (e.g., 2023 data)
- Open-ended ranges (from date to present)
- Invalid date range handling

### 4. Economic Data
- Individual FRED indicators
- Multiple indicators batch retrieval
- Category and importance filtering
- Frequency-based filtering

### 5. News Data
- Symbol-specific news retrieval
- Sentiment-based filtering
- Source-based filtering
- Date range filtering for news

### 6. Google Trends Data
- Symbol-based trends
- Keyword-based trends
- Geographic filtering
- Search interest thresholds

### 7. Custom Queries
- Complex multi-filter queries
- QueryFilter class usage
- Sorting and limiting results
- Multiple data types in one query

### 8. Utility Functions
- List available symbols
- Get data availability info
- Symbol validation

### 9. Error Handling
- Invalid symbols
- Invalid date ranges
- Missing data graceful handling
- Configuration errors

## Configuration

The test suite uses `test_config.yml` for configuration:

```yaml
test_config:
  data_root: "W:/market-data"
  test_symbols:
    max_stocks: 10
    priority_symbols:
      stocks: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
  economic_indicators:
    primary: ["UNRATE", "DGS10", "VIXCLS", "GDP"]
  # ... more settings
```

## Test Data Sources

The test suite uses real instrument lists from:
- `consuela/config/instrument-list/favorites_instruments.yml` (100+ favorite instruments)
- `consuela/config/instrument-list/popular_instruments.yml` (100+ popular instruments)

This ensures tests use realistic symbols and instruments that are commonly traded.

## Expected Output

### Successful Run
```
ODIN'S EYE LIBRARY TEST DRIVER
================================================================================
Start time: 2024-08-23 10:30:00.123456
Data root: W:/market-data
Test instruments loaded: 95 favorites, 100 popular

============================================================
TESTING: Basic Initialization  
============================================================
✓ Default initialization
✓ Get data info
   Data root: W:/market-data
   market_data: {'daily': {'available_symbols': 150, 'symbols': ['AAPL', 'MSFT', ...]}}

============================================================
TESTING: Market Data Access
============================================================
Testing with symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
✓ Get daily data for AAPL
   Records: 2500 rows
   Columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'sma_20', ...]
   Date range: 2015-01-01 to 2024-08-23

... [more test output] ...

================================================================================
TEST RESULTS SUMMARY
================================================================================
Total Tests: 25
✓ Passed: 23
✗ Failed: 0  
⚠ Warnings: 2

WARNINGS:
  ⚠ No news data available for AAPL
  ⚠ No trends data for keyword tests

🎉 ALL TESTS PASSED! (23 tests)

Completed: 2024-08-23 10:35:15.789012
```

### Failed Tests
```
================================================================================
TEST RESULTS SUMMARY
================================================================================
Total Tests: 25
✓ Passed: 20
✗ Failed: 3
⚠ Warnings: 2

ERRORS:
  ✗ Get daily data for AAPL: File not found: W:/market-data/market_data/daily/AAPL.parquet
  ✗ Complex market data query: Invalid filter configuration
  ✗ List available symbols: Permission denied accessing data directory

📊 Test Success Rate: 80.0%
```

## Exit Codes

- `0`: All tests passed
- `1`: Some tests failed or fatal error occurred  
- `130`: Tests interrupted by user (Ctrl+C)

## Troubleshooting

### Common Issues

1. **Data Directory Not Found**
   ```
   ERROR: Data root directory does not exist: W:/market-data
   ```
   - Ensure the data directory exists and is accessible
   - Use `--data-root` to specify correct path

2. **No Test Data Available**
   ```
   WARNING: No stock symbols available for testing
   ```
   - Verify data files exist in the specified directory
   - Check instrument list files are properly loaded

3. **Import Errors**
   ```
   ERROR: Could not import Odin's Eye library
   ```
   - Ensure dependencies are installed: `pip install -r consuela/config/requirements.txt`
   - Verify Odin's Eye library path is correct

4. **Permission Issues**
   ```
   Permission denied accessing data directory
   ```
   - Check file/directory permissions
   - Run with appropriate user permissions

### Debugging

Enable verbose mode for detailed output:
```bash
python run_tests.py --verbose
```

Use dry-run to see which tests would execute:
```bash
python run_tests.py --dry-run
```

## Integration with CI/CD

The test suite can be integrated into continuous integration:

```bash
# In CI/CD pipeline
cd dave/test/odins-eye
python run_tests.py --save-results ci_results.json
exit_code=$?

# Upload results or process them
if [ $exit_code -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    cat ci_results.json
fi
```

## Contributing

When adding new features to Odin's Eye:

1. Add corresponding tests to `test_driver.py`
2. Update test configuration in `test_config.yml` if needed
3. Run tests to ensure no regressions
4. Update this README if new test categories are added