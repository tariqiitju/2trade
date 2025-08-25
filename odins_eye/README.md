# Odin's Eye 👁️

**Financial Data Access Library**

Odin's Eye is a Python library that provides unified access to financial market data, economic indicators, news sentiment, and Google Trends data stored in your trading system's data lake.

## Features

- 📊 **Market Data**: Access OHLCV data with technical indicators across multiple timeframes
- 📈 **Economic Data**: Query FRED economic indicators with category and importance filtering  
- 📰 **News Data**: Retrieve financial news with sentiment analysis and source filtering
- 🔍 **Trends Data**: Access Google Trends search interest data
- ⚡ **Flexible Filtering**: Comprehensive filtering by time ranges, instruments, and data-specific criteria
- 🔧 **Configurable**: Customizable data root and configuration settings

## Quick Start

### Installation

```python
# Add the odins-eye directory to your Python path
import sys
sys.path.append('/path/to/2trade/odins-eye')

from odins_eye import OdinsEye, DateRange, MarketDataInterval
```

### Basic Usage

```python
# Initialize with default data root (W:/market-data)
eye = OdinsEye()

# Or specify custom data root
eye = OdinsEye(data_root="/custom/data/path")

# Get daily market data
aapl_data = eye.get_market_data("AAPL")

# Get multiple symbols
tech_stocks = eye.get_market_data(["AAPL", "MSFT", "GOOGL"])

# Get intraday data
aapl_1min = eye.get_market_data("AAPL", interval=MarketDataInterval.MIN_1)
```

## Data Types and Intervals

### Market Data Intervals
- `DAILY` - Daily OHLCV data (15 years retention)
- `HOURLY` - Hourly intraday data (2 years retention)
- `MIN_30` - 30-minute data (2 months retention)
- `MIN_15` - 15-minute data (2 months retention)  
- `MIN_5` - 5-minute data (2 months retention)
- `MIN_1` - 1-minute data (1 week retention)

### Data Types
- **Market Data**: OHLCV with technical indicators
- **Economic Data**: FRED economic indicators
- **News Data**: Financial news with sentiment analysis
- **Trends Data**: Google Trends search interest

## Advanced Filtering

### Date Range Filtering

```python
from datetime import datetime
from odins_eye import DateRange

# Specific date range
date_range = DateRange(
    start_date="2023-01-01",
    end_date="2023-12-31"
)

data = eye.get_market_data("AAPL", date_range=date_range)

# From date to present
recent = eye.get_market_data(
    "AAPL", 
    date_range=DateRange(start_date="2024-01-01")
)
```

### Market Data Filtering

```python
# Volume filtering
high_volume = eye.get_market_data(
    ["AAPL", "TSLA"], 
    min_volume=1000000,
    date_range=DateRange(start_date="2024-01-01")
)

# Price range filtering
price_filtered = eye.get_market_data(
    "AAPL",
    price_range=(150.0, 200.0),  # $150-$200
    include_indicators=["sma_20", "sma_50", "volatility_20"]
)
```

### Economic Data

```python
# Single indicator
unemployment = eye.get_economic_data("UNRATE")

# Multiple indicators with filters
indicators = eye.get_economic_data(
    ["DGS10", "UNRATE", "VIXCLS"],
    importance="high",
    category="interest_rates"
)
```

### News Data

```python
# Symbol-specific news
aapl_news = eye.get_news_data(
    symbols="AAPL",
    sentiment_label="positive",
    date_range=DateRange(start_date="2024-01-01")
)

# Source filtering
bloomberg_news = eye.get_news_data(
    sources=["Bloomberg", "Reuters"],
    sentiment_score_range=(0.3, 1.0)  # Positive sentiment
)
```

### Google Trends

```python
# Keyword trends
market_trends = eye.get_trends_data(
    keywords=["stock market", "recession", "AI"],
    geo="US",
    min_search_interest=50
)
```

## Custom Queries

For complex filtering requirements, use the `QueryFilter` class:

```python
from odins_eye.filters import QueryFilter, DataType, InstrumentFilter, DataTypeFilter

# Complex multi-filter query
complex_filter = QueryFilter(
    date_range=DateRange(start_date="2024-01-01"),
    instruments=InstrumentFilter(
        symbols=["AAPL", "MSFT", "GOOGL"],
        exclude_symbols=["TSLA"],
        symbol_pattern=r"^[A-Z]{3,4}$"  # 3-4 letter symbols
    ),
    data_types=[
        DataTypeFilter(
            data_type=DataType.MARKET_DATA,
            interval=MarketDataInterval.DAILY,
            min_volume=500000,
            price_range=(50.0, 500.0)
        )
    ],
    limit=1000,
    sort_by="date",
    sort_order="desc"
)

results = eye.query(complex_filter)
```

## Utility Functions

```python
# List available symbols
symbols = eye.list_available_symbols(DataType.MARKET_DATA, MarketDataInterval.DAILY)
print(f"Available symbols: {len(symbols)}")

# Get data availability info
info = eye.get_data_info()
print(info)
```

## Configuration

### Default Configuration

The library uses configuration files in `odins-eye/config/`:
- `data-spec.yml` - Data structure specification
- `data-config.yml` - Storage and source configuration

### Custom Configuration

```python
# Custom data root via environment variable
import os
os.environ['DATA_ROOT'] = '/custom/data/path'
eye = OdinsEye()

# Custom data root via parameter
eye = OdinsEye(data_root='/custom/data/path')

# Custom config file
eye = OdinsEye(config_path='/path/to/custom-config.yml')
```

## Error Handling

```python
from odins_eye.exceptions import DataNotFoundError, InvalidFilterError

try:
    data = eye.get_market_data("INVALID_SYMBOL")
except DataNotFoundError as e:
    print(f"Data not found: {e}")
except InvalidFilterError as e:
    print(f"Invalid filter: {e}")
```

## Data Structure

### Market Data Schema
```python
# DataFrame columns
columns = [
    'date',           # datetime64[ns] - Trading date/time
    'open',           # float64 - Opening price  
    'high',           # float64 - High price
    'low',            # float64 - Low price
    'close',          # float64 - Closing price
    'volume',         # int64 - Trading volume
    'symbol',         # string - Stock symbol
    # Technical indicators (varies by interval)
    'sma_20',         # 20-period SMA (daily)
    'sma_5',          # 5-period SMA (intraday)
    'vwap',           # Volume weighted average price
    'volatility_20',  # 20-period volatility
    'price_change',   # Period-over-period change %
]
```

### Economic Data Schema
```python
columns = [
    'date',           # datetime64[ns] - Observation date
    'value',          # float64 - Indicator value
    'symbol',         # string - FRED series symbol  
    'indicator_name', # string - Human-readable name
    'category',       # string - Indicator category
    'frequency',      # string - Data frequency
    'importance',     # string - Market importance
]
```

## Examples

See `examples.py` for comprehensive usage examples including:
- Basic data retrieval
- Advanced filtering techniques
- News and trends analysis
- Custom query construction
- Utility function usage

## Requirements

- Python 3.7+
- pandas
- pyyaml
- Data stored in the expected directory structure

## License

Part of the 2Trade trading system.