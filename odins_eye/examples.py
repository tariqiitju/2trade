"""
Example usage of the Odin's Eye library
"""

from datetime import datetime, date
from odins_eye import OdinsEye, DateRange, InstrumentFilter, DataTypeFilter, MarketDataInterval

def basic_examples():
    """Basic usage examples"""
    
    # Initialize with default data root (W:/market-data)
    eye = OdinsEye()
    
    # Initialize with custom data root
    # eye = OdinsEye(data_root="C:/custom/data/path")
    
    print("=== Basic Market Data Retrieval ===")
    
    # Get daily data for a single symbol
    aapl_daily = eye.get_market_data("AAPL")
    print(f"AAPL daily data: {len(aapl_daily)} records")
    
    # Get data for multiple symbols
    tech_stocks = eye.get_market_data(["AAPL", "MSFT", "GOOGL"])
    print(f"Tech stocks data: {len(tech_stocks)} records")
    
    # Get intraday data
    aapl_1min = eye.get_market_data("AAPL", interval=MarketDataInterval.MIN_1)
    print(f"AAPL 1-minute data: {len(aapl_1min)} records")
    
    print("\n=== Economic Data ===")
    
    # Get unemployment rate data
    unemployment = eye.get_economic_data("UNRATE")
    print(f"Unemployment rate data: {len(unemployment)} records")
    
    # Get multiple economic indicators
    indicators = eye.get_economic_data(["DGS10", "UNRATE", "VIXCLS"])
    print(f"Economic indicators data: {len(indicators)} records")


def advanced_filtering_examples():
    """Advanced filtering examples"""
    
    eye = OdinsEye()
    
    print("=== Date Range Filtering ===")
    
    # Get data for specific date range
    date_range = DateRange(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    aapl_2023 = eye.get_market_data(
        "AAPL", 
        date_range=date_range
    )
    print(f"AAPL 2023 data: {len(aapl_2023)} records")
    
    # Get data from start date to present
    recent_data = eye.get_market_data(
        "AAPL",
        date_range=DateRange(start_date="2024-01-01")
    )
    print(f"AAPL recent data: {len(recent_data)} records")
    
    print("\n=== Volume and Price Filtering ===")
    
    # Get high-volume trading data
    high_volume = eye.get_market_data(
        ["AAPL", "TSLA", "SPY"],
        min_volume=1000000,  # Minimum 1M volume
        date_range=DateRange(start_date="2024-01-01")
    )
    print(f"High volume data: {len(high_volume)} records")
    
    # Get data within price range
    price_filtered = eye.get_market_data(
        "AAPL",
        price_range=(150.0, 200.0),  # Price between $150-200
        date_range=DateRange(start_date="2024-01-01")
    )
    print(f"Price filtered data: {len(price_filtered)} records")


def news_and_trends_examples():
    """News and trends data examples"""
    
    eye = OdinsEye()
    
    print("=== News Data ===")
    
    # Get news for specific symbols
    aapl_news = eye.get_news_data(
        symbols="AAPL",
        date_range=DateRange(start_date="2024-01-01")
    )
    print(f"AAPL news articles: {len(aapl_news)}")
    
    # Get positive sentiment news
    positive_news = eye.get_news_data(
        symbols=["AAPL", "MSFT"],
        sentiment_label="positive",
        date_range=DateRange(start_date="2024-08-01")
    )
    print(f"Positive news: {len(positive_news)}")
    
    # Get news from specific sources
    reuters_news = eye.get_news_data(
        sources=["Reuters", "Bloomberg"],
        date_range=DateRange(start_date="2024-08-01")
    )
    print(f"Reuters/Bloomberg news: {len(reuters_news)}")
    
    print("\n=== Google Trends Data ===")
    
    # Get trends for stock symbols
    aapl_trends = eye.get_trends_data(
        keywords="AAPL",
        date_range=DateRange(start_date="2024-01-01")
    )
    print(f"AAPL trends data: {len(aapl_trends)}")
    
    # Get trends for multiple keywords
    market_trends = eye.get_trends_data(
        keywords=["stock market", "recession", "AI"],
        geo="US"
    )
    print(f"Market trends: {len(market_trends)}")


def custom_query_examples():
    """Custom query examples using QueryFilter"""
    
    from odins_eye.filters import QueryFilter, DataType, SentimentLabel
    
    eye = OdinsEye()
    
    print("=== Custom Query Examples ===")
    
    # Complex query combining multiple filters
    complex_filter = QueryFilter(
        date_range=DateRange(start_date="2024-01-01"),
        instruments=InstrumentFilter(
            symbols=["AAPL", "MSFT", "GOOGL"],
            exclude_symbols=["TSLA"]
        ),
        data_types=[
            DataTypeFilter(
                data_type=DataType.MARKET_DATA,
                interval=MarketDataInterval.DAILY,
                min_volume=500000,
                include_indicators=["sma_20", "sma_50", "volatility_20"]
            )
        ],
        limit=1000,
        sort_by="date",
        sort_order="desc"
    )
    
    results = eye.query(complex_filter)
    print(f"Complex query results: {len(results)} records")
    
    # Multi-data-type query
    multi_type_filter = QueryFilter(
        date_range=DateRange(start_date="2024-06-01"),
        instruments=InstrumentFilter(symbols=["AAPL"]),
        data_types=[
            DataTypeFilter(
                data_type=DataType.MARKET_DATA,
                interval=MarketDataInterval.DAILY
            )
        ]
    )
    
    market_results = eye.query(multi_type_filter)
    print(f"Multi-type query - Market: {len(market_results)} records")


def utility_examples():
    """Utility function examples"""
    
    eye = OdinsEye()
    
    print("=== Utility Functions ===")
    
    # List available symbols
    daily_symbols = eye.list_available_symbols(DataType.MARKET_DATA, MarketDataInterval.DAILY)
    print(f"Available daily symbols: {len(daily_symbols)}")
    print(f"Sample symbols: {daily_symbols[:10]}")
    
    # List economic indicators
    econ_indicators = eye.list_available_symbols(DataType.ECONOMIC_DATA)
    print(f"Available economic indicators: {len(econ_indicators)}")
    print(f"Sample indicators: {econ_indicators[:5]}")
    
    # Get data information
    data_info = eye.get_data_info()
    print(f"Data info: {data_info}")


def main():
    """Run all examples"""
    try:
        print("Odin's Eye Library Examples\n")
        print("=" * 50)
        
        basic_examples()
        print("\n" + "=" * 50)
        
        advanced_filtering_examples()
        print("\n" + "=" * 50)
        
        news_and_trends_examples()
        print("\n" + "=" * 50)
        
        custom_query_examples()
        print("\n" + "=" * 50)
        
        utility_examples()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your data directory exists and contains data files.")


if __name__ == "__main__":
    main()