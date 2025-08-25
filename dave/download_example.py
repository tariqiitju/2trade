#!/usr/bin/env python3
"""
Example usage of the Data Download system.

This script demonstrates how to use the data download functionality
with practical examples for different use cases.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dave.data_download_driver import DataDownloadDriver
from consuela.config.instrument_list_loader import (
    get_active_symbols, get_large_cap_symbols, get_etf_symbols,
    get_symbols_by_sector, get_instrument_stats
)


def example_quick_test():
    """Example: Quick test with a few symbols"""
    print("=" * 60)
    print("EXAMPLE 1: Quick Test Download")
    print("=" * 60)
    
    driver = DataDownloadDriver()
    
    # Download data for just 5 symbols to test
    test_symbols = ["AAPL", "MSFT", "SPY", "VOO", "QQQ"]
    
    print(f"Downloading data for: {test_symbols}")
    
    result = driver.download_all_market_data(
        symbols=test_symbols,
        intervals=["1d"],  # Daily data only for speed
        force_refresh=False
    )
    
    print(f"Result: {result['summary']['successful']}/{len(test_symbols)} successful")


def example_sector_download():
    """Example: Download data for all Technology stocks"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Technology Sector Download")
    print("=" * 60)
    
    driver = DataDownloadDriver()
    
    # Get all Technology sector symbols (limit to 20 for example)
    tech_symbols = get_symbols_by_sector("Technology", limit=20)
    
    print(f"Found {len(tech_symbols)} Technology symbols")
    print(f"Downloading: {tech_symbols[:5]}... (showing first 5)")
    
    result = driver.download_all_market_data(
        symbols=tech_symbols,
        intervals=["1d", "1h"],  # Daily and hourly
        force_refresh=False
    )
    
    summary = result["summary"]
    print(f"Technology sector download completed:")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Duration: {summary['duration_seconds']:.1f} seconds")


def example_etf_download():
    """Example: Download ETF data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: ETF Download")
    print("=" * 60)
    
    driver = DataDownloadDriver()
    
    # Get ETF symbols
    etf_symbols = get_etf_symbols(limit=15)
    
    print(f"Found {len(etf_symbols)} ETF symbols")
    print(f"ETFs: {etf_symbols}")
    
    result = driver.download_all_market_data(
        symbols=etf_symbols,
        intervals=["1d"],  # Daily data for ETFs
        force_refresh=False
    )
    
    print(f"ETF download completed: {result['summary']['successful']} successful")


def example_missing_data_analysis():
    """Example: Analyze what data is missing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Missing Data Analysis")
    print("=" * 60)
    
    driver = DataDownloadDriver()
    
    # Analyze large cap stocks
    large_cap_symbols = get_large_cap_symbols(limit=50)
    
    print(f"Analyzing missing data for {len(large_cap_symbols)} large cap stocks...")
    
    analysis = driver.get_missing_data_analysis(large_cap_symbols)
    
    print("\nMissing Data Analysis Results:")
    print(f"  Total Symbols: {analysis['summary']['total_symbols']}")
    print(f"  Completely Missing: {analysis['summary']['completely_missing']}")
    print(f"  Partial Data: {analysis['summary']['partial_data']}")
    print(f"  Complete Data: {analysis['summary']['complete_data']}")
    print(f"  Outdated Data: {len(analysis.get('outdated_data', []))}")
    
    # Show missing by interval
    for interval, missing_symbols in analysis.get('missing_by_interval', {}).items():
        if missing_symbols:
            print(f"  Missing {interval}: {len(missing_symbols)} symbols")
    
    # Show recommendations
    print("\nRecommendations:")
    for rec in analysis.get('recommendations', []):
        print(f"  • {rec}")


def example_instrument_database_stats():
    """Example: Show instrument database statistics"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Instrument Database Statistics")
    print("=" * 60)
    
    stats = get_instrument_stats()
    
    print("Instrument Database Overview:")
    print(f"  Total Instruments: {stats['total_instruments']}")
    print(f"  Active: {stats['active_count']}")
    print(f"  Delisted: {stats['delisted_count']}")
    
    print(f"\nBy Market Cap:")
    for cap_type, count in sorted(stats['by_market_cap'].items()):
        print(f"  {cap_type}: {count}")
    
    print(f"\nTop Sectors:")
    top_sectors = sorted(stats['by_sector'].items(), key=lambda x: x[1], reverse=True)[:10]
    for sector, count in top_sectors:
        print(f"  {sector}: {count}")
    
    print(f"\nBy Exchange:")
    for exchange, count in sorted(stats['by_exchange'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {exchange}: {count}")


def example_custom_download():
    """Example: Custom download with specific parameters"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Custom Download Parameters")
    print("=" * 60)
    
    driver = DataDownloadDriver(max_workers=5)  # Custom worker count
    
    # Custom symbol list
    custom_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # Big Tech
        "SPY", "QQQ", "IWM",                       # Major ETFs
        "JPM", "BAC", "GS",                       # Major Banks
        "JNJ", "PFE", "UNH"                       # Healthcare
    ]
    
    print(f"Custom download for {len(custom_symbols)} selected symbols")
    
    # Custom configuration - just daily data, last 2 years
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    result = driver.downloader.download_bulk_market_data(
        symbols=custom_symbols,
        start_date=start_date,
        intervals=["1d"],
        force_refresh=False
    )
    
    print(f"Custom download completed:")
    print(f"  Duration: {result['summary']['duration_seconds']:.1f}s")
    print(f"  Rate: {result['summary']['symbols_per_second']:.1f} symbols/second")
    print(f"  Success rate: {result['summary']['successful']/len(custom_symbols)*100:.1f}%")


def main():
    """Run all examples"""
    print("🚀 DATA DOWNLOAD EXAMPLES")
    print("This demonstrates various ways to use the data download system")
    
    try:
        # Start with database stats to show what's available
        example_instrument_database_stats()
        
        # Run missing data analysis first
        example_missing_data_analysis()
        
        # Then run various download examples
        example_quick_test()
        example_sector_download()
        example_etf_download()
        example_custom_download()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\nNext Steps:")
        print("  1. For full bulk download: python dave/data_download_driver.py")
        print("  2. For testing with limited symbols: python dave/data_download_driver.py --limit 100")
        print("  3. For missing data analysis only: python dave/data_download_driver.py --analysis-only")
        print("  4. For testing download functionality: python dave/test_download.py")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("\nTroubleshooting:")
        print("  • Install dependencies: pip install -r consuela/config/requirements.txt")
        print("  • Check network connectivity")
        print("  • Ensure data directory is writable")


if __name__ == "__main__":
    main()