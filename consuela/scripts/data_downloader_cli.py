#!/usr/bin/env python3
"""
Data Downloader CLI for 2Trade System

This script provides a command-line interface for downloading missing data
using the enhanced Odin's Eye data downloader.

Author: Consuela Housekeeping Module
Created: 2025-08-24
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from odins_eye.data_downloader import DataDownloader
    from consuela.config.instrument_list_loader import load_favorites_instruments, load_popular_instruments
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_instrument_symbols(list_type: str = "both") -> List[str]:
    """Get instrument symbols from configuration."""
    symbols = []
    
    if list_type in ["favorites", "both"]:
        try:
            favorites = load_favorites_instruments()
            symbols.extend([instr["symbol"] for instr in favorites])
        except Exception as e:
            print(f"Warning: Could not load favorites: {e}")
    
    if list_type in ["popular", "both"]:
        try:
            popular = load_popular_instruments()
            symbols.extend([instr["symbol"] for instr in popular])
        except Exception as e:
            print(f"Warning: Could not load popular instruments: {e}")
    
    return list(set(symbols))


def download_missing_economic_data(downloader: DataDownloader, verbose: bool = False) -> None:
    """Download missing economic indicators."""
    print("Downloading missing economic indicators...")
    
    # Focus on the missing indicators
    missing_indicators = ['INDPRO', 'WPU10170301']  # Industrial Production & Gold Price Index
    
    result = downloader.download_economic_data(fred_symbols=missing_indicators)
    
    if result["status"] == "completed":
        print("Economic data download completed!")
        print(f"   Successful: {result['successful']}/{result['total_symbols']}")
        
        if verbose and "results" in result:
            for symbol, symbol_result in result["results"].items():
                if symbol_result["status"] == "success":
                    print(f"   OK {symbol}: {symbol_result['records']} records")
                else:
                    print(f"   ERROR {symbol}: {symbol_result.get('error', 'Unknown error')}")
    else:
        print(f"Economic data download failed: {result.get('error', 'Unknown error')}")


def download_comprehensive_economic_data(downloader: DataDownloader, categories: List[str] = None,
                                        importance: List[str] = None, max_indicators: int = None,
                                        verbose: bool = False) -> None:
    """Download comprehensive FRED economic data."""
    print("Downloading comprehensive FRED economic data...")
    
    # Show what will be downloaded
    if categories:
        print(f"   Categories: {', '.join(categories)}")
    if importance:
        print(f"   Importance: {', '.join(importance)}")
    if max_indicators:
        print(f"   Limited to: {max_indicators} indicators")
    
    # Get symbol count
    symbols = downloader.get_fred_symbols_by_criteria(categories, importance)
    if max_indicators:
        symbols = symbols[:max_indicators]
    
    print(f"   Found {len(symbols)} indicators matching criteria")
    
    result = downloader.download_comprehensive_fred_data(
        categories=categories,
        importance=importance,
        max_indicators=max_indicators
    )
    
    if result["status"] == "completed":
        print("SUCCESS: Comprehensive economic data download completed!")
        print(f"   Total: {result['total_symbols']} indicators")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
        
        if verbose and "results" in result:
            successful_symbols = []
            failed_symbols = []
            
            for symbol, symbol_result in result["results"].items():
                if symbol_result["status"] == "success":
                    successful_symbols.append(f"{symbol} ({symbol_result['records']} records)")
                else:
                    failed_symbols.append(f"{symbol}: {symbol_result.get('error', 'Unknown error')}")
            
            if successful_symbols:
                print(f"\n   Successful downloads:")
                for item in successful_symbols:
                    print(f"     OK {item}")
            
            if failed_symbols:
                print(f"\n   Failed downloads:")
                for item in failed_symbols:
                    print(f"     ERROR {item}")
    else:
        print(f"ERROR: Comprehensive economic data download failed: {result.get('error', 'Unknown error')}")


def download_corporate_earnings(downloader: DataDownloader, symbols: List[str], source: str = "fmp",
                               earnings_days: int = 90, verbose: bool = False) -> None:
    """Download earnings calendar data for given symbols."""
    print(f"Downloading earnings calendar for {len(symbols)} symbols using {source}...")
    print(f"   Looking ahead {earnings_days} days")
    
    result = downloader.download_earnings_calendar(
        symbols=symbols,
        source=source,
        end_date=(datetime.now() + timedelta(days=earnings_days)).strftime('%Y-%m-%d')
    )
    
    if result["status"] == "completed":
        print(f"SUCCESS: Earnings calendar download completed!")
        print(f"   Source: {result['source']}")
        print(f"   Successful: {result['successful']}/{result['total_symbols']}")
        
        if verbose and "results" in result:
            for symbol, symbol_result in result["results"].items():
                if symbol_result["status"] == "success":
                    print(f"   OK {symbol}: earnings data retrieved")
                else:
                    print(f"   ERROR {symbol}: {symbol_result.get('error', 'Unknown error')}")
    else:
        print(f"ERROR: Earnings calendar download failed: {result.get('error', 'Unknown error')}")


def download_sec_filings(downloader: DataDownloader, symbols: List[str], filing_types: List[str],
                        verbose: bool = False) -> None:
    """Download SEC filings for given symbols."""
    print(f"Downloading SEC filings for {len(symbols)} symbols...")
    print(f"   Filing types: {', '.join(filing_types)}")
    
    result = downloader.download_sec_filings(
        symbols=symbols,
        filing_types=filing_types
    )
    
    if result["status"] == "completed":
        print(f"SUCCESS: SEC filings download completed!")
        print(f"   Successful: {result['successful']}/{result['total_symbols']}")
        
        if verbose and "results" in result:
            for symbol, symbol_result in result["results"].items():
                if symbol_result["status"] == "success":
                    print(f"   OK {symbol}: SEC filings retrieved")
                else:
                    print(f"   ERROR {symbol}: {symbol_result.get('error', 'Unknown error')}")
    else:
        print(f"ERROR: SEC filings download failed: {result.get('error', 'Unknown error')}")


def download_insider_trading(downloader: DataDownloader, symbols: List[str], source: str = "sec_edgar",
                            verbose: bool = False) -> None:
    """Download insider trading data for given symbols."""
    print(f"Downloading insider trading data for {len(symbols)} symbols using {source}...")
    
    result = downloader.download_insider_trading(
        symbols=symbols,
        source=source
    )
    
    if result["status"] == "completed":
        print(f"SUCCESS: Insider trading download completed!")
        print(f"   Source: {result['source']}")
        print(f"   Successful: {result['successful']}/{result['total_symbols']}")
        
        if verbose and "results" in result:
            for symbol, symbol_result in result["results"].items():
                if symbol_result["status"] == "success":
                    print(f"   OK {symbol}: insider trading data retrieved")
                else:
                    print(f"   ERROR {symbol}: {symbol_result.get('error', 'Unknown error')}")
    else:
        print(f"ERROR: Insider trading download failed: {result.get('error', 'Unknown error')}")


def download_news_data(downloader: DataDownloader, symbols: List[str], source: str = "sample", 
                      api_key: str = None, days_back: int = 30, verbose: bool = False) -> None:
    """Download news data for given symbols."""
    print(f"Downloading news data for {len(symbols)} symbols using {source}...")
    
    # Limit to first 10 symbols for API calls to avoid rate limits
    if source != "sample":
        symbols = symbols[:10]
        print(f"   Limited to first {len(symbols)} symbols to avoid API rate limits")
    
    result = downloader.download_news_data(
        symbols=symbols,
        api_key=api_key,
        days_back=days_back,
        source=source
    )
    
    if result["status"] == "completed":
        print(f"SUCCESS: News data download completed!")
        print(f"   Successful: {result['successful']}/{result['total_symbols']}")
        
        if verbose and "results" in result:
            for symbol, symbol_result in result["results"].items():
                if symbol_result["status"] == "success":
                    articles = symbol_result.get('articles', 0)
                    print(f"   OK {symbol}: {articles} articles")
                else:
                    print(f"   ERROR {symbol}: {symbol_result.get('error', 'Unknown error')}")
    else:
        print(f"ERROR: News data download failed: {result.get('error', 'Unknown error')}")


def download_trends_data(downloader: DataDownloader, keywords: List[str], timeframe: str = "today 3-m",
                        verbose: bool = False) -> None:
    """Download Google Trends data for given keywords."""
    print(f"Downloading trends data for {len(keywords)} keywords...")
    
    # Limit to reasonable number to avoid rate limits
    keywords = keywords[:20]
    print(f"   Limited to first {len(keywords)} keywords to avoid rate limits")
    
    result = downloader.download_trends_data(
        keywords=keywords,
        timeframe=timeframe
    )
    
    if result["status"] == "completed":
        print(f"SUCCESS: Trends data download completed!")
        print(f"   Successful: {result['successful']}/{result['total_keywords']}")
        
        if verbose and "results" in result:
            for keyword, keyword_result in result["results"].items():
                if keyword_result["status"] == "success":
                    points = keyword_result.get('data_points', 0)
                    print(f"   OK {keyword}: {points} data points")
                else:
                    print(f"   ERROR {keyword}: {keyword_result.get('error', 'Unknown error')}")
    else:
        print(f"ERROR: Trends data download failed: {result.get('error', 'Unknown error')}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download missing data for 2Trade system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download missing economic indicators
  python consuela/scripts/data_downloader_cli.py --economic
  
  # Download comprehensive FRED data (all high importance)
  python consuela/scripts/data_downloader_cli.py --comprehensive-fred
  
  # Download specific FRED categories
  python consuela/scripts/data_downloader_cli.py --comprehensive-fred --fred-categories employment inflation
  
  # Download all FRED importance levels for interest rates
  python consuela/scripts/data_downloader_cli.py --comprehensive-fred --fred-categories interest_rates --fred-importance high_importance medium_importance
  
  # Test with limited indicators
  python consuela/scripts/data_downloader_cli.py --comprehensive-fred --max-indicators 20
  
  # Download sample news data
  python consuela/scripts/data_downloader_cli.py --news --source sample
  
  # Download real news data (requires API key)
  python consuela/scripts/data_downloader_cli.py --news --source newsapi --api-key YOUR_KEY
  
  # Download trends data
  python consuela/scripts/data_downloader_cli.py --trends
  
  # Download earnings calendar (90 days ahead)
  python consuela/scripts/data_downloader_cli.py --earnings --earnings-source fmp
  
  # Download SEC filings (10-K, 10-Q, 8-K)
  python consuela/scripts/data_downloader_cli.py --sec-filings --filing-types 10-K 10-Q 8-K
  
  # Download insider trading data
  python consuela/scripts/data_downloader_cli.py --insider-trading --insider-source sec_edgar
  
  # Download everything
  python consuela/scripts/data_downloader_cli.py --all
  
Required API Keys (set as environment variables):
  - FRED_API_KEY: For economic data from FRED
  - NEWS_API_KEY: For news data from NewsAPI.org
        """
    )
    
    # Data type options
    parser.add_argument("--economic", action="store_true", help="Download missing economic indicators")
    parser.add_argument("--comprehensive-fred", action="store_true", help="Download comprehensive FRED economic data")
    parser.add_argument("--news", action="store_true", help="Download news data")
    parser.add_argument("--trends", action="store_true", help="Download trends data")
    parser.add_argument("--earnings", action="store_true", help="Download earnings calendar data")
    parser.add_argument("--sec-filings", action="store_true", help="Download SEC filings data") 
    parser.add_argument("--insider-trading", action="store_true", help="Download insider trading data")
    parser.add_argument("--all", action="store_true", help="Download all data types")
    
    # Configuration options
    parser.add_argument("--data-root", type=str, help="Custom data root directory")
    parser.add_argument("--instruments", choices=["favorites", "popular", "both"], 
                       default="favorites", help="Which instrument list to use (default: favorites)")
    
    # News-specific options
    parser.add_argument("--source", choices=["sample", "newsapi", "alpha_vantage"], 
                       default="sample", help="News data source (default: sample)")
    parser.add_argument("--api-key", type=str, help="API key for news/data sources")
    parser.add_argument("--days-back", type=int, default=30, 
                       help="Days of historical news to fetch (default: 30)")
    
    # Trends-specific options  
    parser.add_argument("--timeframe", type=str, default="today 3-m",
                       help="Google Trends timeframe (default: today 3-m)")
    parser.add_argument("--keywords", type=str, nargs="*", 
                       help="Custom keywords for trends (default: use instrument symbols)")
    
    # FRED-specific options
    parser.add_argument("--fred-categories", type=str, nargs="*",
                       help="FRED categories to download (e.g., employment inflation interest_rates)")
    parser.add_argument("--fred-importance", type=str, nargs="*", 
                       choices=["high_importance", "medium_importance", "low_importance"],
                       help="FRED importance levels to include")
    parser.add_argument("--max-indicators", type=int,
                       help="Maximum number of FRED indicators to download (for testing)")
    
    # Corporate data options
    parser.add_argument("--earnings-source", choices=["fmp", "finnhub", "eodhd"], default="fmp",
                       help="Source for earnings data (default: fmp)")
    parser.add_argument("--insider-source", choices=["sec_edgar", "fmp", "finnhub"], default="sec_edgar",
                       help="Source for insider trading data (default: sec_edgar)")
    parser.add_argument("--filing-types", type=str, nargs="*", 
                       choices=["10-K", "10-Q", "8-K", "proxy", "all"],
                       default=["10-K", "10-Q", "8-K"],
                       help="SEC filing types to download")
    parser.add_argument("--earnings-days", type=int, default=90,
                       help="Days ahead to fetch earnings calendar (default: 90)")
    
    # General options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.economic, args.comprehensive_fred, args.news, args.trends, 
                args.earnings, args.sec_filings, args.insider_trading, args.all]):
        parser.error("Must specify at least one data type: --economic, --comprehensive-fred, --news, --trends, --earnings, --sec-filings, --insider-trading, or --all")
    
    # Setup logging
    setup_logging(args.verbose)
    
    if args.dry_run:
        print("DRY RUN MODE - Showing what would be downloaded")
    
    try:
        # Initialize downloader
        print("Initializing Data Downloader...")
        downloader = DataDownloader(data_root=args.data_root)
        print(f"   Data root: {downloader.data_root}")
        
        # Show API key status
        try:
            from consuela.config.api_key_loader import get_api_key_loader
            api_loader = get_api_key_loader()
            validation = api_loader.validate_api_keys()
            
            print("   API Key Status:")
            print(f"     FRED API: {'OK' if validation['fred'] else 'NOT FOUND'}")
            print(f"     NewsAPI:  {'OK' if validation['newsapi'] else 'NOT FOUND'}")
            
            if not validation['fred']:
                print(f"     -> Set FRED API key in consuela/config/api_keys.yml")
            if not validation['newsapi']:
                print(f"     -> Set NewsAPI key in consuela/config/api_keys.yml")
                
        except Exception as e:
            print(f"   Warning: Could not check API key status: {e}")
        
        # Get instrument symbols for various data types
        if any([args.news, args.trends, args.earnings, args.sec_filings, args.insider_trading, args.all]):
            symbols = get_instrument_symbols(args.instruments)
            print(f"   Loaded {len(symbols)} instruments from {args.instruments} list")
        else:
            symbols = []
        
        if args.dry_run:
            print("\nWould download:")
            if args.economic or args.all:
                print("   * Missing economic indicators: INDPRO, WPU10170301")
            if args.comprehensive_fred or args.all:
                # Preview comprehensive FRED download
                fred_symbols = downloader.get_fred_symbols_by_criteria(
                    categories=args.fred_categories, 
                    importance=args.fred_importance
                )
                if args.max_indicators:
                    fred_symbols = fred_symbols[:args.max_indicators]
                categories_str = f" (categories: {', '.join(args.fred_categories)})" if args.fred_categories else ""
                importance_str = f" (importance: {', '.join(args.fred_importance)})" if args.fred_importance else ""
                print(f"   * Comprehensive FRED data: {len(fred_symbols)} indicators{categories_str}{importance_str}")
            if args.news or args.all:
                print(f"   * News data for {len(symbols) if 'symbols' in locals() else 0} symbols using {args.source}")
            if args.trends or args.all:
                keywords = args.keywords or symbols[:20] if 'symbols' in locals() else []
                print(f"   * Trends data for {len(keywords)} keywords")
            if args.earnings or args.all:
                print(f"   * Earnings calendar for {len(symbols) if 'symbols' in locals() else 0} symbols using {args.earnings_source} ({args.earnings_days} days ahead)")
            if args.sec_filings or args.all:
                filing_types_str = ', '.join(args.filing_types) if args.filing_types != ['all'] else 'all types'
                print(f"   * SEC filings for {len(symbols) if 'symbols' in locals() else 0} symbols ({filing_types_str})")
            if args.insider_trading or args.all:
                print(f"   * Insider trading for {len(symbols) if 'symbols' in locals() else 0} symbols using {args.insider_source}")
            print("\nRun without --dry-run to actually download data.")
            return
        
        # Download data
        print("\nStarting data download process...")
        
        # 1. Economic data
        if args.economic or args.all:
            download_missing_economic_data(downloader, args.verbose)
            print()
        
        # 1b. Comprehensive FRED data
        if args.comprehensive_fred or args.all:
            download_comprehensive_economic_data(
                downloader, 
                categories=args.fred_categories,
                importance=args.fred_importance,
                max_indicators=args.max_indicators,
                verbose=args.verbose
            )
            print()
        
        # 2. News data
        if args.news or args.all:
            download_news_data(
                downloader, symbols, args.source, args.api_key, args.days_back, args.verbose
            )
            print()
        
        # 3. Trends data
        if args.trends or args.all:
            keywords = args.keywords or symbols
            download_trends_data(downloader, keywords, args.timeframe, args.verbose)
            print()
        
        # 4. Corporate earnings data
        if args.earnings or args.all:
            download_corporate_earnings(
                downloader, symbols, args.earnings_source, args.earnings_days, args.verbose
            )
            print()
        
        # 5. SEC filings data
        if args.sec_filings or args.all:
            filing_types = args.filing_types if args.filing_types != ['all'] else ["10-K", "10-Q", "8-K", "proxy"]
            download_sec_filings(downloader, symbols, filing_types, args.verbose)
            print()
        
        # 6. Insider trading data
        if args.insider_trading or args.all:
            download_insider_trading(downloader, symbols, args.insider_source, args.verbose)
            print()
        
        print("SUCCESS: Data download process completed!")
        print(f"   Check your data directory: {downloader.data_root}")
        
        # Suggest running data quality evaluation
        print(f"\nTIP: Next steps:")
        print(f"   Run data quality evaluation to see the updated coverage:")
        print(f"   python consuela/scripts/data_quality_evaluator.py --quick --verbose")
        
    except KeyboardInterrupt:
        print("\nWARNING:  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()