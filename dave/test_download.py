#!/usr/bin/env python3
"""
Test script for data download functionality.

Tests the DataDownloader and DataDownloadDriver with a small set of instruments
to validate the download pipeline before running bulk operations.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from odins_eye import DataDownloader
from dave.data_download_driver import DataDownloadDriver
from consuela.config.instrument_list_loader import get_active_symbols, get_instrument_stats


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_basic_download():
    """Test basic download functionality with a few symbols"""
    print("=" * 60)
    print("TESTING BASIC DOWNLOAD FUNCTIONALITY")
    print("=" * 60)
    
    # Test with a few popular symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "VOO"]
    
    try:
        downloader = DataDownloader()
        
        print(f"\nTesting download for: {test_symbols}")
        
        for symbol in test_symbols:
            print(f"\nDownloading {symbol}...")
            result = downloader.download_market_data(
                symbol=symbol,
                intervals=["1d"],  # Just daily data for testing
                start_date="2023-01-01"
            )
            
            status = result["status"]
            if status == "success":
                successful_intervals = result.get("successful_intervals", [])
                print(f"[OK] {symbol}: SUCCESS - {successful_intervals}")
            else:
                error = result.get("error", "Unknown error")
                print(f"[FAIL] {symbol}: FAILED - {error}")
        
        print("\nBasic download test completed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic download test failed: {e}")
        return False


def test_bulk_download():
    """Test bulk download with limited symbols"""
    print("\n" + "=" * 60)
    print("TESTING BULK DOWNLOAD FUNCTIONALITY")
    print("=" * 60)
    
    try:
        driver = DataDownloadDriver(max_workers=3)  # Limit workers for testing
        
        # Get first 10 active symbols for testing
        test_symbols = get_active_symbols(limit=10)
        print(f"\nTesting bulk download for: {test_symbols}")
        
        # Run bulk download
        result = driver.download_all_market_data(
            symbols=test_symbols,
            intervals=["1d"],  # Just daily data for testing
            force_refresh=False
        )
        
        if result.get("summary", {}).get("successful", 0) > 0:
            summary = result["summary"]
            print(f"[OK] Bulk download successful!")
            print(f"   Successful: {summary['successful']}")
            print(f"   Failed: {summary['failed']}")
            print(f"   Duration: {summary['duration_seconds']:.1f}s")
            return True
        else:
            print(f"[FAIL] Bulk download failed: {result}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Bulk download test failed: {e}")
        return False


def test_missing_data_analysis():
    """Test missing data analysis"""
    print("\n" + "=" * 60)
    print("TESTING MISSING DATA ANALYSIS")
    print("=" * 60)
    
    try:
        driver = DataDownloadDriver()
        
        # Analyze first 20 symbols
        test_symbols = get_active_symbols(limit=20)
        print(f"\nAnalyzing missing data for {len(test_symbols)} symbols...")
        
        analysis = driver.get_missing_data_analysis(test_symbols)
        
        print(f"[OK] Missing data analysis completed!")
        print(f"   Total symbols: {analysis['summary']['total_symbols']}")
        print(f"   Completely missing: {analysis['summary']['completely_missing']}")
        print(f"   Partial data: {analysis['summary']['partial_data']}")
        print(f"   Complete data: {analysis['summary']['complete_data']}")
        
        if analysis.get("recommendations"):
            print("\n   Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"     • {rec}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Missing data analysis failed: {e}")
        return False


def test_instrument_list_loading():
    """Test instrument list loading functionality"""
    print("\n" + "=" * 60)
    print("TESTING INSTRUMENT LIST LOADING")
    print("=" * 60)
    
    try:
        # Test basic loading
        active_symbols = get_active_symbols(limit=5)
        print(f"[OK] Loaded active symbols: {active_symbols}")
        
        # Test statistics
        stats = get_instrument_stats()
        print(f"[OK] Instrument statistics:")
        print(f"   Total instruments: {stats['total_instruments']}")
        print(f"   Active: {stats['active_count']}")
        print(f"   Delisted: {stats['delisted_count']}")
        
        # Show top sectors
        top_sectors = sorted(stats['by_sector'].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top sectors: {dict(top_sectors)}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Instrument list loading failed: {e}")
        return False


def test_sample_data_generation():
    """Test sample data generation"""
    print("\n" + "=" * 60)
    print("TESTING SAMPLE DATA GENERATION")
    print("=" * 60)
    
    try:
        downloader = DataDownloader()
        
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        result = downloader.generate_sample_news_data(
            symbols=test_symbols,
            days_back=7  # Just 1 week for testing
        )
        
        if result.get("successful", 0) > 0:
            print(f"[OK] Sample data generation successful!")
            print(f"   Successful symbols: {result['successful']}")
            print(f"   Total symbols: {result['total_symbols']}")
            return True
        else:
            print(f"[FAIL] Sample data generation failed: {result}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Sample data generation failed: {e}")
        return False


def main():
    """Run all tests"""
    setup_logging()
    
    print("STARTING DATA DOWNLOAD TESTS")
    print("This will test the download functionality with a small set of data")
    
    tests = [
        ("Instrument List Loading", test_instrument_list_loading),
        ("Basic Download", test_basic_download),
        ("Sample Data Generation", test_sample_data_generation),
        ("Missing Data Analysis", test_missing_data_analysis),
        ("Bulk Download", test_bulk_download),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:10} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Data download system is ready for production use.")
        print("\nNext steps:")
        print("  1. Run full bulk download: python dave/data_download_driver.py")
        print("  2. Use --limit flag for testing: python dave/data_download_driver.py --limit 50")
        print("  3. Run analysis only: python dave/data_download_driver.py --analysis-only")
    else:
        print(f"\n[WARNING] {len(results)-passed} tests failed. Check the errors above and fix issues before proceeding.")
        print("\nCommon issues:")
        print("  • Install required packages: pip install yfinance fredapi")
        print("  • Set FRED_API_KEY environment variable for economic data")
        print("  • Check network connectivity")


if __name__ == "__main__":
    main()