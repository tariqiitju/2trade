#!/usr/bin/env python3
"""
Odin's Eye Library Test Driver

Comprehensive test suite for the Odin's Eye financial data access library.
Uses instrument lists from consuela/config for realistic testing scenarios.
"""

import sys
import os
import yaml
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add Odin's Eye to Python path
import os
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Added to Python path: {project_root}")
print(f"Looking for odins_eye in: {project_root / 'odins_eye'}")

try:
    from odins_eye import OdinsEye, DateRange, MarketDataInterval
    from odins_eye.filters import DataType, InstrumentFilter, DataTypeFilter, QueryFilter
    from odins_eye.exceptions import OdinsEyeError, DataNotFoundError, ConfigurationError
except ImportError as e:
    print(f"ERROR: Could not import Odin's Eye library: {e}")
    print("Make sure the library is properly installed and in the Python path.")
    sys.exit(1)


class OdinsEyeTestDriver:
    """Comprehensive test driver for Odin's Eye library"""
    
    def __init__(self, data_root: Optional[str] = None):
        """Initialize test driver with optional custom data root"""
        self.data_root = data_root
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": []
        }
        
        # Load instrument lists
        self.instruments = self._load_instruments()
        
        # Initialize Odin's Eye
        try:
            self.eye = OdinsEye(data_root=data_root) if data_root else OdinsEye()
            print(f"[OK] Initialized Odin's Eye with data root: {self.eye.data_root}")
        except Exception as e:
            print(f"[FAIL] Failed to initialize Odin's Eye: {e}")
            raise
    
    def _load_instruments(self) -> Dict[str, List[Dict]]:
        """Load instrument lists from consuela/config"""
        instruments = {"favorites": [], "popular": []}
        
        try:
            # Load favorites
            favorites_path = Path(__file__).parent.parent.parent.parent / "consuela" / "config" / "instrument-list" / "favorites_instruments.yml"
            if favorites_path.exists():
                with open(favorites_path, 'r', encoding='utf-8') as f:
                    favorites_data = yaml.safe_load(f)
                    instruments["favorites"] = favorites_data.get("favorites", [])
                    print(f"[OK] Loaded {len(instruments['favorites'])} favorite instruments")
            
            # Load popular
            popular_path = Path(__file__).parent.parent.parent.parent / "consuela" / "config" / "instrument-list" / "popular_instruments.yml"
            if popular_path.exists():
                with open(popular_path, 'r', encoding='utf-8') as f:
                    popular_data = yaml.safe_load(f)
                    instruments["popular"] = popular_data.get("instruments", [])
                    print(f"[OK] Loaded {len(instruments['popular'])} popular instruments")
        
        except Exception as e:
            self._add_warning(f"Failed to load instrument lists: {e}")
        
        return instruments
    
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
    
    def get_test_symbols(self, count: int = 10, symbol_type: str = "stock") -> List[str]:
        """Get test symbols from instrument lists"""
        symbols = []
        
        # Get symbols from favorites first
        for instrument in self.instruments.get("favorites", []):
            if len(symbols) >= count:
                break
            if instrument.get("type") == symbol_type:
                symbols.append(instrument["symbol"])
        
        # Fill remaining from popular if needed
        for instrument in self.instruments.get("popular", []):
            if len(symbols) >= count:
                break
            if instrument.get("type") == symbol_type and instrument["symbol"] not in symbols:
                symbols.append(instrument["symbol"])
        
        return symbols[:count]
    
    def test_basic_initialization(self):
        """Test basic library initialization"""
        print("\n" + "="*60)
        print("TESTING: Basic Initialization")
        print("="*60)
        
        try:
            # Test default initialization
            eye = OdinsEye()
            self._pass_test("Default initialization")
            
            # Test custom data root
            if self.data_root:
                eye_custom = OdinsEye(data_root=self.data_root)
                self._pass_test("Custom data root initialization")
            
            # Test data info
            info = self.eye.get_data_info()
            if isinstance(info, dict):
                self._pass_test("Get data info")
                print(f"   Data root: {info.get('data_root')}")
                for data_type, details in info.items():
                    if data_type != "data_root" and isinstance(details, dict):
                        print(f"   {data_type}: {details}")
            else:
                self._fail_test("Get data info", "Invalid return type")
                
        except Exception as e:
            self._fail_test("Basic initialization", str(e))
    
    def test_market_data_access(self):
        """Test market data access methods"""
        print("\n" + "="*60)
        print("TESTING: Market Data Access")
        print("="*60)
        
        symbols = self.get_test_symbols(5, "stock")
        if not symbols:
            self._add_warning("No stock symbols available for testing")
            return
        
        print(f"Testing with symbols: {symbols}")
        
        # Test single symbol daily data
        try:
            symbol = symbols[0]
            data = self.eye.get_market_data(symbol)
            if data is not None:
                self._pass_test(f"Get daily data for {symbol}")
                print(f"   Records: {len(data)} rows")
                if not data.empty and 'symbol' in data.columns:
                    print(f"   Columns: {list(data.columns)}")
                    print(f"   Date range: {data['date'].min() if 'date' in data.columns else 'N/A'} to {data['date'].max() if 'date' in data.columns else 'N/A'}")
            else:
                self._fail_test(f"Get daily data for {symbol}", "Returned None")
        except Exception as e:
            self._fail_test(f"Get daily data for {symbol}", str(e))
        
        # Test multiple symbols
        try:
            data = self.eye.get_market_data(symbols[:3])
            if data is not None:
                self._pass_test("Get data for multiple symbols")
                print(f"   Records: {len(data)} rows")
                if not data.empty and 'symbol' in data.columns:
                    unique_symbols = data['symbol'].unique() if 'symbol' in data.columns else []
                    print(f"   Symbols found: {list(unique_symbols)}")
            else:
                self._fail_test("Get data for multiple symbols", "Returned None")
        except Exception as e:
            self._fail_test("Get data for multiple symbols", str(e))
        
        # Test different intervals
        for interval in [MarketDataInterval.DAILY, MarketDataInterval.MIN_1, MarketDataInterval.HOURLY]:
            try:
                data = self.eye.get_market_data(symbols[0], interval=interval)
                if data is not None:
                    self._pass_test(f"Get {interval.value} data")
                    print(f"   Records: {len(data)} rows")
                else:
                    self._add_warning(f"No {interval.value} data available")
            except Exception as e:
                self._add_warning(f"Get {interval.value} data failed: {e}")
    
    def test_date_filtering(self):
        """Test date range filtering"""
        print("\n" + "="*60)
        print("TESTING: Date Range Filtering")
        print("="*60)
        
        symbols = self.get_test_symbols(2, "stock")
        if not symbols:
            self._add_warning("No symbols available for date filtering tests")
            return
        
        symbol = symbols[0]
        
        # Test recent data (last 30 days)
        try:
            start_date = datetime.now() - timedelta(days=30)
            date_range = DateRange(start_date=start_date)
            
            data = self.eye.get_market_data(symbol, date_range=date_range)
            if data is not None:
                self._pass_test("Recent data filtering (last 30 days)")
                print(f"   Records: {len(data)} rows")
            else:
                self._add_warning("No recent data available")
        except Exception as e:
            self._fail_test("Recent data filtering", str(e))
        
        # Test specific date range
        try:
            date_range = DateRange(
                start_date="2023-01-01",
                end_date="2023-12-31"
            )
            
            data = self.eye.get_market_data(symbol, date_range=date_range)
            if data is not None:
                self._pass_test("Specific date range (2023)")
                print(f"   Records: {len(data)} rows")
            else:
                self._add_warning("No 2023 data available")
        except Exception as e:
            self._fail_test("Specific date range filtering", str(e))
    
    def test_economic_data_access(self):
        """Test economic data access"""
        print("\n" + "="*60)
        print("TESTING: Economic Data Access")  
        print("="*60)
        
        # Test common economic indicators
        indicators = ["UNRATE", "DGS10", "VIXCLS", "GDP"]
        
        for indicator in indicators:
            try:
                data = self.eye.get_economic_data(indicator)
                if data is not None and not data.empty:
                    self._pass_test(f"Get economic data for {indicator}")
                    print(f"   Records: {len(data)} rows")
                    if 'date' in data.columns:
                        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
                else:
                    self._add_warning(f"No data available for {indicator}")
            except Exception as e:
                self._add_warning(f"Economic data for {indicator} failed: {e}")
        
        # Test multiple indicators
        try:
            data = self.eye.get_economic_data(indicators[:2])
            if data is not None:
                self._pass_test("Get multiple economic indicators")
                print(f"   Records: {len(data)} rows")
            else:
                self._add_warning("No data for multiple indicators")
        except Exception as e:
            self._fail_test("Multiple economic indicators", str(e))
    
    def test_news_data_access(self):
        """Test news data access"""
        print("\n" + "="*60)
        print("TESTING: News Data Access")
        print("="*60)
        
        symbols = self.get_test_symbols(2, "stock")
        if not symbols:
            self._add_warning("No symbols available for news tests")
            return
        
        # Test news for specific symbol
        try:
            symbol = symbols[0]
            news = self.eye.get_news_data(
                symbols=symbol,
                date_range=DateRange(start_date="2024-01-01")
            )
            
            if news is not None:
                self._pass_test(f"Get news data for {symbol}")
                print(f"   Articles: {len(news)}")
                if news:
                    print(f"   Sample: {news[0].get('title', 'No title')[:50]}...")
            else:
                self._add_warning(f"No news data for {symbol}")
        except Exception as e:
            self._add_warning(f"News data access failed: {e}")
        
        # Test sentiment filtering
        try:
            news = self.eye.get_news_data(
                sentiment_label="positive",
                date_range=DateRange(start_date="2024-08-01")
            )
            
            if news is not None:
                self._pass_test("Get positive sentiment news")
                print(f"   Positive articles: {len(news)}")
            else:
                self._add_warning("No positive sentiment news available")
        except Exception as e:
            self._add_warning(f"Sentiment filtering failed: {e}")
    
    def test_trends_data_access(self):
        """Test Google Trends data access"""
        print("\n" + "="*60)
        print("TESTING: Google Trends Data Access")
        print("="*60)
        
        symbols = self.get_test_symbols(2, "stock")
        keywords = ["stock market", "recession", "inflation"]
        
        # Test trends for symbols
        if symbols:
            try:
                symbol = symbols[0]
                trends = self.eye.get_trends_data(
                    keywords=symbol,
                    date_range=DateRange(start_date="2024-01-01")
                )
                
                if trends is not None:
                    self._pass_test(f"Get trends data for {symbol}")
                    print(f"   Trends entries: {len(trends)}")
                else:
                    self._add_warning(f"No trends data for {symbol}")
            except Exception as e:
                self._add_warning(f"Symbol trends data failed: {e}")
        
        # Test trends for keywords
        try:
            trends = self.eye.get_trends_data(
                keywords=keywords[:2],
                geo="US"
            )
            
            if trends is not None:
                self._pass_test("Get trends data for keywords")
                print(f"   Keyword trends: {len(trends)}")
            else:
                self._add_warning("No keyword trends data available")
        except Exception as e:
            self._add_warning(f"Keyword trends failed: {e}")
    
    def test_custom_queries(self):
        """Test custom query functionality"""
        print("\n" + "="*60)
        print("TESTING: Custom Queries")
        print("="*60)
        
        symbols = self.get_test_symbols(3, "stock")
        if not symbols:
            self._add_warning("No symbols available for custom query tests")
            return
        
        # Test complex market data query
        try:
            query_filter = QueryFilter(
                date_range=DateRange(start_date="2024-01-01"),
                instruments=InstrumentFilter(symbols=symbols[:2]),
                data_types=[DataTypeFilter(
                    data_type=DataType.MARKET_DATA,
                    interval=MarketDataInterval.DAILY,
                    min_volume=100000
                )],
                limit=100,
                sort_by="date",
                sort_order="desc"
            )
            
            results = self.eye.query(query_filter)
            if results is not None:
                self._pass_test("Complex market data query")
                print(f"   Results: {len(results)} rows")
            else:
                self._add_warning("Complex query returned no results")
        except Exception as e:
            self._fail_test("Complex market data query", str(e))
    
    def test_utility_functions(self):
        """Test utility functions"""
        print("\n" + "="*60)
        print("TESTING: Utility Functions")
        print("="*60)
        
        # Test list available symbols
        try:
            symbols = self.eye.list_available_symbols(DataType.MARKET_DATA, MarketDataInterval.DAILY)
            if symbols:
                self._pass_test("List available market data symbols")
                print(f"   Available symbols: {len(symbols)}")
                print(f"   Sample: {symbols[:10]}")
            else:
                self._add_warning("No market data symbols available")
        except Exception as e:
            self._fail_test("List available symbols", str(e))
        
        # Test economic indicators
        try:
            indicators = self.eye.list_available_symbols(DataType.ECONOMIC_DATA)
            if indicators:
                self._pass_test("List available economic indicators")
                print(f"   Available indicators: {len(indicators)}")
                print(f"   Sample: {indicators[:5]}")
            else:
                self._add_warning("No economic indicators available")
        except Exception as e:
            self._add_warning(f"List economic indicators failed: {e}")
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n" + "="*60)
        print("TESTING: Error Handling")
        print("="*60)
        
        # Test invalid symbol
        try:
            data = self.eye.get_market_data("INVALID_SYMBOL_XYZ")
            if data is not None and data.empty:
                self._pass_test("Handle invalid symbol gracefully")
            else:
                self._add_warning("Invalid symbol handling unclear")
        except Exception as e:
            self._add_warning(f"Invalid symbol test: {e}")
        
        # Test invalid date range
        try:
            invalid_range = DateRange(
                start_date="2025-01-01",
                end_date="2024-01-01"  # End before start
            )
            data = self.eye.get_market_data("AAPL", date_range=invalid_range)
            self._add_warning("Invalid date range not caught")
        except Exception as e:
            self._pass_test("Catch invalid date range")
    
    def run_all_tests(self):
        """Run all tests"""
        print("ODIN'S EYE LIBRARY TEST DRIVER")
        print("=" * 80)
        print(f"Start time: {datetime.now()}")
        print(f"Data root: {self.eye.data_root}")
        print(f"Test instruments loaded: {len(self.instruments.get('favorites', []))} favorites, {len(self.instruments.get('popular', []))} popular")
        
        # Run test suites
        test_suites = [
            self.test_basic_initialization,
            self.test_market_data_access,
            self.test_date_filtering,
            self.test_economic_data_access,
            self.test_news_data_access,
            self.test_trends_data_access,
            self.test_custom_queries,
            self.test_utility_functions,
            self.test_error_handling
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
    
    parser = argparse.ArgumentParser(description="Odin's Eye Library Test Driver")
    parser.add_argument("--data-root", help="Custom data root directory")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize test driver
        driver = OdinsEyeTestDriver(data_root=args.data_root)
        
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