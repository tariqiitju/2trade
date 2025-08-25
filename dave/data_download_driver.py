"""
Data Download Driver for Dave Testing Framework

Bulk downloads missing data for all instruments in the base list using
Odin's Eye DataDownloader interface. Provides progress tracking, error
handling, and comprehensive reporting.
"""

import logging
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import time

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from odins_eye import DataDownloader
from consuela.config.instrument_list_loader import load_base_list_all

logger = logging.getLogger(__name__)


class DataDownloadDriver:
    """
    Driver for bulk downloading missing data across all instruments.
    
    Provides comprehensive data acquisition for the trading system including:
    - Market data (OHLCV) for multiple timeframes
    - Economic indicators from FRED
    - Sample news sentiment data
    - Progress tracking and error reporting
    """
    
    def __init__(self, data_root: Optional[str] = None, max_workers: int = 10):
        """
        Initialize download driver.
        
        Args:
            data_root: Root directory for data storage
            max_workers: Maximum concurrent downloads
        """
        self.downloader = DataDownloader(data_root=data_root, max_workers=max_workers)
        self.data_root = Path(self.downloader.data_root)
        
        # Download configuration
        self.config = {
            "market_data": {
                "intervals": ["1d", "1h", "5m"],  # Skip 1m for initial bulk download
                "lookback_years": 5,
                "force_refresh": False
            },
            "economic_data": {
                "enabled": True,
                "lookback_years": 10
            },
            "news_data": {
                "enabled": True,
                "sample_days": 30
            }
        }
        
        # Progress tracking
        self.progress = {
            "total_symbols": 0,
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "current_symbol": None
        }
        
        logger.info("Data download driver initialized")
    
    def load_instrument_list(self, list_name: str = "base-list-all") -> List[Dict[str, Any]]:
        """
        Load instrument list for bulk download.
        
        Args:
            list_name: Name of instrument list to load
            
        Returns:
            List of instrument dictionaries
        """
        try:
            if list_name == "base-list-all":
                instruments = load_base_list_all()
            else:
                # Load other instrument lists
                list_path = Path(__file__).parent.parent / "consuela" / "config" / "instrument_list" / f"{list_name}.yml"
                
                with open(list_path, 'r') as f:
                    data = yaml.safe_load(f)
                    instruments = data.get("instruments", [])
            
            logger.info(f"Loaded {len(instruments)} instruments from {list_name}")
            return instruments
            
        except Exception as e:
            logger.error(f"Failed to load instrument list {list_name}: {e}")
            return []
    
    def filter_instruments(
        self,
        instruments: List[Dict[str, Any]],
        status_filter: Optional[str] = None,
        market_cap_filter: Optional[str] = None,
        sector_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Filter instruments based on criteria.
        
        Args:
            instruments: List of instrument dictionaries
            status_filter: Filter by status (active, delisted, etc.)
            market_cap_filter: Filter by market cap category
            sector_filter: Filter by sector
            limit: Maximum number of symbols to return
            
        Returns:
            List of filtered symbols
        """
        filtered = instruments.copy()
        
        # Apply filters
        if status_filter:
            filtered = [inst for inst in filtered if inst.get("status") == status_filter]
        
        if market_cap_filter:
            filtered = [inst for inst in filtered if inst.get("market_cap_category") == market_cap_filter]
        
        if sector_filter:
            filtered = [inst for inst in filtered if inst.get("sector") == sector_filter]
        
        # Extract symbols
        symbols = [inst["symbol"] for inst in filtered if "symbol" in inst]
        
        # Apply limit
        if limit and len(symbols) > limit:
            symbols = symbols[:limit]
            logger.info(f"Limited to first {limit} symbols")
        
        logger.info(f"Filtered to {len(symbols)} symbols")
        return symbols
    
    def download_all_market_data(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False,
        intervals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Download market data for all specified symbols.
        
        Args:
            symbols: List of symbols (None = load from base list)
            force_refresh: Whether to re-download existing data
            intervals: List of intervals to download
            
        Returns:
            Dict with download results and summary
        """
        try:
            # Load symbols if not provided
            if symbols is None:
                instruments = self.load_instrument_list("base-list-all")
                symbols = self.filter_instruments(instruments, status_filter="active")
            
            if not symbols:
                return {"status": "error", "message": "No symbols to download"}
            
            # Use configured intervals if not specified
            if intervals is None:
                intervals = self.config["market_data"]["intervals"]
            
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 * self.config["market_data"]["lookback_years"])).strftime('%Y-%m-%d')
            
            logger.info(f"Starting market data download for {len(symbols)} symbols")
            logger.info(f"Date range: {start_date} to {end_date}")
            logger.info(f"Intervals: {intervals}")
            
            # Initialize progress tracking
            self.progress.update({
                "total_symbols": len(symbols),
                "completed": 0,
                "successful": 0,
                "failed": 0,
                "start_time": datetime.now(),
                "current_symbol": None
            })
            
            # Progress callback
            def progress_callback(completed, total, symbol, status):
                self.progress["completed"] = completed
                self.progress["current_symbol"] = symbol
                
                if status == "success":
                    self.progress["successful"] += 1
                elif status in ["error", "failed"]:
                    self.progress["failed"] += 1
                
                # Log progress every 10 symbols
                if completed % 10 == 0 or completed == total:
                    elapsed = (datetime.now() - self.progress["start_time"]).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                              f"Success: {self.progress['successful']} "
                              f"Failed: {self.progress['failed']} "
                              f"Rate: {rate:.1f}/s ETA: {eta/60:.1f}min")
            
            # Execute bulk download
            result = self.downloader.download_bulk_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                intervals=intervals,
                force_refresh=force_refresh,
                progress_callback=progress_callback
            )
            
            # Save detailed results
            self._save_download_report(result, "market_data")
            
            return result
            
        except Exception as e:
            logger.error(f"Bulk market data download failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def download_economic_data(self) -> Dict[str, Any]:
        """Download economic indicators from FRED"""
        
        if not self.config["economic_data"]["enabled"]:
            return {"status": "skipped", "message": "Economic data download disabled"}
        
        logger.info("Starting economic data download")
        
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 * self.config["economic_data"]["lookback_years"])).strftime('%Y-%m-%d')
            
            result = self.downloader.download_economic_data(
                start_date=start_date,
                end_date=end_date
            )
            
            # Save results
            self._save_download_report(result, "economic_data")
            
            return result
            
        except Exception as e:
            logger.error(f"Economic data download failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_news_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate sample news sentiment data"""
        
        if not self.config["news_data"]["enabled"]:
            return {"status": "skipped", "message": "News data generation disabled"}
        
        logger.info("Starting sample news data generation")
        
        try:
            # Load symbols if not provided
            if symbols is None:
                instruments = self.load_instrument_list("base-list-all")
                symbols = self.filter_instruments(instruments, status_filter="active", limit=100)  # Limit for news
            
            result = self.downloader.generate_sample_news_data(
                symbols=symbols,
                days_back=self.config["news_data"]["sample_days"]
            )
            
            # Save results
            self._save_download_report(result, "news_data")
            
            return result
            
        except Exception as e:
            logger.error(f"News data generation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_complete_download(
        self,
        symbol_limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete data download process for all data types.
        
        Args:
            symbol_limit: Limit number of symbols for testing
            skip_existing: Whether to skip symbols with existing data
            
        Returns:
            Dict with comprehensive download results
        """
        overall_start = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DATA DOWNLOAD")
        logger.info("=" * 80)
        
        # Load and filter symbols
        instruments = self.load_instrument_list("base-list-all")
        symbols = self.filter_instruments(instruments, status_filter="active", limit=symbol_limit)
        
        logger.info(f"Target symbols: {len(symbols)}")
        
        results = {
            "overall_status": "in_progress",
            "start_time": overall_start.isoformat(),
            "total_symbols": len(symbols),
            "stages": {}
        }
        
        # Stage 1: Market Data Download
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 1: MARKET DATA DOWNLOAD")
        logger.info("=" * 50)
        
        market_result = self.download_all_market_data(
            symbols=symbols,
            force_refresh=not skip_existing
        )
        results["stages"]["market_data"] = market_result
        
        # Stage 2: Economic Data Download
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2: ECONOMIC DATA DOWNLOAD")
        logger.info("=" * 50)
        
        economic_result = self.download_economic_data()
        results["stages"]["economic_data"] = economic_result
        
        # Stage 3: News Data Generation
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 3: NEWS DATA GENERATION")
        logger.info("=" * 50)
        
        news_result = self.generate_news_data(symbols=symbols[:50])  # Limit news to top 50 symbols
        results["stages"]["news_data"] = news_result
        
        # Final Summary
        overall_end = datetime.now()
        overall_duration = (overall_end - overall_start).total_seconds()
        
        results.update({
            "overall_status": "completed",
            "end_time": overall_end.isoformat(),
            "duration_seconds": overall_duration,
            "duration_minutes": overall_duration / 60,
        })
        
        # Generate summary statistics
        market_summary = market_result.get("summary", {})
        economic_summary = economic_result if isinstance(economic_result, dict) else {}
        news_summary = news_result if isinstance(news_result, dict) else {}
        
        total_successful = (
            market_summary.get("successful", 0) +
            economic_summary.get("successful", 0) + 
            news_summary.get("successful", 0)
        )
        
        results["summary"] = {
            "total_downloads": total_successful,
            "market_data_successful": market_summary.get("successful", 0),
            "economic_data_successful": economic_summary.get("successful", 0),
            "news_data_successful": news_summary.get("successful", 0),
            "overall_success_rate": total_successful / len(symbols) if symbols else 0
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("DOWNLOAD COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {overall_duration/60:.1f} minutes")
        logger.info(f"Market Data: {market_summary.get('successful', 0)}/{len(symbols)} symbols")
        logger.info(f"Economic Data: {economic_summary.get('successful', 0)} indicators")
        logger.info(f"News Data: {news_summary.get('successful', 0)} symbols")
        logger.info(f"Overall Success Rate: {results['summary']['overall_success_rate']*100:.1f}%")
        
        # Save comprehensive report
        self._save_download_report(results, "complete_download")
        
        return results
    
    def get_missing_data_analysis(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze what data is missing for given symbols.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dict with missing data analysis
        """
        if symbols is None:
            instruments = self.load_instrument_list("base-list-all")
            symbols = self.filter_instruments(instruments, status_filter="active")
        
        logger.info(f"Analyzing missing data for {len(symbols)} symbols")
        
        analysis = self.downloader.get_missing_data_report(symbols)
        
        # Add recommendations
        analysis["recommendations"] = []
        
        if analysis["summary"]["completely_missing"] > 0:
            analysis["recommendations"].append(
                f"Priority 1: Download data for {analysis['summary']['completely_missing']} symbols with no data"
            )
        
        if analysis["summary"]["partial_data"] > 0:
            analysis["recommendations"].append(
                f"Priority 2: Complete data for {analysis['summary']['partial_data']} symbols with partial data"
            )
        
        if len(analysis["outdated_data"]) > 0:
            analysis["recommendations"].append(
                f"Priority 3: Update {len(analysis['outdated_data'])} outdated datasets"
            )
        
        return analysis
    
    def _save_download_report(self, results: Dict[str, Any], report_type: str) -> None:
        """Save download results to file"""
        try:
            reports_dir = self.data_root / "download_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.json"
            filepath = reports_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Download report saved: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save download report: {e}")


def main():
    """Main function for running data download from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bulk Data Download Driver")
    parser.add_argument("--limit", type=int, help="Limit number of symbols for testing")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download existing data")
    parser.add_argument("--analysis-only", action="store_true", help="Only run missing data analysis")
    parser.add_argument("--data-root", help="Custom data root directory")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize driver
    driver = DataDownloadDriver(data_root=args.data_root, max_workers=args.workers)
    
    if args.analysis_only:
        # Run analysis only
        logger.info("Running missing data analysis...")
        analysis = driver.get_missing_data_analysis()
        
        print("\n" + "=" * 60)
        print("MISSING DATA ANALYSIS")
        print("=" * 60)
        print(f"Total Symbols Analyzed: {analysis['summary']['total_symbols']}")
        print(f"Completely Missing: {analysis['summary']['completely_missing']}")
        print(f"Partial Data: {analysis['summary']['partial_data']}")
        print(f"Complete Data: {analysis['summary']['complete_data']}")
        print(f"Outdated Datasets: {len(analysis['outdated_data'])}")
        
        print("\nRecommendations:")
        for rec in analysis.get("recommendations", []):
            print(f"  • {rec}")
            
    else:
        # Run complete download
        logger.info("Starting complete data download...")
        results = driver.run_complete_download(
            symbol_limit=args.limit,
            skip_existing=not args.force_refresh
        )
        
        if results["overall_status"] == "completed":
            print(f"\n[SUCCESS] Download completed successfully in {results['duration_minutes']:.1f} minutes")
        else:
            print(f"\n[ERROR] Download failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()