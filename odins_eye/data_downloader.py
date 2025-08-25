"""
Data Downloader for Odin's Eye

Provides interfaces for downloading missing market data, economic data,
news sentiment, and Google Trends data for specified instruments.
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

from .core import OdinsEye
from .exceptions import DataNotFoundError, ConfigurationError

logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Downloads and stores market data, economic indicators, news sentiment,
    and Google Trends data for specified instruments.
    """
    
    def __init__(self, data_root: Optional[str] = None, max_workers: int = 5):
        """
        Initialize data downloader.
        
        Args:
            data_root: Root directory for data storage
            max_workers: Maximum concurrent download threads
        """
        self.odins_eye = OdinsEye(data_root=data_root)
        self.data_root = Path(self.odins_eye.data_root)
        self.max_workers = max_workers
        
        # Download statistics
        self.download_stats = {
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        # Load comprehensive FRED indicators
        self.fred_indicators = self._load_fred_indicators()
        
        # Load corporate data sources configuration
        self.corporate_data_sources = self._load_corporate_data_sources()
        
        logger.info(f"Data downloader initialized with data_root: {self.data_root}")
    
    def _load_fred_indicators(self) -> Dict[str, List[Dict]]:
        """Load comprehensive FRED indicators from configuration file."""
        try:
            # Try to find the configuration file
            config_paths = [
                self.data_root.parent / "consuela" / "config" / "comprehensive_fred_indicators.yml",
                Path.cwd() / "consuela" / "config" / "comprehensive_fred_indicators.yml",
                Path(__file__).parent.parent / "consuela" / "config" / "comprehensive_fred_indicators.yml"
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if not config_file:
                logger.warning("Comprehensive FRED indicators config not found, using basic set")
                return self._get_basic_fred_indicators()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                indicators = yaml.safe_load(f)
            
            logger.info(f"Loaded comprehensive FRED indicators from {config_file}")
            return indicators
            
        except Exception as e:
            logger.warning(f"Failed to load FRED indicators config: {e}, using basic set")
            return self._get_basic_fred_indicators()
    
    def _get_basic_fred_indicators(self) -> Dict[str, List[Dict]]:
        """Get basic FRED indicators as fallback."""
        return {
            "employment": {
                "high_importance": [
                    {"symbol": "UNRATE", "name": "Unemployment Rate", "frequency": "monthly"},
                    {"symbol": "PAYEMS", "name": "Total Nonfarm Payrolls", "frequency": "monthly"},
                ]
            },
            "inflation": {
                "high_importance": [
                    {"symbol": "CPIAUCSL", "name": "Consumer Price Index", "frequency": "monthly"},
                ]
            },
            "interest_rates": {
                "high_importance": [
                    {"symbol": "FEDFUNDS", "name": "Federal Funds Rate", "frequency": "monthly"},
                    {"symbol": "DGS10", "name": "10-Year Treasury Rate", "frequency": "daily"},
                ]
            },
            "growth": {
                "high_importance": [
                    {"symbol": "GDP", "name": "Gross Domestic Product", "frequency": "quarterly"},
                ]
            }
        }
    
    def get_fred_symbols_by_criteria(self, categories: Optional[List[str]] = None, 
                                   importance: Optional[List[str]] = None) -> List[str]:
        """
        Get FRED symbols filtered by category and importance.
        
        Args:
            categories: List of categories to include (e.g., ['employment', 'inflation'])
            importance: List of importance levels to include (e.g., ['high_importance', 'medium_importance'])
            
        Returns:
            List of FRED symbols matching criteria
        """
        symbols = []
        
        for category, importance_levels in self.fred_indicators.items():
            # Filter by category
            if categories and category not in categories:
                continue
            
            for imp_level, indicators in importance_levels.items():
                # Filter by importance
                if importance and imp_level not in importance:
                    continue
                
                for indicator in indicators:
                    symbols.append(indicator['symbol'])
        
        return list(set(symbols))  # Remove duplicates
    
    def _load_corporate_data_sources(self) -> Dict[str, Any]:
        """Load corporate data sources configuration."""
        try:
            # Try to find the configuration file
            config_paths = [
                self.data_root.parent / "consuela" / "config" / "corporate_data_sources.yml",
                Path.cwd() / "consuela" / "config" / "corporate_data_sources.yml",
                Path(__file__).parent.parent / "consuela" / "config" / "corporate_data_sources.yml"
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if not config_file:
                logger.warning("Corporate data sources config not found, using minimal set")
                return self._get_minimal_corporate_sources()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                sources = yaml.safe_load(f)
            
            logger.info(f"Loaded corporate data sources from {config_file}")
            return sources
            
        except Exception as e:
            logger.warning(f"Failed to load corporate data sources config: {e}")
            return self._get_minimal_corporate_sources()
    
    def _get_minimal_corporate_sources(self) -> Dict[str, Any]:
        """Get minimal corporate data sources as fallback."""
        return {
            "sec_edgar": {
                "base_url": "https://data.sec.gov",
                "description": "SEC EDGAR API for filings",
                "data_types": ["sec_filings"]
            }
        }
    
    def download_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        intervals: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Download market data for a single instrument.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format  
            intervals: List of intervals to download ['1d', '1h', '5m', etc.]
            force_refresh: Whether to re-download existing data
            
        Returns:
            Dict with download results and metadata
        """
        try:
            logger.info(f"Downloading market data for {symbol}")
            
            # Set default parameters
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if intervals is None:
                intervals = ['1d', '1h', '5m', '1m']  # Default intervals
            
            results = {}
            
            for interval in intervals:
                try:
                    result = self._download_single_interval(
                        symbol, start_date, end_date, interval, force_refresh
                    )
                    results[interval] = result
                    
                    # Rate limiting to avoid being blocked
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to download {symbol} {interval}: {e}")
                    results[interval] = {"status": "error", "error": str(e)}
            
            # Determine overall status
            successful_intervals = [k for k, v in results.items() if v.get("status") == "success"]
            
            if successful_intervals:
                self.download_stats["successful"] += 1
                status = "success"
            else:
                self.download_stats["failed"] += 1
                status = "failed"
            
            return {
                "symbol": symbol,
                "status": status,
                "intervals": results,
                "successful_intervals": successful_intervals,
                "download_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to download market data for {symbol}: {e}")
            self.download_stats["failed"] += 1
            self.download_stats["errors"].append(f"{symbol}: {str(e)}")
            
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "download_date": datetime.now().isoformat()
            }
    
    def _download_single_interval(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        force_refresh: bool
    ) -> Dict[str, Any]:
        """Download data for a single symbol and interval"""
        
        # Check if data already exists
        output_dir = self.data_root / "market_data" / interval
        output_file = output_dir / f"{symbol}.parquet"
        
        if output_file.exists() and not force_refresh:
            # Check if data is recent enough
            existing_data = pd.read_parquet(output_file)
            if not existing_data.empty:
                last_date = pd.to_datetime(existing_data.index.max()).date()
                today = datetime.now().date()
                
                if (today - last_date).days <= 1:  # Data is up to date
                    return {
                        "status": "skipped", 
                        "reason": "data_current",
                        "last_date": str(last_date),
                        "records": len(existing_data)
                    }
        
        # Download new data
        ticker = yf.Ticker(symbol)
        
        try:
            # Map interval to yfinance format
            yf_interval_map = {
                '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m', '60m': '60m',
                '1h': '1h', '1d': '1d', '5d': '5d', '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
            }
            
            yf_interval = yf_interval_map.get(interval, interval)
            
            # Download data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,  # Get adjusted prices
                prepost=True      # Include pre/post market
            )
            
            if data.empty:
                return {
                    "status": "error",
                    "error": f"No data returned for {symbol} {interval}"
                }
            
            # Clean and prepare data
            data = self._clean_market_data(data, symbol)
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            data.to_parquet(output_file)
            
            logger.info(f"Saved {len(data)} records for {symbol} {interval}")
            
            return {
                "status": "success",
                "records": len(data),
                "date_range": [str(data.index.min().date()), str(data.index.max().date())],
                "file_path": str(output_file)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _clean_market_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize market data"""
        
        # Standardize column names
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for {symbol}: {missing_cols}")
        
        # Add adjusted close if not present
        if 'adj_close' not in data.columns:
            data['adj_close'] = data['close']
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Ensure high >= low and close/open within high/low bounds
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Fix any data inconsistencies
            data['high'] = data[['high', 'open', 'close']].max(axis=1)
            data['low'] = data[['low', 'open', 'close']].min(axis=1)
        
        # Ensure positive volumes
        if 'volume' in data.columns:
            data['volume'] = data['volume'].clip(lower=0)
        
        # Sort by index (datetime)
        data = data.sort_index()
        
        return data
    
    def download_bulk_market_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        intervals: Optional[List[str]] = None,
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Download market data for multiple instruments in parallel.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            intervals: List of intervals to download
            force_refresh: Whether to re-download existing data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with bulk download results
        """
        logger.info(f"Starting bulk download for {len(symbols)} symbols")
        
        # Reset stats
        self.download_stats = {"successful": 0, "failed": 0, "skipped": 0, "errors": []}
        
        results = {}
        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(
                    self.download_market_data,
                    symbol, start_date, end_date, intervals, force_refresh
                ): symbol
                for symbol in symbols
            }
            
            # Process completed downloads
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                
                try:
                    result = future.result()
                    results[symbol] = result
                    
                    if progress_callback:
                        progress_callback(i + 1, len(symbols), symbol, result["status"])
                    
                    logger.info(f"Completed {symbol}: {result['status']} ({i+1}/{len(symbols)})")
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol}: {e}")
                    results[symbol] = {"status": "error", "error": str(e)}
                    self.download_stats["errors"].append(f"{symbol}: {str(e)}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile summary
        summary = {
            "total_symbols": len(symbols),
            "successful": self.download_stats["successful"],
            "failed": self.download_stats["failed"],
            "skipped": self.download_stats["skipped"],
            "duration_seconds": duration,
            "symbols_per_second": len(symbols) / duration if duration > 0 else 0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "errors": self.download_stats["errors"][:10]  # First 10 errors
        }
        
        logger.info(f"Bulk download completed: {summary}")
        
        return {
            "summary": summary,
            "results": results
        }
    
    def download_economic_data(
        self,
        fred_symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download economic data from FRED (Federal Reserve Economic Data).
        
        Args:
            fred_symbols: List of FRED symbols to download
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict with download results
        """
        try:
            import fredapi
        except ImportError:
            logger.error("fredapi not installed. Install with: pip install fredapi")
            return {"status": "error", "error": "fredapi not installed"}
        
        # Default economic indicators - use high importance indicators
        if fred_symbols is None:
            fred_symbols = self.get_fred_symbols_by_criteria(importance=['high_importance'])
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        results = {}
        
        try:
            # Get FRED API key from configuration
            from consuela.config.api_key_loader import get_fred_api_key
            fred_api_key = get_fred_api_key()
            
            if not fred_api_key:
                return {
                    "status": "error",
                    "error": "FRED API key not found. Set in consuela/config/api_keys.yml, environment variable FRED_API_KEY, or pass as parameter."
                }
            
            # Initialize FRED API
            fred = fredapi.Fred(api_key=fred_api_key)
            
            economic_dir = self.data_root / "economic_data"
            economic_dir.mkdir(parents=True, exist_ok=True)
            
            for symbol in fred_symbols:
                try:
                    data = fred.get_series(symbol, start=start_date, end=end_date)
                    
                    if not data.empty:
                        # Get metadata from FRED
                        try:
                            info = fred.get_series_info(symbol)
                        except:
                            info = {}
                        
                        # Convert to DataFrame with metadata
                        df = pd.DataFrame({
                            'date': data.index,
                            'value': data.values,
                            'symbol': symbol,
                            'indicator_name': info.get('title', symbol),
                            'category': self._classify_economic_indicator(symbol),
                            'frequency': info.get('frequency', 'Unknown'),
                            'importance': self._get_indicator_importance(symbol),
                            'description': info.get('title', f"{symbol} economic indicator")
                        })
                        
                        # Save to parquet
                        output_file = economic_dir / f"{symbol}.parquet"
                        df.to_parquet(output_file, index=False)
                        
                        results[symbol] = {
                            "status": "success",
                            "records": len(df),
                            "file_path": str(output_file)
                        }
                        
                        logger.info(f"Downloaded {len(df)} records for {symbol}")
                    else:
                        results[symbol] = {"status": "error", "error": "No data returned"}
                        
                except Exception as e:
                    logger.warning(f"Failed to download {symbol}: {e}")
                    results[symbol] = {"status": "error", "error": str(e)}
                
                # Rate limiting
                time.sleep(0.1)
            
            successful = sum(1 for r in results.values() if r["status"] == "success")
            
            return {
                "status": "completed",
                "total_symbols": len(fred_symbols),
                "successful": successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize FRED API: {e}")
            return {
                "status": "error", 
                "error": f"FRED API error: {str(e)}. Set FRED_API_KEY environment variable."
            }
    
    def download_comprehensive_fred_data(
        self,
        categories: Optional[List[str]] = None,
        importance: Optional[List[str]] = None,
        max_indicators: Optional[int] = None,
        batch_size: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rate_limit_delay: float = 0.1
    ) -> Dict[str, Any]:
        """
        Download comprehensive FRED economic data with advanced filtering and batching.
        
        Args:
            categories: Economic categories to download (e.g., ['employment', 'inflation'])
            importance: Importance levels to include (e.g., ['high_importance', 'medium_importance'])
            max_indicators: Maximum number of indicators to download (for testing)
            batch_size: Number of indicators to download per batch
            start_date: Start date for data collection
            end_date: End date for data collection
            rate_limit_delay: Delay between API calls to avoid rate limiting
            
        Returns:
            Dict with comprehensive download results
        """
        try:
            import fredapi
        except ImportError:
            logger.error("fredapi not installed. Install with: pip install fredapi")
            return {"status": "error", "error": "fredapi not installed"}
        
        # Get filtered symbols
        fred_symbols = self.get_fred_symbols_by_criteria(categories, importance)
        
        if max_indicators:
            fred_symbols = fred_symbols[:max_indicators]
        
        if not fred_symbols:
            return {"status": "error", "error": "No FRED symbols found matching criteria"}
        
        logger.info(f"Starting comprehensive FRED download: {len(fred_symbols)} indicators")
        if categories:
            logger.info(f"  Categories: {categories}")
        if importance:
            logger.info(f"  Importance: {importance}")
        
        # Download in batches
        results = {}
        total_successful = 0
        
        for i in range(0, len(fred_symbols), batch_size):
            batch_symbols = fred_symbols[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(fred_symbols) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_symbols)} indicators")
            
            # Download batch
            batch_result = self.download_economic_data(
                fred_symbols=batch_symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Merge results
            if "results" in batch_result:
                results.update(batch_result["results"])
                total_successful += batch_result.get("successful", 0)
            
            # Rate limiting
            if i + batch_size < len(fred_symbols):
                time.sleep(rate_limit_delay)
        
        return {
            "status": "completed",
            "total_symbols": len(fred_symbols),
            "successful": total_successful,
            "failed": len(fred_symbols) - total_successful,
            "results": results,
            "categories_downloaded": categories or "all",
            "importance_levels": importance or "all"
        }
    
    def download_earnings_calendar(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = "fmp"
    ) -> Dict[str, Any]:
        """
        Download earnings calendar data for specified symbols.
        
        Args:
            symbols: List of stock symbols to download earnings for
            start_date: Start date for earnings data
            end_date: End date for earnings data
            source: Data source to use (fmp, finnhub, eodhd)
            
        Returns:
            Dict with download results
        """
        if symbols is None:
            symbols = []
        
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
        
        logger.info(f"Downloading earnings calendar for {len(symbols)} symbols from {source}")
        
        # Create earnings data directory
        earnings_dir = self.data_root / "earnings_data"
        earnings_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        if source == "fmp":
            results = self._download_earnings_from_fmp(symbols, start_date, end_date, earnings_dir)
        elif source == "finnhub":
            results = self._download_earnings_from_finnhub(symbols, start_date, end_date, earnings_dir)
        else:
            return {"status": "error", "error": f"Unsupported earnings source: {source}"}
        
        successful = sum(1 for r in results.values() if r["status"] == "success")
        
        return {
            "status": "completed",
            "source": source,
            "total_symbols": len(symbols),
            "successful": successful,
            "results": results
        }
    
    def download_sec_filings(
        self,
        symbols: List[str],
        filing_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Download SEC filings for specified symbols.
        
        Args:
            symbols: List of stock symbols
            filing_types: List of filing types (10-K, 10-Q, 8-K, etc.)
            
        Returns:
            Dict with download results
        """
        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]
        
        logger.info(f"Downloading SEC filings for {len(symbols)} symbols, types: {filing_types}")
        
        # Create SEC filings directory
        sec_dir = self.data_root / "sec_filings"
        sec_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for symbol in symbols:
            try:
                # Get CIK for symbol (simplified - in real implementation would use mapping)
                # This is a placeholder - would need to implement CIK lookup
                cik = self._get_cik_for_symbol(symbol)
                if not cik:
                    results[symbol] = {"status": "error", "error": "CIK not found"}
                    continue
                
                # Download filings for this symbol
                symbol_results = self._download_sec_filings_for_symbol(symbol, cik, filing_types, sec_dir)
                results[symbol] = symbol_results
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to download SEC filings for {symbol}: {e}")
                results[symbol] = {"status": "error", "error": str(e)}
        
        successful = sum(1 for r in results.values() if r["status"] == "success")
        
        return {
            "status": "completed",
            "total_symbols": len(symbols),
            "successful": successful,
            "results": results
        }
    
    def download_insider_trading(
        self,
        symbols: List[str],
        source: str = "sec_edgar"
    ) -> Dict[str, Any]:
        """
        Download insider trading data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            source: Data source (sec_edgar, fmp, finnhub)
            
        Returns:
            Dict with download results
        """
        logger.info(f"Downloading insider trading data for {len(symbols)} symbols from {source}")
        
        # Create insider trading directory
        insider_dir = self.data_root / "insider_trading"
        insider_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for symbol in symbols:
            try:
                if source == "sec_edgar":
                    result = self._download_insider_from_sec(symbol, insider_dir)
                elif source == "fmp":
                    result = self._download_insider_from_fmp(symbol, insider_dir)
                elif source == "finnhub":
                    result = self._download_insider_from_finnhub(symbol, insider_dir)
                else:
                    result = {"status": "error", "error": f"Unsupported source: {source}"}
                
                results[symbol] = result
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to download insider trading for {symbol}: {e}")
                results[symbol] = {"status": "error", "error": str(e)}
        
        successful = sum(1 for r in results.values() if r["status"] == "success")
        
        return {
            "status": "completed",
            "source": source,
            "total_symbols": len(symbols),
            "successful": successful,
            "results": results
        }
    
    def _download_earnings_from_fmp(self, symbols: List[str], start_date: str, 
                                   end_date: str, earnings_dir: Path) -> Dict[str, Any]:
        """Download earnings data from Financial Modeling Prep."""
        try:
            # Get FMP API key
            from consuela.config.api_key_loader import get_api_key_loader
            api_loader = get_api_key_loader()
            fmp_key = api_loader.get_api_key('fmp')
            
            if not fmp_key:
                logger.error("FMP API key not found")
                results = {}
                for symbol in symbols:
                    results[symbol] = {"status": "error", "error": "FMP API key not configured"}
                return results
            
            results = {}
            base_url = "https://financialmodelingprep.com/api/v3"
            
            # Download earnings calendar (general calendar, not per symbol)
            calendar_url = f"{base_url}/earning_calendar"
            params = {
                "from": start_date,
                "to": end_date,
                "apikey": fmp_key
            }
            
            logger.info(f"Downloading FMP earnings calendar from {start_date} to {end_date}")
            
            response = requests.get(calendar_url, params=params)
            if response.status_code == 200:
                calendar_data = response.json()
                
                # Filter for our symbols and save
                filtered_earnings = []
                symbol_set = set(symbols)
                
                for earning in calendar_data:
                    if earning.get('symbol') in symbol_set:
                        filtered_earnings.append(earning)
                
                # Save earnings calendar data
                calendar_file = earnings_dir / f"earnings_calendar_{start_date}_{end_date}.json"
                with open(calendar_file, 'w') as f:
                    json.dump(filtered_earnings, f, indent=2)
                
                # Create results for each symbol
                found_symbols = set(earning.get('symbol') for earning in filtered_earnings)
                for symbol in symbols:
                    if symbol in found_symbols:
                        symbol_earnings = [e for e in filtered_earnings if e.get('symbol') == symbol]
                        results[symbol] = {
                            "status": "success",
                            "earnings_count": len(symbol_earnings),
                            "file_path": str(calendar_file)
                        }
                    else:
                        results[symbol] = {
                            "status": "no_data", 
                            "error": "No earnings data found for this symbol in date range"
                        }
                
                logger.info(f"Downloaded earnings data for {len(found_symbols)} symbols")
                
            else:
                logger.error(f"FMP API request failed: {response.status_code}")
                for symbol in symbols:
                    results[symbol] = {
                        "status": "error", 
                        "error": f"FMP API error: {response.status_code}"
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"FMP earnings download failed: {e}")
            results = {}
            for symbol in symbols:
                results[symbol] = {"status": "error", "error": str(e)}
            return results
    
    def _download_earnings_from_finnhub(self, symbols: List[str], start_date: str,
                                       end_date: str, earnings_dir: Path) -> Dict[str, Any]:
        """Download earnings data from Finnhub."""
        try:
            # Get Finnhub API key
            from consuela.config.api_key_loader import get_api_key_loader
            api_loader = get_api_key_loader()
            finnhub_key = api_loader.get_api_key('finnhub')
            
            if not finnhub_key:
                logger.error("Finnhub API key not found")
                results = {}
                for symbol in symbols:
                    results[symbol] = {"status": "error", "error": "Finnhub API key not configured"}
                return results
            
            results = {}
            base_url = "https://finnhub.io/api/v1"
            
            # Download earnings calendar
            calendar_url = f"{base_url}/calendar/earnings"
            params = {
                "from": start_date,
                "to": end_date,
                "token": finnhub_key
            }
            
            logger.info(f"Downloading Finnhub earnings calendar from {start_date} to {end_date}")
            
            response = requests.get(calendar_url, params=params)
            if response.status_code == 200:
                calendar_data = response.json()
                
                # Extract earnings announcements
                earnings_announcements = calendar_data.get('earningsAnnouncements', [])
                
                # Filter for our symbols and save
                filtered_earnings = []
                symbol_set = set(symbols)
                
                for earning in earnings_announcements:
                    if earning.get('symbol') in symbol_set:
                        filtered_earnings.append(earning)
                
                # Save earnings calendar data
                calendar_file = earnings_dir / f"finnhub_earnings_calendar_{start_date}_{end_date}.json"
                with open(calendar_file, 'w') as f:
                    json.dump(filtered_earnings, f, indent=2)
                
                # Create results for each symbol
                found_symbols = set(earning.get('symbol') for earning in filtered_earnings)
                for symbol in symbols:
                    if symbol in found_symbols:
                        symbol_earnings = [e for e in filtered_earnings if e.get('symbol') == symbol]
                        results[symbol] = {
                            "status": "success",
                            "earnings_count": len(symbol_earnings),
                            "file_path": str(calendar_file)
                        }
                    else:
                        results[symbol] = {
                            "status": "no_data", 
                            "error": "No earnings data found for this symbol in date range"
                        }
                
                logger.info(f"Downloaded Finnhub earnings data for {len(found_symbols)} symbols")
                
            else:
                logger.error(f"Finnhub API request failed: {response.status_code}")
                for symbol in symbols:
                    results[symbol] = {
                        "status": "error", 
                        "error": f"Finnhub API error: {response.status_code} - {response.text[:100]}"
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Finnhub earnings download failed: {e}")
            results = {}
            for symbol in symbols:
                results[symbol] = {"status": "error", "error": str(e)}
            return results
    
    def _get_cik_for_symbol(self, symbol: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a stock symbol."""
        # Placeholder - would implement CIK lookup logic
        # Could use SEC's company tickers JSON file
        return None
    
    def _download_sec_filings_for_symbol(self, symbol: str, cik: str, 
                                        filing_types: List[str], sec_dir: Path) -> Dict[str, Any]:
        """Download SEC filings for a specific symbol."""
        # Placeholder implementation
        return {"status": "placeholder", "error": "SEC EDGAR implementation needed"}
    
    def _download_insider_from_sec(self, symbol: str, insider_dir: Path) -> Dict[str, Any]:
        """Download insider trading from SEC EDGAR."""
        # Placeholder implementation
        return {"status": "placeholder", "error": "SEC insider trading implementation needed"}
    
    def _download_insider_from_fmp(self, symbol: str, insider_dir: Path) -> Dict[str, Any]:
        """Download insider trading from Financial Modeling Prep."""
        # Placeholder implementation
        return {"status": "placeholder", "error": "FMP insider trading implementation needed"}
    
    def _download_insider_from_finnhub(self, symbol: str, insider_dir: Path) -> Dict[str, Any]:
        """Download insider trading from Finnhub."""
        # Placeholder implementation
        return {"status": "placeholder", "error": "Finnhub insider trading implementation needed"}
    
    def _classify_economic_indicator(self, symbol: str) -> str:
        """Classify economic indicator by category."""
        categories = {
            'employment': ['UNRATE', 'PAYEMS', 'CIVPART'],
            'interest_rates': ['FEDFUNDS', 'DGS10', 'DGS2', 'DFF'],
            'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI'],
            'growth': ['GDP', 'GDPC1'],
            'markets': ['VIXCLS', 'DEXUSEU', 'DTWEXBGS'],
            'commodities': ['DCOILWTICO', 'GOLDAMGBD228NLBM'],
            'industrial': ['INDPRO']
        }
        
        for category, symbols in categories.items():
            if symbol in symbols:
                return category
        return 'other'
    
    def _get_indicator_importance(self, symbol: str) -> str:
        """Get importance level of economic indicator."""
        high_importance = ['UNRATE', 'GDP', 'FEDFUNDS', 'CPIAUCSL', 'PAYEMS', 'DGS10', 'VIXCLS']
        medium_importance = ['DEXUSEU', 'DCOILWTICO', 'INDPRO', 'DGS2']
        
        if symbol in high_importance:
            return 'high'
        elif symbol in medium_importance:
            return 'medium'
        else:
            return 'low'
    
    def download_news_data(
        self,
        symbols: List[str],
        api_key: Optional[str] = None,
        days_back: int = 30,
        source: str = "newsapi"
    ) -> Dict[str, Any]:
        """
        Download real financial news data for given symbols.
        
        Args:
            symbols: List of stock symbols
            api_key: News API key (or use environment variable)
            days_back: Number of days to fetch news for
            source: News source ('newsapi', 'alpha_vantage', or 'sample')
            
        Returns:
            Dict with download results
        """
        if source == "sample":
            return self.generate_sample_news_data(symbols, days_back)
        
        if source == "newsapi":
            return self._download_newsapi_data(symbols, api_key, days_back)
        elif source == "alpha_vantage":
            return self._download_alphavantage_news(symbols, api_key, days_back)
        else:
            return {"status": "error", "error": f"Unsupported news source: {source}"}
    
    def _download_newsapi_data(
        self,
        symbols: List[str],
        api_key: Optional[str],
        days_back: int
    ) -> Dict[str, Any]:
        """Download news from NewsAPI.org"""
        try:
            from newsapi import NewsApiClient
        except ImportError:
            logger.error("newsapi-python not installed. Install with: pip install newsapi-python")
            return {"status": "error", "error": "newsapi-python not installed"}
        
        # Get News API key from configuration
        from consuela.config.api_key_loader import get_news_api_key
        api_key = get_news_api_key('newsapi', api_key)
        
        if not api_key:
            return {
                "status": "error", 
                "error": "NewsAPI key not found. Set in consuela/config/api_keys.yml, environment variable NEWS_API_KEY, or pass as parameter."
            }
        
        try:
            newsapi = NewsApiClient(api_key=api_key)
            news_dir = self.data_root / "news_data"
            news_dir.mkdir(parents=True, exist_ok=True)
            
            results = {}
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            for symbol in symbols:
                try:
                    logger.info(f"Downloading news for {symbol}")
                    
                    # Search for news about this symbol
                    articles = newsapi.get_everything(
                        q=f'"{symbol}" OR "{self._get_company_name(symbol)}"',
                        language='en',
                        from_param=start_date.strftime('%Y-%m-%d'),
                        to=end_date.strftime('%Y-%m-%d'),
                        sort_by='publishedAt',
                        page_size=100
                    )
                    
                    if articles['articles']:
                        # Process and save articles
                        processed_articles = []
                        for article in articles['articles']:
                            processed_article = {
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'content': article.get('content', ''),
                                'source': article.get('source', {}).get('name', ''),
                                'author': article.get('author', ''),
                                'url': article.get('url', ''),
                                'published_at': article.get('publishedAt', ''),
                                'symbols': [symbol],
                                'sentiment_score': self._calculate_sentiment(article.get('title', '') + ' ' + article.get('description', '')),
                                'sentiment_label': '',  # Will be set by sentiment calculation
                                'category': 'company',
                                'article_id': abs(hash(article.get('url', '') + article.get('publishedAt', ''))) % (10**16)
                            }
                            
                            # Set sentiment label
                            score = processed_article['sentiment_score']
                            processed_article['sentiment_label'] = (
                                'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
                            )
                            
                            processed_articles.append(processed_article)
                        
                        # Save to JSON file
                        today = datetime.now().strftime('%Y%m%d')
                        output_file = news_dir / f"news_{today}.json"
                        
                        # Append to existing file or create new
                        existing_articles = []
                        if output_file.exists():
                            with open(output_file, 'r') as f:
                                existing_articles = json.load(f)
                        
                        all_articles = existing_articles + processed_articles
                        
                        with open(output_file, 'w') as f:
                            json.dump(all_articles, f, indent=2, default=str)
                        
                        results[symbol] = {
                            "status": "success",
                            "articles": len(processed_articles),
                            "file_path": str(output_file)
                        }
                        
                        logger.info(f"Downloaded {len(processed_articles)} news articles for {symbol}")
                    else:
                        results[symbol] = {"status": "success", "articles": 0}
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to download news for {symbol}: {e}")
                    results[symbol] = {"status": "error", "error": str(e)}
            
            successful = sum(1 for r in results.values() if r["status"] == "success")
            
            return {
                "status": "completed",
                "total_symbols": len(symbols),
                "successful": successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for better news search."""
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'NFLX': 'Netflix'
        }
        return company_names.get(symbol, symbol)
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text."""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except ImportError:
            # Simple keyword-based sentiment if TextBlob not available
            positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'growth']
            negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'decline', 'drop']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.0
            return (pos_count - neg_count) / (pos_count + neg_count)
    
    def download_trends_data(
        self,
        keywords: List[str],
        timeframe: str = 'today 3-m',
        geo: str = 'US'
    ) -> Dict[str, Any]:
        """
        Download Google Trends data for given keywords/symbols.
        
        Args:
            keywords: List of keywords or stock symbols to track
            timeframe: Google Trends timeframe ('today 1-m', 'today 3-m', etc.)
            geo: Geographic region (default: 'US')
            
        Returns:
            Dict with download results
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.error("pytrends not installed. Install with: pip install pytrends")
            return {"status": "error", "error": "pytrends not installed"}
        
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            trends_dir = self.data_root / "trends_data"
            trends_dir.mkdir(parents=True, exist_ok=True)
            
            results = {}
            
            # Process keywords in batches of 5 (Google Trends limit)
            for i in range(0, len(keywords), 5):
                batch = keywords[i:i+5]
                
                try:
                    # Build payload
                    pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo, gprop='')
                    
                    # Get interest over time
                    interest_df = pytrends.interest_over_time()
                    
                    if not interest_df.empty:
                        # Get related queries
                        related_queries = pytrends.related_queries()
                        
                        # Process each keyword in batch
                        for keyword in batch:
                            if keyword in interest_df.columns:
                                # Create trends data structure
                                trends_data = {
                                    'keyword': keyword,
                                    'symbol': keyword if keyword in self._get_all_symbols() else '',
                                    'timeframe': timeframe,
                                    'geo': geo,
                                    'category': 0,
                                    'search_interest': [
                                        {
                                            'date': date.strftime('%Y-%m-%d'),
                                            keyword: int(value) if pd.notna(value) else 0
                                        }
                                        for date, value in interest_df[keyword].items()
                                    ],
                                    'related_queries': {
                                        'top': related_queries.get(keyword, {}).get('top', pd.DataFrame()).to_dict('records') if related_queries.get(keyword, {}).get('top') is not None else [],
                                        'rising': related_queries.get(keyword, {}).get('rising', pd.DataFrame()).to_dict('records') if related_queries.get(keyword, {}).get('rising') is not None else []
                                    }
                                }
                                
                                # Save to JSON file
                                today = datetime.now().strftime('%Y%m%d')
                                output_file = trends_dir / f"{keyword}_{today}.json"
                                
                                with open(output_file, 'w') as f:
                                    json.dump(trends_data, f, indent=2, default=str)
                                
                                results[keyword] = {
                                    "status": "success",
                                    "data_points": len(trends_data['search_interest']),
                                    "file_path": str(output_file)
                                }
                                
                                logger.info(f"Downloaded trends data for {keyword}")
                            else:
                                results[keyword] = {"status": "error", "error": "No data returned"}
                    
                    # Rate limiting to avoid being blocked
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Failed to download trends for batch {batch}: {e}")
                    for keyword in batch:
                        results[keyword] = {"status": "error", "error": str(e)}
            
            successful = sum(1 for r in results.values() if r["status"] == "success")
            
            return {
                "status": "completed",
                "total_keywords": len(keywords),
                "successful": successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Google Trends API error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_all_symbols(self) -> List[str]:
        """Get all known stock symbols for validation."""
        try:
            from consuela.config.instrument_list_loader import load_favorites_instruments, load_popular_instruments
            favorites = load_favorites_instruments()
            popular = load_popular_instruments()
            return list(set([instr["symbol"] for instr in favorites + popular]))
        except:
            return []
    
    def generate_sample_news_data(
        self,
        symbols: List[str],
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Generate sample news sentiment data for testing.
        
        Args:
            symbols: List of symbols to generate news for
            days_back: Number of days to generate data for
            
        Returns:
            Dict with generation results
        """
        logger.info(f"Generating sample news data for {len(symbols)} symbols")
        
        news_dir = self.data_root / "news_data"
        news_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generate sample data for each symbol
        for symbol in symbols:
            try:
                dates = pd.date_range(
                    end=datetime.now().date(),
                    periods=days_back,
                    freq='D'
                )
                
                # Generate synthetic news sentiment data
                np.random.seed(hash(symbol) % 2**32)  # Reproducible per symbol
                
                news_data = []
                for date in dates:
                    # Generate 0-5 news articles per day
                    num_articles = np.random.poisson(2)
                    
                    for i in range(num_articles):
                        sentiment_score = np.random.normal(0.05, 0.3)  # Slight positive bias
                        sentiment_score = np.clip(sentiment_score, -1, 1)
                        
                        article = {
                            "date": date.strftime('%Y-%m-%d'),
                            "symbol": symbol,
                            "headline": f"Sample news headline {i+1} for {symbol}",
                            "sentiment_score": sentiment_score,
                            "sentiment_label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral",
                            "source": "sample_news_generator",
                            "article_id": f"{symbol}_{date.strftime('%Y%m%d')}_{i}",
                            "confidence": np.random.uniform(0.6, 0.95)
                        }
                        
                        news_data.append(article)
                
                if news_data:
                    # Save as JSON
                    output_file = news_dir / f"{symbol}_news.json"
                    with open(output_file, 'w') as f:
                        json.dump(news_data, f, indent=2, default=str)
                    
                    results[symbol] = {
                        "status": "success",
                        "articles": len(news_data),
                        "file_path": str(output_file)
                    }
                else:
                    results[symbol] = {"status": "error", "error": "No data generated"}
                    
            except Exception as e:
                logger.error(f"Failed to generate news data for {symbol}: {e}")
                results[symbol] = {"status": "error", "error": str(e)}
        
        successful = sum(1 for r in results.values() if r["status"] == "success")
        
        return {
            "status": "completed",
            "total_symbols": len(symbols),
            "successful": successful,
            "results": results
        }
    
    def get_download_status(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check download status for given symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dict with status information for each symbol
        """
        status_info = {}
        
        for symbol in symbols:
            symbol_status = {
                "market_data": {},
                "news_data": False,
                "last_updated": None
            }
            
            # Check market data intervals
            for interval in ['1d', '1h', '5m', '1m']:
                file_path = self.data_root / "market_data" / interval / f"{symbol}.parquet"
                
                if file_path.exists():
                    try:
                        data = pd.read_parquet(file_path)
                        symbol_status["market_data"][interval] = {
                            "exists": True,
                            "records": len(data),
                            "date_range": [str(data.index.min().date()), str(data.index.max().date())],
                            "last_updated": file_path.stat().st_mtime
                        }
                        
                        # Update overall last_updated
                        if (symbol_status["last_updated"] is None or 
                            file_path.stat().st_mtime > symbol_status["last_updated"]):
                            symbol_status["last_updated"] = file_path.stat().st_mtime
                            
                    except Exception as e:
                        symbol_status["market_data"][interval] = {
                            "exists": True,
                            "error": str(e)
                        }
                else:
                    symbol_status["market_data"][interval] = {"exists": False}
            
            # Check news data
            news_file = self.data_root / "news_data" / f"{symbol}_news.json"
            if news_file.exists():
                symbol_status["news_data"] = True
            
            status_info[symbol] = symbol_status
        
        return status_info
    
    def get_missing_data_report(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Generate a report of missing data for given symbols.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dict with missing data analysis
        """
        status = self.get_download_status(symbols)
        
        missing_report = {
            "summary": {
                "total_symbols": len(symbols),
                "completely_missing": 0,
                "partial_data": 0,
                "complete_data": 0
            },
            "missing_by_interval": {
                "1d": [],
                "1h": [],
                "5m": [], 
                "1m": []
            },
            "missing_news": [],
            "outdated_data": []
        }
        
        today = datetime.now().date()
        
        for symbol, symbol_status in status.items():
            market_data = symbol_status["market_data"]
            
            # Check if any market data exists
            has_any_data = any(interval_data.get("exists", False) 
                              for interval_data in market_data.values())
            
            if not has_any_data:
                missing_report["summary"]["completely_missing"] += 1
            else:
                # Check each interval
                intervals_with_data = sum(1 for interval_data in market_data.values() 
                                        if interval_data.get("exists", False))
                
                if intervals_with_data == len(market_data):
                    missing_report["summary"]["complete_data"] += 1
                else:
                    missing_report["summary"]["partial_data"] += 1
                
                # Track missing intervals
                for interval, interval_data in market_data.items():
                    if not interval_data.get("exists", False):
                        missing_report["missing_by_interval"][interval].append(symbol)
                    elif "date_range" in interval_data:
                        # Check if data is outdated (more than 2 days old)
                        last_date = pd.to_datetime(interval_data["date_range"][1]).date()
                        if (today - last_date).days > 2:
                            missing_report["outdated_data"].append({
                                "symbol": symbol,
                                "interval": interval,
                                "last_date": str(last_date),
                                "days_behind": (today - last_date).days
                            })
            
            # Check news data
            if not symbol_status["news_data"]:
                missing_report["missing_news"].append(symbol)
        
        return missing_report