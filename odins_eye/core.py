"""
Core Odin's Eye library implementation
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime

from .filters import (QueryFilter, DataType, MarketDataInterval, DateRange, InstrumentFilter, DataTypeFilter,
                      SECFilingType, CorporateActionType, InsiderTransactionType)
from .exceptions import OdinsEyeError, DataNotFoundError, ConfigurationError


class OdinsEye:
    """
    Main interface for accessing financial data stored in the trading system's data lake.
    
    Provides unified access to market data, economic indicators, news sentiment,
    and Google Trends data with flexible filtering capabilities.
    """
    
    def __init__(self, data_root: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize Odin's Eye data access library.
        
        Args:
            data_root: Override default data root directory
            config_path: Path to custom configuration file
        """
        self.config = self._load_config(config_path)
        self.data_root = Path(data_root or self._get_default_data_root())
        self._validate_data_root()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        try:
            # Load default configs
            package_dir = Path(__file__).parent
            
            # Load data spec
            spec_path = package_dir / "config" / "data-spec.yml"
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec_config = yaml.safe_load(f)
                
            # Load data config
            data_config_path = package_dir / "config" / "data-config.yml"  
            with open(data_config_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                
            # Merge configs
            config = {**spec_config, **data_config}
            
            # Load custom config if provided
            if config_path:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                    config.update(custom_config)
                    
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _get_default_data_root(self) -> str:
        """Get default data root from config"""
        # Try environment variable first
        env_root = os.environ.get('DATA_ROOT')
        if env_root:
            return env_root
            
        # Use config default
        return self.config.get('config', {}).get('default_root', 'W:/market-data')
    
    def _validate_data_root(self):
        """Validate that data root directory exists"""
        if not self.data_root.exists():
            raise ConfigurationError(f"Data root directory does not exist: {self.data_root}")
            
    def get_market_data(self, 
                       symbols: Union[str, List[str]] = None,
                       interval: Union[str, MarketDataInterval] = MarketDataInterval.DAILY,
                       date_range: Optional[DateRange] = None,
                       include_indicators: Optional[List[str]] = None,
                       **filters) -> pd.DataFrame:
        """
        Get market data with filtering options.
        
        Args:
            symbols: Single symbol or list of symbols to retrieve
            interval: Data interval (daily, hourly, 30min, 15min, 5min, 1min)
            date_range: Date range filter
            include_indicators: Specific technical indicators to include
            **filters: Additional market-data-specific filters
            
        Returns:
            DataFrame with market data
        """
        if isinstance(interval, str):
            interval = MarketDataInterval(interval)
            
        # Build query filter
        query_filter = QueryFilter(
            date_range=date_range,
            instruments=InstrumentFilter(symbols=[symbols] if isinstance(symbols, str) else symbols),
            data_types=[DataTypeFilter(
                data_type=DataType.MARKET_DATA,
                interval=interval,
                include_indicators=include_indicators,
                **filters
            )]
        )
        
        return self._execute_query(query_filter)
    
    def get_economic_data(self,
                         indicators: Union[str, List[str]] = None,
                         date_range: Optional[DateRange] = None,
                         category: Optional[str] = None,
                         importance: Optional[str] = None,
                         **filters) -> pd.DataFrame:
        """
        Get economic indicator data.
        
        Args:
            indicators: FRED indicator symbols to retrieve
            date_range: Date range filter  
            category: Economic category filter
            importance: Importance level filter
            **filters: Additional economic-data-specific filters
            
        Returns:
            DataFrame with economic data
        """
        query_filter = QueryFilter(
            date_range=date_range,
            instruments=InstrumentFilter(symbols=[indicators] if isinstance(indicators, str) else indicators),
            data_types=[DataTypeFilter(
                data_type=DataType.ECONOMIC_DATA,
                economic_category=category,
                importance=importance,
                **filters
            )]
        )
        
        return self._execute_query(query_filter)
    
    def get_news_data(self,
                     symbols: Union[str, List[str]] = None,
                     date_range: Optional[DateRange] = None,
                     sentiment_label: Optional[str] = None,
                     sources: Optional[List[str]] = None,
                     **filters) -> List[Dict[str, Any]]:
        """
        Get news data with sentiment analysis.
        
        Args:
            symbols: Symbols to get news for
            date_range: Date range filter
            sentiment_label: Filter by sentiment (positive, negative, neutral)
            sources: Filter by news sources
            **filters: Additional news-specific filters
            
        Returns:
            List of news articles
        """
        query_filter = QueryFilter(
            date_range=date_range,
            instruments=InstrumentFilter(symbols=[symbols] if isinstance(symbols, str) else symbols),
            data_types=[DataTypeFilter(
                data_type=DataType.NEWS_DATA,
                sentiment_label=sentiment_label,
                sources=sources,
                **filters
            )]
        )
        
        return self._execute_news_query(query_filter)
    
    def get_trends_data(self,
                       keywords: Union[str, List[str]] = None,
                       date_range: Optional[DateRange] = None,
                       geo: str = "US",
                       **filters) -> List[Dict[str, Any]]:
        """
        Get Google Trends data.
        
        Args:
            keywords: Search keywords or symbols
            date_range: Date range filter
            geo: Geographic region
            **filters: Additional trends-specific filters
            
        Returns:
            List of trends data
        """
        query_filter = QueryFilter(
            date_range=date_range,
            instruments=InstrumentFilter(symbols=[keywords] if isinstance(keywords, str) else keywords),
            data_types=[DataTypeFilter(
                data_type=DataType.TRENDS_DATA,
                geo=geo,
                **filters
            )]
        )
        
        return self._execute_trends_query(query_filter)
    
    def query(self, query_filter: QueryFilter) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute a custom query with comprehensive filtering.
        
        Args:
            query_filter: Complete query filter specification
            
        Returns:
            Query results (DataFrame for tabular data, List for JSON data)
        """
        if not query_filter.data_types:
            raise ValueError("At least one data type filter must be specified")
            
        # Route to appropriate handler based on data type
        data_type = query_filter.data_types[0].data_type
        
        if data_type in [DataType.MARKET_DATA, DataType.ECONOMIC_DATA]:
            return self._execute_query(query_filter)
        elif data_type == DataType.NEWS_DATA:
            return self._execute_news_query(query_filter)
        elif data_type == DataType.TRENDS_DATA:
            return self._execute_trends_query(query_filter)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _execute_query(self, query_filter: QueryFilter) -> pd.DataFrame:
        """Execute query for tabular data (market/economic)"""
        results = []
        
        for data_type_filter in query_filter.data_types:
            if data_type_filter.data_type == DataType.MARKET_DATA:
                data = self._load_market_data(query_filter, data_type_filter)
            elif data_type_filter.data_type == DataType.ECONOMIC_DATA:
                data = self._load_economic_data(query_filter, data_type_filter)
            else:
                continue
                
            if data is not None and not data.empty:
                results.append(data)
        
        if not results:
            return pd.DataFrame()
            
        # Combine results
        combined = pd.concat(results, ignore_index=True)
        
        # Apply additional filtering
        combined = self._apply_filters(combined, query_filter)
        
        return combined
    
    def _execute_news_query(self, query_filter: QueryFilter) -> List[Dict[str, Any]]:
        """Execute query for news data"""
        return self._load_news_data(query_filter, query_filter.data_types[0])
    
    def _execute_trends_query(self, query_filter: QueryFilter) -> List[Dict[str, Any]]:
        """Execute query for trends data"""
        return self._load_trends_data(query_filter, query_filter.data_types[0])
    
    def _load_market_data(self, query_filter: QueryFilter, data_type_filter: DataTypeFilter) -> pd.DataFrame:
        """Load market data from parquet files"""
        interval_dir = data_type_filter.interval.value if data_type_filter.interval else "daily"
        data_dir = self.data_root / "market_data" / interval_dir
        
        if not data_dir.exists():
            return pd.DataFrame()
        
        results = []
        symbols = query_filter.instruments.symbols if query_filter.instruments else []
        
        # If no specific symbols, get all available
        if not symbols:
            symbols = [f.stem for f in data_dir.glob("*.parquet")]
        
        for symbol in symbols:
            file_path = data_dir / f"{symbol}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    df['symbol'] = symbol
                    results.append(df)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _load_economic_data(self, query_filter: QueryFilter, data_type_filter: DataTypeFilter) -> pd.DataFrame:
        """Load economic data from parquet files"""
        data_dir = self.data_root / "economic_data"
        
        if not data_dir.exists():
            return pd.DataFrame()
        
        results = []
        indicators = query_filter.instruments.symbols if query_filter.instruments else []
        
        # If no specific indicators, get all available
        if not indicators:
            indicators = [f.stem for f in data_dir.glob("*.parquet")]
        
        for indicator in indicators:
            file_path = data_dir / f"{indicator}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    results.append(df)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _load_news_data(self, query_filter: QueryFilter, data_type_filter: DataTypeFilter) -> List[Dict[str, Any]]:
        """Load news data from JSON files"""
        data_dir = self.data_root / "news_data"
        
        if not data_dir.exists():
            return []
        
        results = []
        
        # Get all news files in date range
        for file_path in data_dir.glob("news_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                    
                if isinstance(news_data, list):
                    results.extend(news_data)
                    
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return results
    
    def _load_trends_data(self, query_filter: QueryFilter, data_type_filter: DataTypeFilter) -> List[Dict[str, Any]]:
        """Load trends data from JSON files"""
        data_dir = self.data_root / "trends_data"
        
        if not data_dir.exists():
            return []
        
        results = []
        
        # Get all trends files
        for file_path in data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    trends_data = json.load(f)
                    results.append(trends_data)
                    
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return results
    
    def _apply_filters(self, df: pd.DataFrame, query_filter: QueryFilter) -> pd.DataFrame:
        """Apply additional filtering to DataFrame"""
        if df.empty:
            return df
        
        # Date range filtering
        if query_filter.date_range and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            if query_filter.date_range.start_date:
                start_date = pd.to_datetime(query_filter.date_range.start_date)
                # Ensure both dates have the same timezone handling
                if df['date'].dt.tz is not None and start_date.tz is None:
                    start_date = start_date.tz_localize(df['date'].dt.tz)
                elif df['date'].dt.tz is None and start_date.tz is not None:
                    start_date = start_date.tz_localize(None)
                df = df[df['date'] >= start_date]
                
            if query_filter.date_range.end_date:
                end_date = pd.to_datetime(query_filter.date_range.end_date)
                # Ensure both dates have the same timezone handling
                if df['date'].dt.tz is not None and end_date.tz is None:
                    end_date = end_date.tz_localize(df['date'].dt.tz)
                elif df['date'].dt.tz is None and end_date.tz is not None:
                    end_date = end_date.tz_localize(None)
                df = df[df['date'] <= end_date]
        
        # Sorting
        if query_filter.sort_by and query_filter.sort_by in df.columns:
            ascending = query_filter.sort_order == "asc"
            df = df.sort_values(query_filter.sort_by, ascending=ascending)
        
        # Limit and offset
        if query_filter.offset:
            df = df.iloc[query_filter.offset:]
            
        if query_filter.limit:
            df = df.head(query_filter.limit)
        
        return df
    
    def list_available_symbols(self, data_type: DataType, interval: Optional[MarketDataInterval] = None) -> List[str]:
        """
        List all available symbols for a given data type.
        
        Args:
            data_type: Type of data to check
            interval: For market data, specify interval
            
        Returns:
            List of available symbols
        """
        if data_type == DataType.MARKET_DATA:
            interval_dir = interval.value if interval else "daily"
            data_dir = self.data_root / "market_data" / interval_dir
        elif data_type == DataType.ECONOMIC_DATA:
            data_dir = self.data_root / "economic_data"
        else:
            raise ValueError(f"Symbol listing not supported for {data_type}")
        
        if not data_dir.exists():
            return []
        
        return [f.stem for f in data_dir.glob("*.parquet")]
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about available data.
        
        Returns:
            Dictionary with data availability information
        """
        info = {
            "data_root": str(self.data_root),
            "market_data": {},
            "economic_data": {"available_indicators": 0},
            "news_data": {"available_files": 0},
            "trends_data": {"available_files": 0}
        }
        
        # Market data info
        for interval in MarketDataInterval:
            data_dir = self.data_root / "market_data" / interval.value
            if data_dir.exists():
                symbols = list(data_dir.glob("*.parquet"))
                info["market_data"][interval.value] = {
                    "available_symbols": len(symbols),
                    "symbols": [f.stem for f in symbols[:10]]  # First 10 as sample
                }
        
        # Economic data info
        econ_dir = self.data_root / "economic_data"
        if econ_dir.exists():
            indicators = list(econ_dir.glob("*.parquet"))
            info["economic_data"]["available_indicators"] = len(indicators)
        
        # News data info  
        news_dir = self.data_root / "news_data"
        if news_dir.exists():
            news_files = list(news_dir.glob("news_*.json"))
            info["news_data"]["available_files"] = len(news_files)
        
        # Trends data info
        trends_dir = self.data_root / "trends_data"
        if trends_dir.exists():
            trends_files = list(trends_dir.glob("*.json"))
            info["trends_data"]["available_files"] = len(trends_files)
        
        return info
    
    # Corporate Data Access Methods
    
    def get_earnings_data(self,
                         symbols: Union[str, List[str]] = None,
                         date_range: Optional[DateRange] = None,
                         source: str = "all",
                         **filters) -> List[Dict[str, Any]]:
        """
        Get earnings calendar data.
        
        Args:
            symbols: Single symbol or list of symbols
            date_range: Date range filter
            source: Data source filter (fmp, finnhub, eodhd, etc.)
            **filters: Additional earnings-specific filters
            
        Returns:
            List of earnings announcement records
        """
        earnings_dir = self.data_root / "earnings_data"
        if not earnings_dir.exists():
            return []
        
        all_earnings = []
        
        # Process earnings calendar files
        for earnings_file in earnings_dir.glob("*.json"):
            try:
                with open(earnings_file, 'r') as f:
                    earnings_data = json.load(f)
                    
                # Filter by symbols if specified
                if symbols:
                    if isinstance(symbols, str):
                        symbols = [symbols]
                    earnings_data = [e for e in earnings_data if e.get('symbol') in symbols]
                
                # Filter by date range if specified
                if date_range:
                    earnings_data = self._filter_earnings_by_date(earnings_data, date_range)
                
                all_earnings.extend(earnings_data)
                
            except Exception as e:
                continue
        
        return all_earnings
    
    def get_sec_filings(self,
                       symbols: Union[str, List[str]] = None,
                       filing_types: Union[str, List[str], List[SECFilingType]] = None,
                       date_range: Optional[DateRange] = None,
                       **filters) -> List[Dict[str, Any]]:
        """
        Get SEC filings data.
        
        Args:
            symbols: Single symbol or list of symbols
            filing_types: Specific SEC filing types (10-K, 10-Q, 8-K, etc.)
            date_range: Date range filter
            **filters: Additional SEC filing filters
            
        Returns:
            List of SEC filing records
        """
        sec_dir = self.data_root / "sec_filings"
        if not sec_dir.exists():
            return []
        
        all_filings = []
        
        # Process SEC filing files
        for filing_file in sec_dir.glob("**/*.json"):
            try:
                with open(filing_file, 'r') as f:
                    filing_data = json.load(f)
                    
                # Handle both single records and lists
                if isinstance(filing_data, dict):
                    filing_data = [filing_data]
                
                # Filter by symbols if specified
                if symbols:
                    if isinstance(symbols, str):
                        symbols = [symbols]
                    filing_data = [f for f in filing_data if f.get('symbol') in symbols]
                
                # Filter by filing types if specified
                if filing_types:
                    if isinstance(filing_types, str):
                        filing_types = [filing_types]
                    # Convert SECFilingType enums to strings
                    filing_type_strs = []
                    for ft in filing_types:
                        if hasattr(ft, 'value'):
                            filing_type_strs.append(ft.value)
                        else:
                            filing_type_strs.append(str(ft))
                    filing_data = [f for f in filing_data if f.get('form_type') in filing_type_strs]
                
                # Filter by date range if specified
                if date_range:
                    filing_data = self._filter_filings_by_date(filing_data, date_range)
                
                all_filings.extend(filing_data)
                
            except Exception as e:
                continue
        
        return all_filings
    
    def get_insider_trading(self,
                           symbols: Union[str, List[str]] = None,
                           transaction_types: Union[str, List[str], List[InsiderTransactionType]] = None,
                           date_range: Optional[DateRange] = None,
                           **filters) -> List[Dict[str, Any]]:
        """
        Get insider trading data.
        
        Args:
            symbols: Single symbol or list of symbols
            transaction_types: Specific transaction types (purchase, sale, etc.)
            date_range: Date range filter
            **filters: Additional insider trading filters
            
        Returns:
            List of insider trading records
        """
        insider_dir = self.data_root / "insider_trading"
        if not insider_dir.exists():
            return []
        
        all_trades = []
        
        # Process insider trading files
        for trade_file in insider_dir.glob("*.json"):
            try:
                with open(trade_file, 'r') as f:
                    trade_data = json.load(f)
                    
                # Handle both single records and lists
                if isinstance(trade_data, dict):
                    trade_data = [trade_data]
                
                # Filter by symbols if specified
                if symbols:
                    if isinstance(symbols, str):
                        symbols = [symbols]
                    trade_data = [t for t in trade_data if t.get('symbol') in symbols]
                
                # Filter by transaction types if specified
                if transaction_types:
                    if isinstance(transaction_types, str):
                        transaction_types = [transaction_types]
                    # Convert InsiderTransactionType enums to strings
                    trans_type_strs = []
                    for tt in transaction_types:
                        if hasattr(tt, 'value'):
                            trans_type_strs.append(tt.value)
                        else:
                            trans_type_strs.append(str(tt))
                    trade_data = [t for t in trade_data if t.get('transaction_type') in trans_type_strs]
                
                # Filter by date range if specified
                if date_range:
                    trade_data = self._filter_trades_by_date(trade_data, date_range)
                
                all_trades.extend(trade_data)
                
            except Exception as e:
                continue
        
        return all_trades
    
    def get_corporate_actions(self,
                             symbols: Union[str, List[str]] = None,
                             action_types: Union[str, List[str], List[CorporateActionType]] = None,
                             date_range: Optional[DateRange] = None,
                             **filters) -> List[Dict[str, Any]]:
        """
        Get corporate actions data.
        
        Args:
            symbols: Single symbol or list of symbols
            action_types: Specific action types (split, dividend, merger, etc.)
            date_range: Date range filter
            **filters: Additional corporate action filters
            
        Returns:
            List of corporate action records
        """
        actions_dir = self.data_root / "corporate_actions"
        if not actions_dir.exists():
            return []
        
        all_actions = []
        
        # Process corporate action files
        for action_file in actions_dir.glob("*.json"):
            try:
                with open(action_file, 'r') as f:
                    action_data = json.load(f)
                    
                # Handle both single records and lists
                if isinstance(action_data, dict):
                    action_data = [action_data]
                
                # Filter by symbols if specified
                if symbols:
                    if isinstance(symbols, str):
                        symbols = [symbols]
                    action_data = [a for a in action_data if a.get('symbol') in symbols]
                
                # Filter by action types if specified
                if action_types:
                    if isinstance(action_types, str):
                        action_types = [action_types]
                    # Convert CorporateActionType enums to strings
                    action_type_strs = []
                    for at in action_types:
                        if hasattr(at, 'value'):
                            action_type_strs.append(at.value)
                        else:
                            action_type_strs.append(str(at))
                    action_data = [a for a in action_data if a.get('action_type') in action_type_strs]
                
                # Filter by date range if specified
                if date_range:
                    action_data = self._filter_actions_by_date(action_data, date_range)
                
                all_actions.extend(action_data)
                
            except Exception as e:
                continue
        
        return all_actions
    
    # Helper methods for corporate data filtering
    
    def _filter_earnings_by_date(self, earnings_data: List[Dict], date_range: DateRange) -> List[Dict]:
        """Filter earnings data by date range"""
        filtered = []
        for earning in earnings_data:
            earn_date = earning.get('date')
            if earn_date and self._is_date_in_range(earn_date, date_range):
                filtered.append(earning)
        return filtered
    
    def _filter_filings_by_date(self, filing_data: List[Dict], date_range: DateRange) -> List[Dict]:
        """Filter SEC filings by date range"""
        filtered = []
        for filing in filing_data:
            filing_date = filing.get('filing_date') or filing.get('date')
            if filing_date and self._is_date_in_range(filing_date, date_range):
                filtered.append(filing)
        return filtered
    
    def _filter_trades_by_date(self, trade_data: List[Dict], date_range: DateRange) -> List[Dict]:
        """Filter insider trades by date range"""
        filtered = []
        for trade in trade_data:
            trade_date = trade.get('transaction_date') or trade.get('date')
            if trade_date and self._is_date_in_range(trade_date, date_range):
                filtered.append(trade)
        return filtered
    
    def _filter_actions_by_date(self, action_data: List[Dict], date_range: DateRange) -> List[Dict]:
        """Filter corporate actions by date range"""
        filtered = []
        for action in action_data:
            action_date = action.get('effective_date') or action.get('date')
            if action_date and self._is_date_in_range(action_date, date_range):
                filtered.append(action)
        return filtered
    
    def _is_date_in_range(self, date_str: str, date_range: DateRange) -> bool:
        """Check if date string falls within date range"""
        try:
            if isinstance(date_str, str):
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date_obj = date_str
            
            if date_range.start_date and date_obj < date_range.start_date:
                return False
            if date_range.end_date and date_obj > date_range.end_date:
                return False
            return True
        except:
            return False