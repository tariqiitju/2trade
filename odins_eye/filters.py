"""
Filter classes for querying data
"""

from typing import List, Optional, Union, Dict, Any
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum


class MarketDataInterval(Enum):
    """Supported market data intervals"""
    DAILY = "daily"
    HOURLY = "hourly"
    MIN_30 = "30min"
    MIN_15 = "15min"
    MIN_5 = "5min"
    MIN_1 = "1min"


class DataType(Enum):
    """Supported data types"""
    MARKET_DATA = "market_data"
    ECONOMIC_DATA = "economic_data" 
    NEWS_DATA = "news_data"
    TRENDS_DATA = "trends_data"
    EARNINGS_DATA = "earnings_data"
    SEC_FILINGS = "sec_filings"
    INSIDER_TRADING = "insider_trading"
    CORPORATE_ACTIONS = "corporate_actions"


class SentimentLabel(Enum):
    """News sentiment labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class NewsCategory(Enum):
    """News article categories"""
    EARNINGS = "earnings"
    MARKET = "market"
    ECONOMIC = "economic"
    COMPANY = "company"


class SECFilingType(Enum):
    """SEC filing types"""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    PROXY = "proxy"
    FORM_3 = "3"
    FORM_4 = "4"
    FORM_5 = "5"


class CorporateActionType(Enum):
    """Corporate action types"""
    SPLIT = "split"
    DIVIDEND = "dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    ACQUISITION = "acquisition"


class InsiderTransactionType(Enum):
    """Insider transaction types"""
    PURCHASE = "purchase"
    SALE = "sale"
    GIFT = "gift"
    EXERCISE = "exercise"


@dataclass
class DateRange:
    """Date range filter for time-based queries"""
    start_date: Union[str, date, datetime]
    end_date: Union[str, date, datetime, None] = None
    
    def __post_init__(self):
        # Convert strings to datetime objects
        if isinstance(self.start_date, str):
            self.start_date = datetime.fromisoformat(self.start_date.replace('Z', '+00:00'))
        elif isinstance(self.start_date, date):
            self.start_date = datetime.combine(self.start_date, datetime.min.time())
            
        if self.end_date is not None:
            if isinstance(self.end_date, str):
                self.end_date = datetime.fromisoformat(self.end_date.replace('Z', '+00:00'))
            elif isinstance(self.end_date, date):
                self.end_date = datetime.combine(self.end_date, datetime.max.time())


@dataclass
class InstrumentFilter:
    """Filter for specific instruments/symbols"""
    symbols: Optional[List[str]] = None
    symbol_pattern: Optional[str] = None  # Regex pattern for symbols
    exclude_symbols: Optional[List[str]] = None
    
    def matches(self, symbol: str) -> bool:
        """Check if symbol matches this filter"""
        import re
        
        # Check exclusions first
        if self.exclude_symbols and symbol in self.exclude_symbols:
            return False
            
        # Check explicit symbols list
        if self.symbols and symbol not in self.symbols:
            return False
            
        # Check pattern
        if self.symbol_pattern:
            if not re.match(self.symbol_pattern, symbol):
                return False
                
        return True


@dataclass
class DataTypeFilter:
    """Filter for specific data types and subtypes"""
    data_type: DataType
    interval: Optional[MarketDataInterval] = None  # For market data only
    
    # Market data specific filters
    include_indicators: Optional[List[str]] = None
    exclude_indicators: Optional[List[str]] = None
    min_volume: Optional[int] = None
    price_range: Optional[tuple] = None  # (min_price, max_price)
    
    # Economic data specific filters
    economic_category: Optional[str] = None
    importance: Optional[str] = None  # "high", "medium", "low"
    frequency: Optional[str] = None  # "daily", "weekly", "monthly", etc.
    
    # News data specific filters
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_score_range: Optional[tuple] = None  # (min_score, max_score)
    news_category: Optional[NewsCategory] = None
    sources: Optional[List[str]] = None
    
    # Trends data specific filters
    geo: Optional[str] = None
    category: Optional[int] = None
    min_search_interest: Optional[int] = None


@dataclass 
class QueryFilter:
    """Combined filter for comprehensive data queries"""
    date_range: Optional[DateRange] = None
    instruments: Optional[InstrumentFilter] = None
    data_types: Optional[List[DataTypeFilter]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # "asc" or "desc"
    
    def __post_init__(self):
        if self.data_types is None:
            self.data_types = []