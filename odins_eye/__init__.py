"""
Odin's Eye - Financial Data Access Library

A Python library for accessing financial market data, economic indicators, 
news sentiment, and Google Trends data stored in the trading system's data lake.
"""

from .core import OdinsEye
from .filters import DateRange, InstrumentFilter, DataTypeFilter, MarketDataInterval
from .exceptions import OdinsEyeError, DataNotFoundError, InvalidFilterError
from .data_downloader import DataDownloader

__version__ = "1.0.0"
__all__ = [
    "OdinsEye",
    "DataDownloader",
    "DateRange", 
    "InstrumentFilter",
    "DataTypeFilter",
    "MarketDataInterval",
    "OdinsEyeError",
    "DataNotFoundError", 
    "InvalidFilterError"
]