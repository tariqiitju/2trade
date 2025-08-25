"""
Instrument List Loader for Consuela Configuration

Utility functions for loading different instrument lists from YAML files.
"""

import yaml
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_base_list_all() -> List[Dict[str, Any]]:
    """
    Load the complete base instrument list including delisted instruments.
    
    Returns:
        List of instrument dictionaries
    """
    try:
        config_dir = Path(__file__).parent / "instrument-list"
        file_path = config_dir / "base-list-all.yml"
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        instruments = data.get("instruments", [])
        logger.info(f"Loaded {len(instruments)} instruments from base-list-all.yml")
        
        return instruments
        
    except Exception as e:
        logger.error(f"Failed to load base-list-all.yml: {e}")
        return []


def load_favorites_instruments() -> List[Dict[str, Any]]:
    """
    Load favorite instruments list.
    
    Returns:
        List of instrument dictionaries
    """
    try:
        config_dir = Path(__file__).parent / "instrument-list"
        file_path = config_dir / "favorites_instruments.yml"
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        instruments = data.get("favorites", [])
        logger.info(f"Loaded {len(instruments)} instruments from favorites_instruments.yml")
        
        return instruments
        
    except Exception as e:
        logger.error(f"Failed to load favorites_instruments.yml: {e}")
        return []


def load_popular_instruments() -> List[Dict[str, Any]]:
    """
    Load popular instruments list.
    
    Returns:
        List of instrument dictionaries
    """
    try:
        config_dir = Path(__file__).parent / "instrument-list"
        file_path = config_dir / "popular_instruments.yml"
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        instruments = data.get("popular", [])
        logger.info(f"Loaded {len(instruments)} instruments from popular_instruments.yml")
        
        return instruments
        
    except Exception as e:
        logger.error(f"Failed to load popular_instruments.yml: {e}")
        return []


def get_symbols_by_filter(
    instruments: List[Dict[str, Any]],
    status: str = None,
    sector: str = None,
    market_cap_category: str = None,
    exchange: str = None,
    limit: int = None
) -> List[str]:
    """
    Filter instruments and return list of symbols.
    
    Args:
        instruments: List of instrument dictionaries
        status: Filter by status (active, delisted, etc.)
        sector: Filter by sector
        market_cap_category: Filter by market cap category
        exchange: Filter by exchange
        limit: Maximum number of symbols to return
        
    Returns:
        List of filtered symbols
    """
    filtered = instruments.copy()
    
    if status:
        filtered = [inst for inst in filtered if inst.get("status") == status]
    
    if sector:
        filtered = [inst for inst in filtered if inst.get("sector") == sector]
    
    if market_cap_category:
        filtered = [inst for inst in filtered if inst.get("market_cap_category") == market_cap_category]
    
    if exchange:
        filtered = [inst for inst in filtered if inst.get("exchange") == exchange]
    
    # Extract symbols
    symbols = [inst["symbol"] for inst in filtered if "symbol" in inst]
    
    # Apply limit
    if limit and len(symbols) > limit:
        symbols = symbols[:limit]
    
    return symbols


def get_active_symbols(limit: int = None) -> List[str]:
    """
    Get list of active symbols from base list.
    
    Args:
        limit: Maximum number of symbols to return
        
    Returns:
        List of active symbols
    """
    instruments = load_base_list_all()
    return get_symbols_by_filter(instruments, status="active", limit=limit)


def get_delisted_symbols(limit: int = None) -> List[str]:
    """
    Get list of delisted symbols from base list.
    
    Args:
        limit: Maximum number of symbols to return
        
    Returns:
        List of delisted symbols
    """
    instruments = load_base_list_all()
    return get_symbols_by_filter(instruments, status="delisted", limit=limit)


def get_etf_symbols(limit: int = None) -> List[str]:
    """
    Get list of ETF symbols from base list.
    
    Args:
        limit: Maximum number of symbols to return
        
    Returns:
        List of ETF symbols
    """
    instruments = load_base_list_all()
    etf_symbols = []
    
    for inst in instruments:
        if inst.get("market_cap_category") == "etf" or inst.get("sector") in ["ETF", "Commodity ETF", "Bond ETF"]:
            etf_symbols.append(inst["symbol"])
    
    if limit and len(etf_symbols) > limit:
        etf_symbols = etf_symbols[:limit]
    
    return etf_symbols


def get_large_cap_symbols(limit: int = None) -> List[str]:
    """
    Get list of large cap symbols from base list.
    
    Args:
        limit: Maximum number of symbols to return
        
    Returns:
        List of large cap symbols
    """
    instruments = load_base_list_all()
    return get_symbols_by_filter(
        instruments, 
        status="active",
        market_cap_category="large_cap", 
        limit=limit
    )


def get_symbols_by_sector(sector: str, limit: int = None) -> List[str]:
    """
    Get symbols for a specific sector.
    
    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare')
        limit: Maximum number of symbols to return
        
    Returns:
        List of symbols in the sector
    """
    instruments = load_base_list_all()
    return get_symbols_by_filter(
        instruments,
        status="active", 
        sector=sector,
        limit=limit
    )


def get_instrument_metadata(symbol: str) -> Dict[str, Any]:
    """
    Get metadata for a specific instrument.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dict with instrument metadata or empty dict if not found
    """
    instruments = load_base_list_all()
    
    for inst in instruments:
        if inst.get("symbol") == symbol:
            return inst
    
    return {}


def list_available_sectors() -> List[str]:
    """
    Get list of all available sectors in the base list.
    
    Returns:
        List of unique sectors
    """
    instruments = load_base_list_all()
    sectors = set()
    
    for inst in instruments:
        if "sector" in inst:
            sectors.add(inst["sector"])
    
    return sorted(list(sectors))


def get_instrument_stats() -> Dict[str, Any]:
    """
    Get statistics about the instrument database.
    
    Returns:
        Dict with statistics
    """
    instruments = load_base_list_all()
    
    stats = {
        "total_instruments": len(instruments),
        "active_count": 0,
        "delisted_count": 0,
        "by_market_cap": {},
        "by_sector": {},
        "by_exchange": {},
    }
    
    for inst in instruments:
        # Count by status
        status = inst.get("status", "unknown")
        if status == "active":
            stats["active_count"] += 1
        elif status == "delisted":
            stats["delisted_count"] += 1
        
        # Count by market cap
        market_cap = inst.get("market_cap_category", "unknown")
        stats["by_market_cap"][market_cap] = stats["by_market_cap"].get(market_cap, 0) + 1
        
        # Count by sector
        sector = inst.get("sector", "unknown")
        stats["by_sector"][sector] = stats["by_sector"].get(sector, 0) + 1
        
        # Count by exchange
        exchange = inst.get("exchange", "unknown")
        stats["by_exchange"][exchange] = stats["by_exchange"].get(exchange, 0) + 1
    
    return stats