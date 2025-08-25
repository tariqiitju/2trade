#!/usr/bin/env python3
"""
Data Configuration Loader for 2Trade System

Loads configuration data for various data collection and processing components.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

def load_data_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load data configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. If None, uses default locations.
        
    Returns:
        Dictionary containing data configuration
    """
    if config_path is None:
        # Try multiple default locations
        possible_paths = [
            Path(__file__).parent / "data-config.yml",
            Path(__file__).parent.parent.parent / "odins_eye" / "config" / "data-config.yml",
            "data-config.yml"
        ]
        
        config_path = None
        for path in possible_paths:
            if Path(path).exists():
                config_path = str(path)
                break
        
        if config_path is None:
            # Return default configuration if no file found
            return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default data configuration.
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        'data_sources': {
            'yahoo_finance': {
                'enabled': True,
                'rate_limit': 5,
                'timeout': 30,
                'retry_attempts': 3,
                'priority': 1
            },
            'alpha_vantage': {
                'enabled': False,
                'api_key': None,
                'rate_limit': 5,
                'timeout': 30,
                'retry_attempts': 3,
                'priority': 2
            },
            'finnhub': {
                'enabled': False,
                'api_key': None,
                'rate_limit': 60,
                'timeout': 30,
                'retry_attempts': 3,
                'priority': 3
            },
            'fred': {
                'enabled': True,
                'api_key': '684e67a4a3b7a9a27729a1ed62546f2e',
                'rate_limit': 10,
                'timeout': 30,
                'retry_attempts': 3,
                'priority': 4
            },
            'newsapi': {
                'enabled': True,
                'api_key': '972972a223b3486ca754f0f90b49485a',
                'rate_limit': 1000,
                'timeout': 30,
                'retry_attempts': 3,
                'priority': 5
            }
        },
        'storage': {
            'type': 'file',
            'base_path': os.environ.get('DATA_ROOT', 'W:/market-data'),
            'compression': 'snappy',
            'format': 'parquet'
        },
        'intervals': {
            'market_data': ['1d', '1h', '30min', '15min', '5min', '1min'],
            'default_interval': '1d'
        },
        'data_retention': {
            'market_data_days': 3650,  # 10 years
            'news_data_days': 365,     # 1 year
            'trends_data_days': 730,   # 2 years
            'earnings_data_days': 1825 # 5 years
        }
    }

def get_data_root() -> str:
    """
    Get the data root directory from configuration or environment.
    
    Returns:
        Path to data root directory
    """
    config = load_data_config()
    return config.get('storage', {}).get('base_path', os.environ.get('DATA_ROOT', 'W:/market-data'))

def get_api_config(source_name: str) -> Dict[str, Any]:
    """
    Get API configuration for a specific data source.
    
    Args:
        source_name: Name of the data source (e.g., 'finnhub', 'alpha_vantage')
        
    Returns:
        Dictionary with API configuration
    """
    config = load_data_config()
    return config.get('data_sources', {}).get(source_name, {})

def is_source_enabled(source_name: str) -> bool:
    """
    Check if a data source is enabled.
    
    Args:
        source_name: Name of the data source
        
    Returns:
        True if source is enabled, False otherwise
    """
    api_config = get_api_config(source_name)
    return api_config.get('enabled', False)