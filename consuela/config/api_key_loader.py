"""
API Key Configuration Loader for 2Trade System

This module provides secure loading of API keys from configuration files
while maintaining security best practices.

Author: Consuela Configuration Module
Created: 2025-08-24
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class APIKeyLoader:
    """
    Secure API key loader that supports multiple sources:
    1. Configuration file (api_keys.yml)
    2. Environment variables
    3. Direct parameter passing
    
    Priority: Direct params > Environment variables > Config file
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize API key loader.
        
        Args:
            config_file: Path to API keys configuration file
        """
        if config_file:
            self.config_file = Path(config_file)
        else:
            # Default to api_keys.yml in consuela/config/
            self.config_file = Path(__file__).parent / "api_keys.yml"
        
        self._config_data = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self._config_data = yaml.safe_load(f)
                logger.debug(f"Loaded API key configuration from {self.config_file}")
            else:
                logger.warning(f"API key config file not found: {self.config_file}")
                self._config_data = {}
        except Exception as e:
            logger.error(f"Failed to load API key config: {e}")
            self._config_data = {}
    
    def get_fred_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """
        Get FRED API key from various sources.
        
        Priority: Direct param > Environment > Config file
        
        Args:
            api_key: Direct API key parameter
            
        Returns:
            FRED API key or None if not found
        """
        # 1. Direct parameter (highest priority)
        if api_key:
            return api_key
        
        # 2. Environment variable
        env_key = os.getenv('FRED_API_KEY')
        if env_key:
            return env_key
        
        # 3. Configuration file
        try:
            return self._config_data.get('fred', {}).get('api_key')
        except (AttributeError, TypeError):
            return None
    
    def get_news_api_key(self, source: str = "newsapi", api_key: Optional[str] = None) -> Optional[str]:
        """
        Get News API key for specified source.
        
        Args:
            source: News API source ('newsapi', 'alpha_vantage', etc.)
            api_key: Direct API key parameter
            
        Returns:
            News API key or None if not found
        """
        # 1. Direct parameter (highest priority)
        if api_key:
            return api_key
        
        # 2. Environment variable
        env_key_map = {
            'newsapi': 'NEWS_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
            'polygon': 'POLYGON_API_KEY'
        }
        
        env_var = env_key_map.get(source)
        if env_var:
            env_key = os.getenv(env_var)
            if env_key:
                return env_key
        
        # 3. Configuration file
        try:
            news_config = self._config_data.get('news', {})
            if source in news_config:
                return news_config[source].get('api_key')
            
            # Fallback for newsapi being the default
            if source == 'newsapi':
                return news_config.get('newsapi', {}).get('api_key')
        except (AttributeError, TypeError):
            pass
        
        return None
    
    def get_all_api_keys(self) -> Dict[str, Any]:
        """
        Get all configured API keys for debugging/status purposes.
        
        Returns:
            Dictionary with API key status (masked for security)
        """
        status = {
            'fred': {
                'configured': bool(self.get_fred_api_key()),
                'source': self._get_key_source('fred')
            },
            'news': {
                'newsapi': {
                    'configured': bool(self.get_news_api_key('newsapi')),
                    'source': self._get_key_source('news', 'newsapi')
                }
            }
        }
        
        return status
    
    def _get_key_source(self, service: str, subservice: Optional[str] = None) -> str:
        """Determine the source of an API key for debugging."""
        if service == 'fred':
            if os.getenv('FRED_API_KEY'):
                return 'environment'
            elif self._config_data.get('fred', {}).get('api_key'):
                return 'config_file'
            else:
                return 'not_found'
        
        elif service == 'news' and subservice == 'newsapi':
            if os.getenv('NEWS_API_KEY'):
                return 'environment'
            elif self._config_data.get('news', {}).get('newsapi', {}).get('api_key'):
                return 'config_file'
            else:
                return 'not_found'
        
        return 'unknown'
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that required API keys are available.
        
        Returns:
            Dictionary mapping service to availability status
        """
        return {
            'fred': bool(self.get_fred_api_key()),
            'newsapi': bool(self.get_news_api_key('newsapi')),
            'fmp': bool(self.get_api_key('fmp')),
            'finnhub': bool(self.get_api_key('finnhub')),
            'eodhd': bool(self.get_api_key('eodhd')),
            'alpha_vantage': bool(self.get_api_key('alpha_vantage')),
        }
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Generic method to get API key for any service.
        
        Args:
            service: Service name (e.g., 'fmp', 'finnhub', 'eodhd', 'alpha_vantage')
            
        Returns:
            API key if found, None otherwise
        """
        # Try environment variable first (SERVICE_API_KEY format)
        env_key = f"{service.upper()}_API_KEY"
        if env_key in os.environ:
            return os.environ[env_key]
        
        # Try config file
        if self._config_data and service in self._config_data:
            api_key = self._config_data[service].get('api_key')
            if api_key and api_key != f"your_{service}_key_here":
                return api_key
        
        return None
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration for display.
        
        Returns:
            Configuration summary (API keys are masked)
        """
        summary = {
            'config_file': str(self.config_file),
            'config_file_exists': self.config_file.exists(),
            'api_keys': self.get_all_api_keys(),
            'validation': self.validate_api_keys()
        }
        
        return summary


# Global instance for easy access
_api_key_loader = None

def get_api_key_loader() -> APIKeyLoader:
    """Get the global API key loader instance."""
    global _api_key_loader
    if _api_key_loader is None:
        _api_key_loader = APIKeyLoader()
    return _api_key_loader


def get_fred_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Convenience function to get FRED API key."""
    return get_api_key_loader().get_fred_api_key(api_key)


def get_news_api_key(source: str = "newsapi", api_key: Optional[str] = None) -> Optional[str]:
    """Convenience function to get News API key."""
    return get_api_key_loader().get_news_api_key(source, api_key)