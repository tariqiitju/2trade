#!/usr/bin/env python3
"""
Comprehensive Earnings Data Collection System for 2Trade

This script downloads and processes earnings data from multiple sources including:
- Earnings calendars and forecasts
- Historical earnings reports
- Earnings estimates and revisions
- EPS data and guidance
- Conference call transcripts summaries

Author: 2Trade Data Collection System
Created: 2025-08-24
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
import requests
import pandas as pd
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from consuela.config.instrument_list_loader import load_favorites_instruments, load_popular_instruments
    from consuela.config.data_config_loader import load_data_config
    from consuela.config.api_key_loader import get_api_key_loader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)

class ComprehensiveEarningsDataCollector:
    """
    Comprehensive earnings data collection system supporting multiple APIs and data sources.
    
    Features:
    - Earnings calendars (upcoming and historical)
    - Earnings estimates and revisions  
    - EPS data and guidance
    - Conference call transcripts
    - Earnings surprise analysis
    """
    
    def __init__(self, data_root: str = "W:/market-data", config_path: Optional[str] = None):
        """Initialize the comprehensive earnings data collector."""
        self.data_root = Path(data_root)
        self.earnings_dir = self.data_root / "earnings_data"
        self.earnings_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize API clients
        self.api_key_loader = get_api_key_loader()
        self.api_keys = {
            'finnhub': {'api_key': self.api_key_loader.get_api_key('finnhub')},
            'alpha_vantage': {'api_key': self.api_key_loader.get_api_key('alpha_vantage')},
            'fmp': {'api_key': self.api_key_loader.get_api_key('fmp')},
            'eodhd': {'api_key': self.api_key_loader.get_api_key('eodhd')}
        }
        
        # Load instrument lists
        self.favorites = self.load_instruments("favorites")
        self.popular = self.load_instruments("popular") 
        self.all_instruments = list(set(self.favorites + self.popular))
        
        # Data collection settings
        self.rate_limits = {
            'finnhub': 60,      # calls per minute
            'alpha_vantage': 5,  # calls per minute
            'polygon': 100,      # calls per minute
            'yahoofinance': 2000  # calls per hour
        }
        
        # Data types to collect
        self.earnings_data_types = {
            'earnings_calendar': {
                'description': 'Upcoming and historical earnings dates',
                'apis': ['finnhub', 'alpha_vantage'],
                'frequency': 'daily'
            },
            'earnings_estimates': {
                'description': 'Analyst earnings estimates and revisions',
                'apis': ['finnhub', 'alpha_vantage'],
                'frequency': 'weekly'
            },
            'earnings_history': {
                'description': 'Historical earnings reports and EPS',
                'apis': ['finnhub', 'alpha_vantage', 'polygon'],
                'frequency': 'quarterly'
            },
            'earnings_surprises': {
                'description': 'Earnings surprise analysis',
                'apis': ['finnhub'],
                'frequency': 'quarterly'
            },
            'earnings_guidance': {
                'description': 'Company earnings guidance',
                'apis': ['alpha_vantage'],
                'frequency': 'quarterly'
            }
        }
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = self.data_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"earnings_collection_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load data collection configuration."""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Try to load default config
                default_config = project_root / "consuela" / "config" / "data-config.yml"
                if default_config.exists():
                    with open(default_config, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    self.logger.warning("No configuration file found, using defaults")
                    return self.get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'api_keys': {
                'finnhub': {'api_key': ''},
                'alpha_vantage': {'api_key': ''},
                'polygon': {'api_key': ''},
                'yahoofinance': {}  # No API key required
            }
        }
    
    def load_instruments(self, list_type: str) -> List[str]:
        """Load instrument symbols."""
        try:
            if list_type == "favorites":
                instruments = load_favorites_instruments()
            elif list_type == "popular":
                instruments = load_popular_instruments()
            else:
                return []
            return [instr["symbol"] for instr in instruments]
        except Exception as e:
            self.logger.warning(f"Failed to load {list_type} instruments: {e}")
            return []
    
    def download_earnings_calendar_finnhub(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Download earnings calendar from Finnhub API."""
        api_key = self.api_keys.get('finnhub', {}).get('api_key')
        if not api_key:
            self.logger.warning("Finnhub API key not configured")
            return []
        
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            'token': api_key,
            'from': start_date,
            'to': end_date
        }
        
        try:
            self.logger.info(f"Downloading earnings calendar from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            earnings_events = data.get('earningsCalendar', [])
            
            # Filter for our instruments
            filtered_events = [
                event for event in earnings_events
                if event.get('symbol') in self.all_instruments
            ]
            
            self.logger.info(f"Retrieved {len(filtered_events)} earnings events for tracked symbols")
            return filtered_events
            
        except Exception as e:
            self.logger.error(f"Error downloading Finnhub earnings calendar: {e}")
            return []
    
    def download_earnings_estimates_finnhub(self, symbol: str) -> Dict[str, Any]:
        """Download earnings estimates for a symbol from Finnhub."""
        api_key = self.api_keys.get('finnhub', {}).get('api_key')
        if not api_key:
            return {}
        
        url = "https://finnhub.io/api/v1/stock/earnings"
        params = {
            'symbol': symbol,
            'token': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return {
                'symbol': symbol,
                'estimates': data,
                'timestamp': datetime.now().isoformat(),
                'source': 'finnhub'
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading earnings estimates for {symbol}: {e}")
            return {}
    
    def download_earnings_history_alpha_vantage(self, symbol: str) -> Dict[str, Any]:
        """Download historical earnings from Alpha Vantage."""
        api_key = self.api_keys.get('alpha_vantage', {}).get('api_key')
        if not api_key:
            return {}
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API limit
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage API limit reached for {symbol}")
                return {}
            
            return {
                'symbol': symbol,
                'earnings_history': data,
                'timestamp': datetime.now().isoformat(),
                'source': 'alpha_vantage'
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading earnings history for {symbol}: {e}")
            return {}
    
    def download_earnings_surprises_finnhub(self, symbol: str) -> Dict[str, Any]:
        """Download earnings surprises for a symbol."""
        api_key = self.api_keys.get('finnhub', {}).get('api_key')
        if not api_key:
            return {}
        
        url = "https://finnhub.io/api/v1/stock/earnings"
        params = {
            'symbol': symbol,
            'token': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Process earnings surprises
            surprises = []
            for earning in data:
                if 'actual' in earning and 'estimate' in earning:
                    surprise = {
                        'date': earning.get('date'),
                        'period': earning.get('period'),
                        'actual': earning.get('actual'),
                        'estimate': earning.get('estimate'),
                        'surprise': earning.get('actual', 0) - earning.get('estimate', 0),
                        'surprise_percent': ((earning.get('actual', 0) - earning.get('estimate', 0)) / earning.get('estimate', 1)) * 100 if earning.get('estimate', 0) != 0 else 0
                    }
                    surprises.append(surprise)
            
            return {
                'symbol': symbol,
                'earnings_surprises': surprises,
                'timestamp': datetime.now().isoformat(),
                'source': 'finnhub'
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading earnings surprises for {symbol}: {e}")
            return {}
    
    def collect_comprehensive_earnings_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Collect comprehensive earnings data for all or specified symbols."""
        if symbols is None:
            symbols = self.all_instruments[:50]  # Limit to first 50 for API quotas
        
        self.logger.info(f"Starting comprehensive earnings data collection for {len(symbols)} symbols")
        
        collection_report = {
            'timestamp': datetime.now().isoformat(),
            'symbols_processed': 0,
            'data_collected': {
                'earnings_calendar': 0,
                'earnings_estimates': 0,
                'earnings_history': 0,
                'earnings_surprises': 0
            },
            'errors': [],
            'api_calls_made': 0
        }
        
        # 1. Download earnings calendar (covers multiple symbols)
        try:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
            
            calendar_data = self.download_earnings_calendar_finnhub(start_date, end_date)
            if calendar_data:
                calendar_file = self.earnings_dir / f"earnings_calendar_{start_date}_{end_date}.json"
                with open(calendar_file, 'w') as f:
                    json.dump(calendar_data, f, indent=2)
                collection_report['data_collected']['earnings_calendar'] = len(calendar_data)
                collection_report['api_calls_made'] += 1
        except Exception as e:
            collection_report['errors'].append(f"Earnings calendar error: {str(e)}")
        
        # 2. Download individual symbol data
        for symbol in symbols:
            try:
                self.logger.info(f"Processing earnings data for {symbol}")
                symbol_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'data_sources': {}
                }
                
                # Earnings estimates
                estimates_data = self.download_earnings_estimates_finnhub(symbol)
                if estimates_data:
                    symbol_data['data_sources']['estimates'] = estimates_data
                    collection_report['api_calls_made'] += 1
                    time.sleep(1)  # Rate limiting
                
                # Earnings history 
                history_data = self.download_earnings_history_alpha_vantage(symbol)
                if history_data:
                    symbol_data['data_sources']['history'] = history_data
                    collection_report['api_calls_made'] += 1
                    time.sleep(12)  # Alpha Vantage rate limiting (5 calls/minute)
                
                # Earnings surprises
                surprises_data = self.download_earnings_surprises_finnhub(symbol)
                if surprises_data:
                    symbol_data['data_sources']['surprises'] = surprises_data
                    collection_report['api_calls_made'] += 1
                    time.sleep(1)  # Rate limiting
                
                # Save symbol-specific data
                if symbol_data['data_sources']:
                    symbol_file = self.earnings_dir / f"{symbol}_earnings.json"
                    with open(symbol_file, 'w') as f:
                        json.dump(symbol_data, f, indent=2)
                    
                    collection_report['symbols_processed'] += 1
                    
                    # Update counters
                    if 'estimates' in symbol_data['data_sources']:
                        collection_report['data_collected']['earnings_estimates'] += 1
                    if 'history' in symbol_data['data_sources']:
                        collection_report['data_collected']['earnings_history'] += 1
                    if 'surprises' in symbol_data['data_sources']:
                        collection_report['data_collected']['earnings_surprises'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                self.logger.error(error_msg)
                collection_report['errors'].append(error_msg)
        
        # Save collection report
        report_file = self.earnings_dir / f"earnings_collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(collection_report, f, indent=2)
        
        self.logger.info(f"Earnings data collection completed. Report saved: {report_file}")
        return collection_report
    
    def collect_upcoming_earnings(self, days_ahead: int = 30) -> Dict[str, Any]:
        """Collect upcoming earnings for the next N days."""
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        self.logger.info(f"Collecting upcoming earnings from {start_date} to {end_date}")
        
        calendar_data = self.download_earnings_calendar_finnhub(start_date, end_date)
        
        if calendar_data:
            # Process and enhance the data
            processed_data = []
            for event in calendar_data:
                enhanced_event = {
                    'symbol': event.get('symbol'),
                    'date': event.get('date'),
                    'hour': event.get('hour', 'Unknown'),
                    'quarter': event.get('quarter'),
                    'year': event.get('year'),
                    'eps_estimate': event.get('epsEstimate'),
                    'eps_actual': event.get('epsActual'),
                    'revenue_estimate': event.get('revenueEstimate'),
                    'revenue_actual': event.get('revenueActual'),
                    'processed_timestamp': datetime.now().isoformat()
                }
                processed_data.append(enhanced_event)
            
            # Save processed data
            output_file = self.earnings_dir / f"upcoming_earnings_{start_date}_{end_date}.json"
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            self.logger.info(f"Saved {len(processed_data)} upcoming earnings events")
            return {
                'events_found': len(processed_data),
                'file_saved': str(output_file),
                'date_range': f"{start_date} to {end_date}"
            }
        
        return {'events_found': 0}
    
    def analyze_earnings_trends(self) -> Dict[str, Any]:
        """Analyze earnings trends from collected data."""
        earnings_files = list(self.earnings_dir.glob("*_earnings.json"))
        
        if not earnings_files:
            return {'error': 'No earnings data files found'}
        
        analysis = {
            'symbols_analyzed': 0,
            'earnings_patterns': {},
            'surprise_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for earnings_file in earnings_files:
            try:
                with open(earnings_file, 'r') as f:
                    data = json.load(f)
                
                symbol = data.get('symbol')
                if not symbol:
                    continue
                
                analysis['symbols_analyzed'] += 1
                
                # Analyze earnings patterns
                if 'history' in data.get('data_sources', {}):
                    history = data['data_sources']['history']
                    # Pattern analysis logic here
                    analysis['earnings_patterns'][symbol] = {
                        'data_available': True,
                        'quarters_analyzed': len(history.get('quarterlyEarnings', [])),
                        'annual_analyzed': len(history.get('annualEarnings', []))
                    }
                
                # Analyze surprises
                if 'surprises' in data.get('data_sources', {}):
                    surprises = data['data_sources']['surprises']
                    if surprises.get('earnings_surprises'):
                        surprise_count = len(surprises['earnings_surprises'])
                        positive_surprises = len([s for s in surprises['earnings_surprises'] if s.get('surprise', 0) > 0])
                        analysis['surprise_analysis'][symbol] = {
                            'total_quarters': surprise_count,
                            'positive_surprises': positive_surprises,
                            'surprise_rate': positive_surprises / surprise_count if surprise_count > 0 else 0
                        }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {earnings_file}: {e}")
        
        # Save analysis
        analysis_file = self.earnings_dir / f"earnings_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Earnings Data Collection")
    parser.add_argument('--data-root', type=str, help='Custom data root directory')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to collect')
    parser.add_argument('--upcoming-only', action='store_true', help='Collect only upcoming earnings')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing earnings data')
    parser.add_argument('--days-ahead', type=int, default=30, help='Days ahead for upcoming earnings')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE EARNINGS DATA COLLECTION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    try:
        collector = ComprehensiveEarningsDataCollector(
            data_root=args.data_root or "W:/market-data"
        )
        
        if args.analyze:
            print("\nANALYZING: Existing earnings data...")
            analysis = collector.analyze_earnings_trends()
            print(f"Analysis Results:")
            print(f"  Symbols analyzed: {analysis.get('symbols_analyzed', 0)}")
            print(f"  Patterns found: {len(analysis.get('earnings_patterns', {}))}")
            print(f"  Surprise analysis: {len(analysis.get('surprise_analysis', {}))}")
            
        elif args.upcoming_only:
            print(f"\nCOLLECTING: Upcoming earnings for next {args.days_ahead} days...")
            result = collector.collect_upcoming_earnings(args.days_ahead)
            print(f"Upcoming Earnings Results:")
            print(f"  Events found: {result.get('events_found', 0)}")
            print(f"  Date range: {result.get('date_range', 'N/A')}")
            
        else:
            print("\nCOLLECTING: Comprehensive earnings data...")
            result = collector.collect_comprehensive_earnings_data(args.symbols)
            print(f"Collection Results:")
            print(f"  Symbols processed: {result.get('symbols_processed', 0)}")
            print(f"  Calendar events: {result.get('data_collected', {}).get('earnings_calendar', 0)}")
            print(f"  Estimates collected: {result.get('data_collected', {}).get('earnings_estimates', 0)}")
            print(f"  History collected: {result.get('data_collected', {}).get('earnings_history', 0)}")
            print(f"  API calls made: {result.get('api_calls_made', 0)}")
            print(f"  Errors: {len(result.get('errors', []))}")
        
        print(f"\nSUCCESS: Earnings data collection completed!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()