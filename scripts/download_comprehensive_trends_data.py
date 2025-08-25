#!/usr/bin/env python3
"""
Comprehensive Trends Data Collection System for 2Trade

This script downloads comprehensive Google Trends data including:
- Financial market trends (stocks, sectors, economic terms)
- Trade and tariff related searches  
- Economic sentiment indicators
- Global trade patterns
- Policy impact tracking

Author: 2Trade Trends Collection System
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
import yaml
from pytrends.request import TrendReq

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from consuela.config.instrument_list_loader import load_favorites_instruments, load_popular_instruments
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)

class ComprehensiveTrendsDataCollector:
    """
    Comprehensive trends data collector focusing on financial markets, trade, and tariffs.
    
    Features:
    - Stock and sector search trends
    - Trade and tariff search patterns
    - Economic sentiment indicators
    - Global trade policy trends
    - Supply chain disruption tracking
    """
    
    def __init__(self, data_root: str = "W:/market-data"):
        """Initialize the comprehensive trends data collector."""
        self.data_root = Path(data_root)
        self.trends_dir = self.data_root / "trends_data"
        self.trends_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Google Trends client
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.logger.info("Google Trends client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Trends client: {e}")
            self.pytrends = None
        
        # Load instrument lists
        self.favorites = self.load_instruments("favorites")
        self.popular = self.load_instruments("popular")
        self.all_instruments = list(set(self.favorites + self.popular))
        
        # Comprehensive trend categories
        self.trend_categories = {
            'financial_stocks': {
                'description': 'Individual stock search trends',
                'keywords': self.all_instruments[:30],  # Top 30 stocks
                'timeframe': 'today 3-m',
                'geo': 'US'
            },
            'financial_sectors': {
                'description': 'Financial sector trends',
                'keywords': [
                    'technology stocks', 'healthcare stocks', 'financial stocks',
                    'energy stocks', 'consumer stocks', 'industrial stocks',
                    'utilities stocks', 'materials stocks', 'real estate stocks',
                    'communication services stocks'
                ],
                'timeframe': 'today 3-m',
                'geo': 'US'
            },
            'trade_tariffs': {
                'description': 'Trade and tariff related searches',
                'keywords': [
                    'trade tariffs', 'import tariffs', 'china tariffs', 
                    'trade war', 'trade deficit', 'trade surplus',
                    'customs duties', 'import taxes', 'export restrictions',
                    'trade barriers', 'free trade agreement', 'trade negotiations',
                    'wto disputes', 'nafta trade', 'usmca trade',
                    'european trade', 'asia trade', 'tariff rates'
                ],
                'timeframe': 'today 12-m',
                'geo': 'US'
            },
            'economic_indicators': {
                'description': 'Economic sentiment and indicators',
                'keywords': [
                    'inflation rate', 'unemployment rate', 'gdp growth',
                    'interest rates', 'federal reserve', 'monetary policy',
                    'economic recession', 'economic recovery', 'fiscal policy',
                    'consumer confidence', 'business confidence', 'market volatility',
                    'stock market crash', 'bull market', 'bear market'
                ],
                'timeframe': 'today 6-m',
                'geo': 'US'
            },
            'supply_chain': {
                'description': 'Supply chain and logistics trends',
                'keywords': [
                    'supply chain disruption', 'shipping delays', 'port congestion',
                    'semiconductor shortage', 'raw materials shortage', 
                    'logistics costs', 'freight rates', 'shipping containers',
                    'supply chain crisis', 'global supply chain', 'manufacturing delays',
                    'inventory shortage', 'just in time manufacturing'
                ],
                'timeframe': 'today 6-m',
                'geo': ''  # Global
            },
            'global_trade_policy': {
                'description': 'Global trade policy and agreements',
                'keywords': [
                    'trade policy', 'international trade', 'bilateral trade',
                    'multilateral agreements', 'trade sanctions', 'economic sanctions',
                    'export controls', 'import quotas', 'antidumping duties',
                    'countervailing duties', 'trade remedy', 'safeguard measures',
                    'preferential trade', 'most favored nation', 'trade facilitation'
                ],
                'timeframe': 'today 12-m',
                'geo': ''  # Global
            },
            'commodity_trends': {
                'description': 'Commodity and raw material trends',
                'keywords': [
                    'oil prices', 'gold price', 'silver price', 'copper prices',
                    'wheat prices', 'corn prices', 'steel prices', 'aluminum prices',
                    'natural gas prices', 'coal prices', 'uranium prices',
                    'lithium prices', 'rare earth metals', 'agricultural commodities'
                ],
                'timeframe': 'today 6-m',
                'geo': ''  # Global
            },
            'crypto_finance': {
                'description': 'Cryptocurrency and fintech trends',
                'keywords': [
                    'bitcoin', 'ethereum', 'cryptocurrency', 'digital currency',
                    'blockchain', 'defi', 'nft', 'central bank digital currency',
                    'stablecoin', 'crypto regulation', 'crypto trading',
                    'fintech', 'digital payments', 'mobile banking'
                ],
                'timeframe': 'today 6-m',
                'geo': 'US'
            },
            'geopolitical_trade': {
                'description': 'Geopolitical events affecting trade',
                'keywords': [
                    'china us trade', 'europe trade', 'brexit trade',
                    'russia sanctions', 'iran sanctions', 'north korea sanctions',
                    'trade allies', 'trade partnerships', 'regional trade blocks',
                    'tpp agreement', 'rcep agreement', 'african trade'
                ],
                'timeframe': 'today 12-m',
                'geo': ''  # Global
            }
        }
        
        # Rate limiting settings
        self.request_delay = 2  # seconds between requests
        self.batch_delay = 60   # seconds between category batches
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = self.data_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"trends_collection_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
    
    def collect_trends_for_keywords(self, keywords: List[str], timeframe: str = 'today 3-m', 
                                  geo: str = 'US', category_name: str = '') -> Dict[str, Any]:
        """Collect Google Trends data for a list of keywords."""
        if not self.pytrends:
            return {'error': 'Google Trends client not available'}
        
        results = {
            'category': category_name,
            'keywords': keywords,
            'timeframe': timeframe,
            'geo': geo,
            'timestamp': datetime.now().isoformat(),
            'trends_data': {},
            'related_queries': {},
            'errors': []
        }
        
        # Process keywords in batches of 5 (Google Trends limit)
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            
            try:
                self.logger.info(f"Processing batch: {batch}")
                
                # Build payload for Google Trends
                self.pytrends.build_payload(
                    batch, 
                    cat=0, 
                    timeframe=timeframe,
                    geo=geo,
                    gprop=''
                )
                
                # Get interest over time
                interest_over_time = self.pytrends.interest_over_time()
                
                if not interest_over_time.empty:
                    # Convert to JSON-serializable format
                    for keyword in batch:
                        if keyword in interest_over_time.columns:
                            trend_data = []
                            for date, value in interest_over_time[keyword].items():
                                trend_data.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'search_score': int(value) if value == value else 0  # Handle NaN
                                })
                            
                            results['trends_data'][keyword] = trend_data
                
                # Get related queries for the first keyword in batch
                if batch:
                    try:
                        related_queries = self.pytrends.related_queries()
                        if related_queries and batch[0] in related_queries:
                            results['related_queries'][batch[0]] = {
                                'top': related_queries[batch[0]]['top'].to_dict('records') if related_queries[batch[0]]['top'] is not None else [],
                                'rising': related_queries[batch[0]]['rising'].to_dict('records') if related_queries[batch[0]]['rising'] is not None else []
                            }
                    except Exception as e:
                        self.logger.warning(f"Error getting related queries for {batch[0]}: {e}")
                
                # Rate limiting
                time.sleep(self.request_delay)
                
            except Exception as e:
                error_msg = f"Error processing batch {batch}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                time.sleep(5)  # Longer delay on error
        
        return results
    
    def collect_comprehensive_trends_data(self) -> Dict[str, Any]:
        """Collect comprehensive trends data across all categories."""
        self.logger.info("Starting comprehensive trends data collection")
        
        collection_report = {
            'timestamp': datetime.now().isoformat(),
            'categories_processed': 0,
            'keywords_processed': 0,
            'files_created': [],
            'errors': []
        }
        
        for category_name, category_config in self.trend_categories.items():
            try:
                self.logger.info(f"Processing category: {category_name}")
                
                # Collect trends for this category
                trends_result = self.collect_trends_for_keywords(
                    keywords=category_config['keywords'],
                    timeframe=category_config['timeframe'],
                    geo=category_config['geo'],
                    category_name=category_name
                )
                
                if 'error' not in trends_result:
                    # Save category data
                    timestamp = datetime.now().strftime('%Y%m%d')
                    filename = f"{category_name}_trends_{timestamp}.json"
                    filepath = self.trends_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(trends_result, f, indent=2, ensure_ascii=False)
                    
                    collection_report['files_created'].append(str(filepath))
                    collection_report['categories_processed'] += 1
                    collection_report['keywords_processed'] += len(category_config['keywords'])
                    
                    self.logger.info(f"Saved {filename} with {len(trends_result.get('trends_data', {}))} trends")
                
                # Add any errors from this category
                if trends_result.get('errors'):
                    collection_report['errors'].extend(trends_result['errors'])
                
                # Delay between categories to avoid rate limiting
                if category_name != list(self.trend_categories.keys())[-1]:  # Not the last category
                    self.logger.info(f"Waiting {self.batch_delay} seconds before next category...")
                    time.sleep(self.batch_delay)
                
            except Exception as e:
                error_msg = f"Error processing category {category_name}: {str(e)}"
                self.logger.error(error_msg)
                collection_report['errors'].append(error_msg)
        
        # Save collection report
        report_file = self.trends_dir / f"trends_collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(collection_report, f, indent=2)
        
        self.logger.info(f"Trends collection completed. Report saved: {report_file}")
        return collection_report
    
    def collect_tariff_trends_detailed(self) -> Dict[str, Any]:
        """Collect detailed tariff and trade trends data."""
        self.logger.info("Collecting detailed tariff and trade trends")
        
        # Extended tariff-specific keywords
        tariff_keywords = [
            # General tariffs
            'import tariffs', 'export tariffs', 'customs duties', 'trade duties',
            'tariff rates', 'tariff schedules', 'tariff classification',
            
            # Specific tariff types
            'steel tariffs', 'aluminum tariffs', 'solar panel tariffs',
            'automobile tariffs', 'agricultural tariffs', 'textile tariffs',
            'technology tariffs', 'semiconductor tariffs',
            
            # Geographic tariffs
            'china tariffs', 'eu tariffs', 'mexico tariffs', 'canada tariffs',
            'japan tariffs', 'india tariffs', 'south korea tariffs',
            
            # Trade policy terms
            'most favored nation', 'generalized system preferences',
            'free trade agreement', 'preferential trade agreement',
            'bilateral investment treaty', 'trade promotion authority',
            
            # Trade remedies
            'antidumping duties', 'countervailing duties', 'safeguard measures',
            'section 232 tariffs', 'section 301 tariffs', 'trade remedy investigations',
            
            # Economic impact
            'tariff impact', 'trade war effects', 'supply chain costs',
            'consumer prices tariffs', 'manufacturing costs tariffs'
        ]
        
        # Collect data with longer timeframes for policy analysis
        tariff_trends = self.collect_trends_for_keywords(
            keywords=tariff_keywords,
            timeframe='today 5-y',  # 5 year view for policy trends
            geo='US',
            category_name='detailed_tariff_trends'
        )
        
        # Save detailed tariff data
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"detailed_tariff_trends_{timestamp}.json"
        filepath = self.trends_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tariff_trends, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Detailed tariff trends saved to {filename}")
        return {
            'keywords_processed': len(tariff_keywords),
            'file_saved': str(filepath),
            'trends_collected': len(tariff_trends.get('trends_data', {})),
            'related_queries': len(tariff_trends.get('related_queries', {}))
        }
    
    def analyze_trends_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different trend categories."""
        trends_files = list(self.trends_dir.glob("*_trends_*.json"))
        
        if len(trends_files) < 2:
            return {'error': 'Not enough trend files for correlation analysis'}
        
        self.logger.info(f"Analyzing correlations across {len(trends_files)} trend files")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'files_analyzed': len(trends_files),
            'correlations': {},
            'trend_summary': {}
        }
        
        # Load and analyze trend files
        trend_data_by_category = {}
        
        for trend_file in trends_files:
            try:
                with open(trend_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                category = data.get('category', trend_file.stem)
                trends = data.get('trends_data', {})
                
                if trends:
                    trend_data_by_category[category] = trends
                    
                    # Summary statistics for each category
                    analysis['trend_summary'][category] = {
                        'keywords_count': len(trends),
                        'total_data_points': sum(len(keyword_data) for keyword_data in trends.values()),
                        'avg_search_score': self._calculate_avg_search_score(trends)
                    }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {trend_file}: {e}")
        
        # Calculate correlations (simplified analysis)
        for cat1 in trend_data_by_category:
            for cat2 in trend_data_by_category:
                if cat1 != cat2:
                    correlation_key = f"{cat1}_vs_{cat2}"
                    if correlation_key not in analysis['correlations']:
                        correlation_score = self._calculate_category_correlation(
                            trend_data_by_category[cat1],
                            trend_data_by_category[cat2]
                        )
                        analysis['correlations'][correlation_key] = correlation_score
        
        # Save analysis
        analysis_file = self.trends_dir / f"trends_correlation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _calculate_avg_search_score(self, trends_data: Dict[str, List]) -> float:
        """Calculate average search score across all keywords in category."""
        all_scores = []
        for keyword_data in trends_data.values():
            for data_point in keyword_data:
                score = data_point.get('search_score', 0)
                if score > 0:
                    all_scores.append(score)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    def _calculate_category_correlation(self, trends1: Dict, trends2: Dict) -> float:
        """Calculate correlation between two trend categories (simplified)."""
        # This is a simplified correlation calculation
        # In practice, you'd want more sophisticated time-series correlation
        
        scores1 = []
        scores2 = []
        
        for keyword1_data in trends1.values():
            for data_point in keyword1_data:
                scores1.append(data_point.get('search_score', 0))
        
        for keyword2_data in trends2.values():
            for data_point in keyword2_data:
                scores2.append(data_point.get('search_score', 0))
        
        if not scores1 or not scores2:
            return 0.0
        
        avg1 = sum(scores1) / len(scores1)
        avg2 = sum(scores2) / len(scores2)
        
        # Simple correlation proxy based on average differences
        return abs(avg1 - avg2) / max(avg1, avg2, 1)

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Trends Data Collection")
    parser.add_argument('--data-root', type=str, help='Custom data root directory')
    parser.add_argument('--tariff-only', action='store_true', help='Collect only detailed tariff trends')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing trends data')
    parser.add_argument('--category', type=str, help='Collect specific category only')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE TRENDS DATA COLLECTION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    try:
        collector = ComprehensiveTrendsDataCollector(
            data_root=args.data_root or "W:/market-data"
        )
        
        if args.analyze:
            print("\nANALYZING: Existing trends data...")
            analysis = collector.analyze_trends_correlations()
            print(f"Analysis Results:")
            print(f"  Files analyzed: {analysis.get('files_analyzed', 0)}")
            print(f"  Categories found: {len(analysis.get('trend_summary', {}))}")
            print(f"  Correlations calculated: {len(analysis.get('correlations', {}))}")
            
        elif args.tariff_only:
            print("\nCOLLECTING: Detailed tariff and trade trends...")
            result = collector.collect_tariff_trends_detailed()
            print(f"Tariff Trends Results:")
            print(f"  Keywords processed: {result.get('keywords_processed', 0)}")
            print(f"  Trends collected: {result.get('trends_collected', 0)}")
            print(f"  Related queries: {result.get('related_queries', 0)}")
            
        elif args.category:
            print(f"\nCOLLECTING: {args.category} trends only...")
            if args.category in collector.trend_categories:
                category_config = collector.trend_categories[args.category]
                result = collector.collect_trends_for_keywords(
                    keywords=category_config['keywords'],
                    timeframe=category_config['timeframe'],
                    geo=category_config['geo'],
                    category_name=args.category
                )
                print(f"Category Results:")
                print(f"  Keywords: {len(category_config['keywords'])}")
                print(f"  Trends collected: {len(result.get('trends_data', {}))}")
                print(f"  Errors: {len(result.get('errors', []))}")
            else:
                print(f"ERROR: Category '{args.category}' not found")
                print(f"Available categories: {list(collector.trend_categories.keys())}")
            
        else:
            print("\nCOLLECTING: Comprehensive trends data across all categories...")
            result = collector.collect_comprehensive_trends_data()
            print(f"Collection Results:")
            print(f"  Categories processed: {result.get('categories_processed', 0)}")
            print(f"  Keywords processed: {result.get('keywords_processed', 0)}")
            print(f"  Files created: {len(result.get('files_created', []))}")
            print(f"  Errors: {len(result.get('errors', []))}")
        
        print(f"\nSUCCESS: Trends data collection completed!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()