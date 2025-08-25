#!/usr/bin/env python3
"""
Real Financial News Data Downloader for 2Trade System

Downloads real financial news from multiple sources:
- NewsAPI (1,000 requests/day)
- Finnhub (60 calls/minute) 
- Alpha Vantage (25 requests/day)

Replaces dummy news data with real financial news including sentiment analysis.
"""

import os
import sys
import json
import time
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from textblob import TextBlob
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from consuela.config.instrument_list_loader import load_favorites_instruments

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealNewsDataDownloader:
    """Downloads real financial news data from multiple APIs"""
    
    def __init__(self, data_root: str = "W:/market-data"):
        self.data_root = Path(data_root)
        self.news_dir = self.data_root / "news_data"
        self.news_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API configuration
        self.config = self._load_config()
        
        # API clients
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': '2Trade Financial News Downloader 1.0'
        })
        
        # Rate limiting trackers
        self.last_request_time = {}
        self.request_counts = {}
        
        logger.info(f"Initialized news downloader with data root: {data_root}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration"""
        config_path = project_root / "consuela" / "config" / "api_keys.yml"
        corporate_config_path = project_root / "consuela" / "config" / "corporate_data_sources.yml"
        
        # Load API keys
        with open(config_path, 'r') as f:
            api_keys = yaml.safe_load(f)
        
        # Load corporate data sources config
        with open(corporate_config_path, 'r') as f:
            sources_config = yaml.safe_load(f)
        
        return {
            'api_keys': api_keys,
            'sources': sources_config
        }
    
    def _rate_limit_delay(self, source: str, min_delay: float = 0.1) -> None:
        """Apply rate limiting delays"""
        current_time = time.time()
        
        if source in self.last_request_time:
            time_since_last = current_time - self.last_request_time[source]
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time[source] = time.time()
    
    def _get_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Convert to label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                'sentiment_score': polarity,
                'sentiment_label': label,
                'confidence': abs(polarity)  # Use absolute value as confidence proxy
            }
        except:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0
            }
    
    def download_newsapi_data(self, symbols: List[str], days_back: int = 7) -> Dict[str, List[Dict]]:
        """Download news from NewsAPI"""
        logger.info(f"Downloading NewsAPI data for {len(symbols)} symbols")
        
        api_key = self.config['api_keys']['news']['newsapi']['api_key']
        base_url = "https://newsapi.org/v2/everything"
        
        news_by_symbol = {}
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                self._rate_limit_delay('newsapi', 0.1)  # 10 requests per second max
                
                # Search for company news
                params = {
                    'q': f'{symbol} OR "{self._get_company_name(symbol)}"',
                    'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,finance.yahoo.com,fool.com,seekingalpha.com',
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': api_key,
                    'pageSize': 20  # Limit to avoid hitting daily quota too fast
                }
                
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    symbol_news = []
                    
                    for idx, article in enumerate(articles):
                        # Clean and process article
                        headline = article.get('title', '').strip()
                        description = article.get('description', '').strip()
                        content = f"{headline}. {description}"
                        
                        # Skip if too short or generic
                        if len(headline) < 20 or 'stock' not in content.lower():
                            continue
                        
                        # Get sentiment
                        sentiment = self._get_sentiment_analysis(content)
                        
                        news_item = {
                            'date': article.get('publishedAt', '').split('T')[0],
                            'symbol': symbol,
                            'headline': headline,
                            'description': description,
                            'url': article.get('url'),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'sentiment_score': sentiment['sentiment_score'],
                            'sentiment_label': sentiment['sentiment_label'],
                            'confidence': sentiment['confidence'],
                            'article_id': f"{symbol}_newsapi_{int(time.time())}_{idx}"
                        }
                        
                        symbol_news.append(news_item)
                    
                    news_by_symbol[symbol] = symbol_news
                    logger.info(f"Downloaded {len(symbol_news)} news articles for {symbol}")
                
                else:
                    logger.warning(f"NewsAPI error for {symbol}: {data.get('message')}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error downloading NewsAPI data for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Error processing NewsAPI data for {symbol}: {e}")
        
        return news_by_symbol
    
    def download_finnhub_news(self, symbols: List[str], days_back: int = 7) -> Dict[str, List[Dict]]:
        """Download company news from Finnhub"""
        logger.info(f"Downloading Finnhub news for {len(symbols)} symbols")
        
        api_key = self.config['api_keys']['finnhub']['api_key']
        base_url = "https://finnhub.io/api/v1/company-news"
        
        news_by_symbol = {}
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                self._rate_limit_delay('finnhub', 1.0)  # 60 calls per minute = 1 per second
                
                params = {
                    'symbol': symbol,
                    'from': from_date,
                    'to': to_date,
                    'token': api_key
                }
                
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                articles = response.json()
                
                symbol_news = []
                
                for idx, article in enumerate(articles[:15]):  # Limit to 15 per symbol
                    headline = article.get('headline', '').strip()
                    summary = article.get('summary', '').strip()
                    content = f"{headline}. {summary}"
                    
                    if len(headline) < 20:
                        continue
                    
                    # Get sentiment
                    sentiment = self._get_sentiment_analysis(content)
                    
                    # Convert timestamp
                    timestamp = article.get('datetime', 0)
                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else to_date
                    
                    news_item = {
                        'date': date,
                        'symbol': symbol,
                        'headline': headline,
                        'summary': summary,
                        'url': article.get('url'),
                        'source': article.get('source', 'Finnhub'),
                        'sentiment_score': sentiment['sentiment_score'],
                        'sentiment_label': sentiment['sentiment_label'],
                        'confidence': sentiment['confidence'],
                        'article_id': f"{symbol}_finnhub_{timestamp}_{idx}"
                    }
                    
                    symbol_news.append(news_item)
                
                news_by_symbol[symbol] = symbol_news
                logger.info(f"Downloaded {len(symbol_news)} Finnhub articles for {symbol}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error downloading Finnhub data for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Error processing Finnhub data for {symbol}: {e}")
        
        return news_by_symbol
    
    def download_alpha_vantage_news(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Download news sentiment from Alpha Vantage"""
        logger.info(f"Downloading Alpha Vantage news sentiment for {len(symbols)} symbols")
        
        api_key = self.config['api_keys']['alpha_vantage']['api_key']
        base_url = "https://www.alphavantage.co/query"
        
        news_by_symbol = {}
        
        for symbol in symbols[:5]:  # Limit to 5 symbols due to daily API limit
            try:
                self._rate_limit_delay('alpha_vantage', 12.0)  # 25 requests/day = 1 per ~12 seconds safely
                
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': api_key,
                    'limit': 20
                }
                
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'feed' in data:
                    articles = data['feed']
                    symbol_news = []
                    
                    for idx, article in enumerate(articles[:10]):  # Limit per symbol
                        headline = article.get('title', '').strip()
                        summary = article.get('summary', '').strip()
                        
                        if len(headline) < 20:
                            continue
                        
                        # Use Alpha Vantage sentiment if available
                        ticker_sentiments = article.get('ticker_sentiment', [])
                        av_sentiment = None
                        for ts in ticker_sentiments:
                            if ts.get('ticker') == symbol:
                                av_sentiment = ts
                                break
                        
                        if av_sentiment:
                            sentiment_score = float(av_sentiment.get('relevance_score', 0)) * float(av_sentiment.get('ticker_sentiment_score', 0))
                            sentiment_label = av_sentiment.get('ticker_sentiment_label', 'neutral').lower()
                        else:
                            # Fallback to TextBlob
                            sentiment = self._get_sentiment_analysis(f"{headline}. {summary}")
                            sentiment_score = sentiment['sentiment_score']
                            sentiment_label = sentiment['sentiment_label']
                        
                        news_item = {
                            'date': article.get('time_published', '')[:8],  # YYYYMMDD format
                            'symbol': symbol,
                            'headline': headline,
                            'summary': summary,
                            'url': article.get('url'),
                            'source': article.get('source', 'Alpha Vantage'),
                            'sentiment_score': sentiment_score,
                            'sentiment_label': sentiment_label,
                            'confidence': float(av_sentiment.get('relevance_score', 0.5)) if av_sentiment else 0.5,
                            'article_id': f"{symbol}_alphavantage_{article.get('time_published', '')}_{idx}"
                        }
                        
                        symbol_news.append(news_item)
                    
                    news_by_symbol[symbol] = symbol_news
                    logger.info(f"Downloaded {len(symbol_news)} Alpha Vantage articles for {symbol}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error downloading Alpha Vantage data for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Error processing Alpha Vantage data for {symbol}: {e}")
        
        return news_by_symbol
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for better news search"""
        company_names = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla Motors',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix',
            'INTC': 'Intel Corporation'
        }
        return company_names.get(symbol, symbol)
    
    def save_news_data(self, news_by_symbol: Dict[str, List[Dict]], source_name: str) -> None:
        """Save news data to JSON files"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        for symbol, news_list in news_by_symbol.items():
            if not news_list:
                continue
                
            filename = self.news_dir / f"{symbol}_news.json"
            
            # Load existing data if it exists
            existing_data = []
            if filename.exists():
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []
            
            # Add new data (avoiding duplicates by article_id)
            existing_ids = {item.get('article_id') for item in existing_data}
            new_articles = [article for article in news_list if article.get('article_id') not in existing_ids]
            
            if new_articles:
                combined_data = existing_data + new_articles
                
                # Sort by date (newest first)
                combined_data.sort(key=lambda x: x.get('date', ''), reverse=True)
                
                # Keep only last 90 days of data
                cutoff_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                combined_data = [item for item in combined_data if item.get('date', '') >= cutoff_date]
                
                # Save updated data
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {len(new_articles)} new articles for {symbol} from {source_name}")
    
    def download_all_news_sources(self, symbols: Optional[List[str]] = None, days_back: int = 7) -> None:
        """Download news from all available sources"""
        if symbols is None:
            # Load favorite instruments
            instruments = load_favorites_instruments()
            symbols = [inst['symbol'] for inst in instruments[:30]]  # Limit to 30 symbols
        
        logger.info(f"Starting comprehensive news download for {len(symbols)} symbols")
        
        # Download from NewsAPI
        try:
            newsapi_data = self.download_newsapi_data(symbols, days_back)
            self.save_news_data(newsapi_data, 'NewsAPI')
        except Exception as e:
            logger.error(f"Failed to download NewsAPI data: {e}")
        
        # Download from Finnhub
        try:
            finnhub_data = self.download_finnhub_news(symbols, days_back)
            self.save_news_data(finnhub_data, 'Finnhub')
        except Exception as e:
            logger.error(f"Failed to download Finnhub data: {e}")
        
        # Download from Alpha Vantage (limited symbols due to quota)
        try:
            alpha_data = self.download_alpha_vantage_news(symbols[:5])  # Only top 5 due to rate limit
            self.save_news_data(alpha_data, 'AlphaVantage')
        except Exception as e:
            logger.error(f"Failed to download Alpha Vantage data: {e}")
        
        logger.info("Completed comprehensive news download")
    
    def create_download_report(self) -> Dict[str, Any]:
        """Create a report of downloaded news data"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbols_processed': 0,
            'total_articles': 0,
            'sources': {},
            'date_range': {},
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        
        # Analyze existing news files
        for news_file in self.news_dir.glob("*_news.json"):
            if news_file.name.startswith('news_'):  # Skip aggregate files
                continue
                
            try:
                with open(news_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                symbol = news_file.stem.replace('_news', '')
                report['symbols_processed'] += 1
                report['total_articles'] += len(articles)
                
                for article in articles:
                    # Track sources
                    source = article.get('source', 'unknown')
                    report['sources'][source] = report['sources'].get(source, 0) + 1
                    
                    # Track sentiment
                    sentiment = article.get('sentiment_label', 'neutral')
                    if sentiment in report['sentiment_distribution']:
                        report['sentiment_distribution'][sentiment] += 1
                    
                    # Track date range
                    date = article.get('date', '')
                    if date:
                        if 'earliest' not in report['date_range'] or date < report['date_range']['earliest']:
                            report['date_range']['earliest'] = date
                        if 'latest' not in report['date_range'] or date > report['date_range']['latest']:
                            report['date_range']['latest'] = date
            
            except Exception as e:
                logger.error(f"Error analyzing {news_file}: {e}")
        
        # Save report
        report_file = self.news_dir / f"news_download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real financial news data")
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to download')
    parser.add_argument('--days-back', type=int, default=7, help='Days of historical news to download')
    parser.add_argument('--data-root', default='W:/market-data', help='Data root directory')
    parser.add_argument('--sources', nargs='+', choices=['newsapi', 'finnhub', 'alphavantage'], 
                       help='Specific sources to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("REAL FINANCIAL NEWS DATA DOWNLOADER")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    try:
        downloader = RealNewsDataDownloader(args.data_root)
        
        # Download news data
        downloader.download_all_news_sources(
            symbols=args.symbols,
            days_back=args.days_back
        )
        
        # Generate report
        report = downloader.create_download_report()
        
        print(f"\nDownload completed successfully!")
        print(f"Symbols processed: {report['symbols_processed']}")
        print(f"Total articles: {report['total_articles']}")
        print(f"Sources: {list(report['sources'].keys())}")
        print(f"Date range: {report['date_range'].get('earliest')} to {report['date_range'].get('latest')}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()