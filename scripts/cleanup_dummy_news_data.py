#!/usr/bin/env python3
"""
Remove ALL Dummy News Data from 2Trade System

This script removes all dummy "sample_news_generator" entries from news files,
keeping only REAL financial news data for model training and feature selection.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsDataCleaner:
    """Removes dummy data and keeps only real financial news"""
    
    def __init__(self, data_root: str = "W:/market-data"):
        self.data_root = Path(data_root)
        self.news_dir = self.data_root / "news_data"
        
    def clean_all_news_files(self) -> Dict[str, Any]:
        """Clean ALL news files by removing dummy data"""
        
        logger.info("Starting comprehensive news data cleanup...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': 0,
            'symbols_cleaned': [],
            'dummy_articles_removed': 0,
            'real_articles_kept': 0,
            'files_with_no_real_data': [],
            'sources_found': set()
        }
        
        # Process all news files
        for news_file in self.news_dir.glob("*_news.json"):
            if news_file.name.startswith('news_'):  # Skip aggregate files
                continue
                
            symbol = news_file.stem.replace('_news', '')
            logger.info(f"Cleaning {symbol}...")
            
            result = self._clean_single_file(news_file)
            
            if result:
                report['files_processed'] += 1
                report['symbols_cleaned'].append(symbol)
                report['dummy_articles_removed'] += result['dummy_removed']
                report['real_articles_kept'] += result['real_kept']
                
                if result['real_kept'] == 0:
                    report['files_with_no_real_data'].append(symbol)
                
                # Track sources
                report['sources_found'].update(result['sources'])
        
        # Convert set to list for JSON serialization
        report['sources_found'] = list(report['sources_found'])
        
        # Save cleanup report
        report_file = self.news_dir / f"news_cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cleanup completed! Report saved to: {report_file}")
        return report
    
    def _clean_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Clean a single news file"""
        
        try:
            # Load existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            original_count = len(articles)
            
            # Filter out dummy data - keep only REAL news
            real_articles = []
            sources_found = set()
            
            for article in articles:
                source = article.get('source', '')
                headline = article.get('headline', '')
                
                # Remove dummy data conditions
                is_dummy = (
                    source == 'sample_news_generator' or
                    'Sample news headline' in headline or
                    headline.startswith('Sample news headline')
                )
                
                if not is_dummy:
                    real_articles.append(article)
                    sources_found.add(source)
            
            # Sort real articles by date (newest first)
            real_articles.sort(key=lambda x: x.get('date', ''), reverse=True)
            
            # Save cleaned data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(real_articles, f, indent=2, ensure_ascii=False)
            
            dummy_removed = original_count - len(real_articles)
            real_kept = len(real_articles)
            
            logger.info(f"  {file_path.stem}: Removed {dummy_removed} dummy articles, kept {real_kept} real articles")
            
            return {
                'dummy_removed': dummy_removed,
                'real_kept': real_kept,
                'sources': sources_found
            }
            
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {e}")
            return None
    
    def analyze_news_quality(self) -> Dict[str, Any]:
        """Analyze the quality of real news data for model training"""
        
        logger.info("Analyzing real news data quality...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'symbols_with_real_news': 0,
            'symbols_without_news': [],
            'total_real_articles': 0,
            'source_distribution': {},
            'date_coverage': {},
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'average_articles_per_symbol': 0,
            'symbols_with_good_coverage': [],  # 10+ articles
            'symbols_with_poor_coverage': []   # <5 articles
        }
        
        symbols_data = []
        
        for news_file in self.news_dir.glob("*_news.json"):
            if news_file.name.startswith('news_'):  # Skip aggregate files
                continue
                
            symbol = news_file.stem.replace('_news', '')
            
            try:
                with open(news_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                if articles:
                    analysis['symbols_with_real_news'] += 1
                    analysis['total_real_articles'] += len(articles)
                    symbols_data.append((symbol, len(articles)))
                    
                    # Track coverage
                    if len(articles) >= 10:
                        analysis['symbols_with_good_coverage'].append(symbol)
                    elif len(articles) < 5:
                        analysis['symbols_with_poor_coverage'].append(symbol)
                    
                    # Analyze articles
                    for article in articles:
                        # Source distribution
                        source = article.get('source', 'unknown')
                        analysis['source_distribution'][source] = analysis['source_distribution'].get(source, 0) + 1
                        
                        # Sentiment distribution
                        sentiment = article.get('sentiment_label', 'neutral')
                        if sentiment in analysis['sentiment_distribution']:
                            analysis['sentiment_distribution'][sentiment] += 1
                        
                        # Date coverage
                        date = article.get('date', '')[:7]  # YYYY-MM format
                        analysis['date_coverage'][date] = analysis['date_coverage'].get(date, 0) + 1
                
                else:
                    analysis['symbols_without_news'].append(symbol)
                    
            except Exception as e:
                logger.error(f"Error analyzing {news_file}: {e}")
                analysis['symbols_without_news'].append(symbol)
        
        # Calculate averages
        if analysis['symbols_with_real_news'] > 0:
            analysis['average_articles_per_symbol'] = analysis['total_real_articles'] / analysis['symbols_with_real_news']
        
        # Save analysis report
        analysis_file = self.news_dir / f"news_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Quality analysis completed! Report saved to: {analysis_file}")
        return analysis

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("NEWS DATA CLEANUP - REMOVE ALL DUMMY DATA")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    try:
        cleaner = NewsDataCleaner()
        
        # Clean all dummy data
        print("\nCLEANING: Removing dummy data from all news files...")
        cleanup_report = cleaner.clean_all_news_files()
        
        print(f"\nCLEANUP RESULTS:")
        print(f"   Files processed: {cleanup_report['files_processed']}")
        print(f"   Dummy articles removed: {cleanup_report['dummy_articles_removed']}")
        print(f"   Real articles kept: {cleanup_report['real_articles_kept']}")
        print(f"   Real news sources found: {len(cleanup_report['sources_found'])}")
        
        if cleanup_report['files_with_no_real_data']:
            print(f"\nWARNING: Symbols with NO real news data: {len(cleanup_report['files_with_no_real_data'])}")
            print(f"   {cleanup_report['files_with_no_real_data'][:10]}...")  # Show first 10
        
        # Analyze quality for model training
        print(f"\nANALYZING: News data quality for model training...")
        quality_analysis = cleaner.analyze_news_quality()
        
        print(f"\nQUALITY ANALYSIS:")
        print(f"   Symbols with real news: {quality_analysis['symbols_with_real_news']}")
        print(f"   Total real articles: {quality_analysis['total_real_articles']}")
        print(f"   Average articles per symbol: {quality_analysis['average_articles_per_symbol']:.1f}")
        print(f"   Symbols with good coverage (10+ articles): {len(quality_analysis['symbols_with_good_coverage'])}")
        print(f"   Symbols with poor coverage (<5 articles): {len(quality_analysis['symbols_with_poor_coverage'])}")
        
        print(f"\nTOP NEWS SOURCES:")
        top_sources = sorted(quality_analysis['source_distribution'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        for source, count in top_sources:
            print(f"   {source}: {count} articles")
        
        print(f"\nSENTIMENT DISTRIBUTION:")
        for sentiment, count in quality_analysis['sentiment_distribution'].items():
            percentage = (count / quality_analysis['total_real_articles'] * 100) if quality_analysis['total_real_articles'] > 0 else 0
            print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\nSUCCESS: ALL DUMMY DATA REMOVED!")
        print(f"SUCCESS: Your news files now contain ONLY real financial news data")
        print(f"SUCCESS: Ready for model training and feature selection")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()