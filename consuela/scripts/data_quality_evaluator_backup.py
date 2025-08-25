#!/usr/bin/env python3
"""
Data Quality Evaluation Script for 2Trade System

This script evaluates the quality and completeness of stored market data
using Odin's Eye library and generates comprehensive reports.

Author: Consuela Housekeeping Module
Created: 2025-08-24
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from odins_eye import OdinsEye, MarketDataInterval, DateRange
    from odins_eye.exceptions import OdinsEyeError, DataNotFoundError
    from consuela.config.instrument_list_loader import load_favorites_instruments, load_popular_instruments
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)


class DataQualityEvaluator:
    """
    Comprehensive data quality evaluation system for 2Trade market data.
    
    Analyzes:
    - Data completeness across instruments and time periods
    - Missing data points and gaps
    - Data quality metrics (null values, outliers, etc.)
    - Storage utilization and file integrity
    """
    
    def __init__(self, data_root: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize the data quality evaluator."""
        self.data_root = data_root
        self.output_dir = output_dir or str(project_root / "consuela" / "house-keeping-report")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Odin's Eye
        try:
            self.eye = OdinsEye(data_root=data_root) if data_root else OdinsEye()
            self.actual_data_root = self.eye.data_root
        except Exception as e:
            print(f"Failed to initialize Odin's Eye: {e}")
            sys.exit(1)
            
        # Load instrument lists
        self.favorites = self._load_instruments("favorites")
        self.popular = self._load_instruments("popular")
        self.all_instruments = list(set(self.favorites + self.popular))
        
        # Data intervals to check
        self.intervals = [
            MarketDataInterval.DAILY,
            MarketDataInterval.HOURLY, 
            MarketDataInterval.MIN_30,
            MarketDataInterval.MIN_15,
            MarketDataInterval.MIN_5,
            MarketDataInterval.MIN_1
        ]
        
        # Data types to evaluate
        self.data_types = {
            "market_data": {
                "path": "market_data",
                "description": "OHLCV market data",
                "format": "parquet",
                "intervals": self.intervals
            },
            "economic_data": {
                "path": "economic_data", 
                "description": "FRED economic indicators",
                "format": "parquet",
                "intervals": None
            },
            "news_data": {
                "path": "news_data",
                "description": "Financial news with sentiment", 
                "format": "json",
                "intervals": None
            },
            "trends_data": {
                "path": "trends_data",
                "description": "Google Trends search data",
                "format": "json", 
                "intervals": None
            }
        }
        
        # Common economic indicators to check
        self.economic_indicators = [
            "UNRATE", "FEDFUNDS", "GDP", "CPIAUCSL", "PAYEMS", "INDPRO", 
            "DEXUSEU", "DGS10", "VIXCLS", "DCOILWTICO", "GOLDAMGBD228NLBM"
        ]
        
        # Setup logging
        self._setup_logging()
        
        # Initialize report data
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "data_root": self.actual_data_root,
            "evaluation_summary": {},
            "market_data_analysis": {},
            "economic_data_analysis": {},
            "news_data_analysis": {},
            "trends_data_analysis": {},
            "data_quality_metrics": {},
            "missing_data_analysis": {},
            "recommendations": []
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "data_quality_evaluation.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_instruments(self, list_type: str) -> List[str]:
        """Load instrument symbols from configuration files."""
        try:
            if list_type == "favorites":
                instruments = load_favorites_instruments()
            elif list_type == "popular":
                instruments = load_popular_instruments()
            else:
                self.logger.warning(f"Unknown instrument list type: {list_type}")
                return []
            
            return [instr["symbol"] for instr in instruments]
        except Exception as e:
            self.logger.warning(f"Failed to load {list_type} instruments: {e}")
            return []
    
    def evaluate_data_coverage(self) -> Dict[str, Any]:
        """Evaluate data coverage across instruments and intervals."""
        self.logger.info("Starting data coverage evaluation...")
        
        coverage_report = {
            "total_instruments": len(self.all_instruments),
            "intervals_checked": len(self.intervals),
            "coverage_matrix": {},
            "missing_combinations": [],
            "coverage_percentage": {}
        }
        
        # Check each instrument-interval combination
        for interval in self.intervals:
            interval_name = interval.value
            coverage_report["coverage_matrix"][interval_name] = {}
            coverage_report["coverage_percentage"][interval_name] = 0
            available_count = 0
            
            for symbol in self.all_instruments:
                try:
                    data = self.eye.get_market_data(symbol, interval=interval)
                    if data is not None and not data.empty:
                        coverage_report["coverage_matrix"][interval_name][symbol] = {
                            "available": True,
                            "rows": len(data),
                            "date_range": {
                                "start": data.index.min().isoformat() if not data.empty else None,
                                "end": data.index.max().isoformat() if not data.empty else None
                            },
                            "file_size_mb": self._estimate_file_size(symbol, interval)
                        }
                        available_count += 1
                    else:
                        coverage_report["coverage_matrix"][interval_name][symbol] = {
                            "available": False,
                            "error": "No data or empty dataset"
                        }
                        coverage_report["missing_combinations"].append(f"{symbol}:{interval_name}")
                        
                except (DataNotFoundError, FileNotFoundError) as e:
                    coverage_report["coverage_matrix"][interval_name][symbol] = {
                        "available": False,
                        "error": f"Data not found: {str(e)}"
                    }
                    coverage_report["missing_combinations"].append(f"{symbol}:{interval_name}")
                    
                except Exception as e:
                    coverage_report["coverage_matrix"][interval_name][symbol] = {
                        "available": False,
                        "error": f"Unexpected error: {str(e)}"
                    }
                    coverage_report["missing_combinations"].append(f"{symbol}:{interval_name}")
            
            # Calculate coverage percentage for this interval
            coverage_report["coverage_percentage"][interval_name] = (
                available_count / len(self.all_instruments) * 100
            )
        
        self.report_data["market_data_analysis"] = coverage_report
        return coverage_report
    
    def _estimate_file_size(self, symbol: str, interval: MarketDataInterval) -> Optional[float]:
        """Estimate file size in MB for a given symbol-interval combination."""
        try:
            interval_dir = os.path.join(self.actual_data_root, "market_data", interval.value)
            file_path = os.path.join(interval_dir, f"{symbol}.parquet")
            
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                return round(size_bytes / (1024 * 1024), 2)
            return None
        except Exception:
            return None
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality metrics for available datasets."""
        self.logger.info("Starting data quality analysis...")
        
        quality_report = {
            "instruments_analyzed": 0,
            "quality_metrics": {},
            "issues_found": [],
            "overall_quality_score": 0
        }
        
        quality_scores = []
        
        # Sample a subset of instruments for detailed analysis (to avoid long runtime)
        sample_instruments = self.all_instruments[:20]  # Analyze first 20 instruments
        
        for symbol in sample_instruments:
            try:
                # Focus on daily data for quality analysis
                data = self.eye.get_market_data(symbol, interval=MarketDataInterval.DAILY)
                
                if data is None or data.empty:
                    continue
                    
                quality_report["instruments_analyzed"] += 1
                
                # Calculate quality metrics
                metrics = self._calculate_quality_metrics(data, symbol)
                quality_report["quality_metrics"][symbol] = metrics
                
                # Calculate quality score for this instrument
                instrument_score = self._calculate_quality_score(metrics)
                quality_scores.append(instrument_score)
                
                # Check for specific issues
                issues = self._identify_quality_issues(data, symbol, metrics)
                if issues:
                    quality_report["issues_found"].extend(issues)
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze quality for {symbol}: {e}")
                quality_report["issues_found"].append({
                    "symbol": symbol,
                    "issue": "Analysis failed",
                    "details": str(e)
                })
        
        # Calculate overall quality score
        if quality_scores:
            quality_report["overall_quality_score"] = round(sum(quality_scores) / len(quality_scores), 2)
        
        self.report_data["data_quality_metrics"] = quality_report
        return quality_report
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate detailed quality metrics for a dataset."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics["total_records"] = len(data)
            metrics["date_range_days"] = (data.index.max() - data.index.min()).days if len(data) > 1 else 0
            
            # Null value analysis
            null_counts = data.isnull().sum()
            metrics["null_values"] = {
                "total": int(null_counts.sum()),
                "by_column": {col: int(count) for col, count in null_counts.items()}
            }
            metrics["null_percentage"] = round((null_counts.sum() / (len(data) * len(data.columns))) * 100, 2)
            
            # Price data quality (assuming OHLCV format)
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Check for impossible price relationships
                price_issues = 0
                price_issues += (data['high'] < data['low']).sum()  # High < Low
                price_issues += (data['high'] < data['open']).sum()  # High < Open
                price_issues += (data['high'] < data['close']).sum()  # High < Close
                price_issues += (data['low'] > data['open']).sum()  # Low > Open
                price_issues += (data['low'] > data['close']).sum()  # Low > Close
                
                metrics["price_relationship_errors"] = int(price_issues)
                
                # Check for zero or negative prices
                price_columns = ['open', 'high', 'low', 'close']
                zero_negative_prices = (data[price_columns] <= 0).sum().sum()
                metrics["invalid_prices"] = int(zero_negative_prices)
                
                # Calculate price volatility
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > 1:
                        metrics["return_volatility"] = float(returns.std())
                        metrics["max_single_day_return"] = float(returns.abs().max())
            
            # Volume analysis
            if 'volume' in data.columns:
                metrics["zero_volume_days"] = int((data['volume'] == 0).sum())
                metrics["negative_volume_days"] = int((data['volume'] < 0).sum())
                metrics["avg_volume"] = float(data['volume'].mean())
            
            # Data completeness over time
            if len(data) > 1:
                expected_trading_days = pd.bdate_range(
                    start=data.index.min(), 
                    end=data.index.max()
                )
                actual_days = len(data)
                expected_days = len(expected_trading_days)
                metrics["completeness_ratio"] = round(actual_days / expected_days, 3) if expected_days > 0 else 0
                metrics["missing_trading_days"] = max(0, expected_days - actual_days)
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics for {symbol}: {e}")
            metrics["calculation_error"] = str(e)
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a quality score (0-100) based on metrics."""
        score = 100.0
        
        # Penalize high null percentage
        if "null_percentage" in metrics:
            score -= min(metrics["null_percentage"] * 2, 30)  # Max 30 point penalty
        
        # Penalize price relationship errors
        if "price_relationship_errors" in metrics and metrics["total_records"] > 0:
            error_rate = metrics["price_relationship_errors"] / metrics["total_records"]
            score -= min(error_rate * 100, 20)  # Max 20 point penalty
        
        # Penalize invalid prices
        if "invalid_prices" in metrics and metrics["total_records"] > 0:
            invalid_rate = metrics["invalid_prices"] / metrics["total_records"]
            score -= min(invalid_rate * 100, 25)  # Max 25 point penalty
        
        # Reward good completeness ratio
        if "completeness_ratio" in metrics:
            if metrics["completeness_ratio"] < 0.8:
                score -= (0.8 - metrics["completeness_ratio"]) * 25  # Penalty for low completeness
        
        return max(0.0, min(100.0, score))
    
    def _identify_quality_issues(self, data: pd.DataFrame, symbol: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific data quality issues."""
        issues = []
        
        # High null percentage
        if metrics.get("null_percentage", 0) > 5:
            issues.append({
                "symbol": symbol,
                "issue": "High null value percentage",
                "severity": "high" if metrics["null_percentage"] > 15 else "medium",
                "details": f"{metrics['null_percentage']}% null values"
            })
        
        # Price relationship errors
        if metrics.get("price_relationship_errors", 0) > 0:
            issues.append({
                "symbol": symbol,
                "issue": "Price relationship violations",
                "severity": "high",
                "details": f"{metrics['price_relationship_errors']} records with impossible OHLC relationships"
            })
        
        # Invalid prices
        if metrics.get("invalid_prices", 0) > 0:
            issues.append({
                "symbol": symbol,
                "issue": "Invalid price values",
                "severity": "high",
                "details": f"{metrics['invalid_prices']} records with zero or negative prices"
            })
        
        # Low completeness
        if metrics.get("completeness_ratio", 1) < 0.7:
            issues.append({
                "symbol": symbol,
                "issue": "Low data completeness",
                "severity": "medium" if metrics["completeness_ratio"] > 0.5 else "high",
                "details": f"Only {metrics['completeness_ratio']*100:.1f}% of expected trading days present"
            })
        
        return issues
    
    def evaluate_economic_data_coverage(self) -> Dict[str, Any]:
        """Evaluate coverage and quality of economic data (FRED indicators)."""
        self.logger.info("Starting economic data evaluation...")
        
        economic_report = {
            "total_indicators_checked": len(self.economic_indicators),
            "available_indicators": {},
            "missing_indicators": [],
            "data_quality_issues": [],
            "coverage_percentage": 0
        }
        
        economic_dir = os.path.join(self.actual_data_root, "economic_data")
        available_count = 0
        
        for indicator in self.economic_indicators:
            try:
                # Try to get economic data using Odin's Eye
                data = self.eye.get_economic_data(indicator)
                
                if data is not None and not data.empty:
                    # Calculate metrics for this indicator
                    metrics = {
                        "available": True,
                        "rows": len(data),
                        "date_range": {
                            "start": data.index.min().isoformat() if not data.empty else None,
                            "end": data.index.max().isoformat() if not data.empty else None
                        },
                        "null_count": int(data['value'].isnull().sum()) if 'value' in data.columns else 0,
                        "file_size_mb": self._get_file_size(os.path.join(economic_dir, f"{indicator}.parquet"))
                    }
                    
                    # Check for data quality issues
                    if metrics["null_count"] > len(data) * 0.1:  # More than 10% null values
                        economic_report["data_quality_issues"].append({
                            "indicator": indicator,
                            "issue": "High null value percentage",
                            "details": f"{(metrics['null_count']/len(data)*100):.1f}% null values"
                        })
                    
                    economic_report["available_indicators"][indicator] = metrics
                    available_count += 1
                else:
                    economic_report["missing_indicators"].append(indicator)
                    
            except Exception as e:
                economic_report["missing_indicators"].append(indicator)
                economic_report["data_quality_issues"].append({
                    "indicator": indicator,
                    "issue": "Data access failed",
                    "details": str(e)
                })
        
        economic_report["coverage_percentage"] = round((available_count / len(self.economic_indicators)) * 100, 2)
        
        self.report_data["economic_data_analysis"] = economic_report
        return economic_report\n    \n    def evaluate_news_data_coverage(self) -> Dict[str, Any]:\n        \"\"\"Evaluate coverage and quality of news data.\"\"\"\n        self.logger.info(\"Starting news data evaluation...\")\n        \n        news_report = {\n            \"news_files_found\": 0,\n            \"date_range\": {},\n            \"total_articles\": 0,\n            \"articles_by_date\": {},\n            \"sentiment_distribution\": {\"positive\": 0, \"negative\": 0, \"neutral\": 0},\n            \"data_quality_issues\": [],\n            \"storage_size_mb\": 0\n        }\n        \n        news_dir = os.path.join(self.actual_data_root, \"news_data\")\n        \n        if os.path.exists(news_dir):\n            news_files = [f for f in os.listdir(news_dir) if f.endswith('.json') and f.startswith('news_')]\n            news_report[\"news_files_found\"] = len(news_files)\n            \n            # Calculate total storage size\n            total_size = 0\n            dates_found = []\n            total_articles = 0\n            sentiment_counts = {\"positive\": 0, \"negative\": 0, \"neutral\": 0, \"unknown\": 0}\n            \n            for file in news_files[:20]:  # Sample first 20 files to avoid long processing\n                file_path = os.path.join(news_dir, file)\n                file_size = self._get_file_size(file_path)\n                total_size += file_size\n                \n                # Extract date from filename (news_YYYYMMDD.json)\n                try:\n                    date_str = file.replace('news_', '').replace('.json', '')\n                    dates_found.append(date_str)\n                    \n                    # Try to load and analyze the JSON file\n                    with open(file_path, 'r', encoding='utf-8') as f:\n                        news_data = json.load(f)\n                        \n                        if isinstance(news_data, list):\n                            article_count = len(news_data)\n                            total_articles += article_count\n                            news_report[\"articles_by_date\"][date_str] = article_count\n                            \n                            # Analyze sentiment distribution (sample first 10 articles per file)\n                            for article in news_data[:10]:\n                                sentiment = article.get('sentiment_label', 'unknown').lower()\n                                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1\n                                \n                        else:\n                            news_report[\"data_quality_issues\"].append({\n                                \"file\": file,\n                                \"issue\": \"Invalid JSON structure\",\n                                \"details\": \"Expected array of articles\"\n                            })\n                            \n                except Exception as e:\n                    news_report[\"data_quality_issues\"].append({\n                        \"file\": file,\n                        \"issue\": \"File processing error\",\n                        \"details\": str(e)\n                    })\n            \n            # Set summary statistics\n            news_report[\"total_articles\"] = total_articles\n            news_report[\"storage_size_mb\"] = round(total_size, 2)\n            news_report[\"sentiment_distribution\"] = sentiment_counts\n            \n            if dates_found:\n                dates_found.sort()\n                news_report[\"date_range\"] = {\n                    \"start\": dates_found[0],\n                    \"end\": dates_found[-1]\n                }\n        \n        self.report_data[\"news_data_analysis\"] = news_report\n        return news_report\n    \n    def evaluate_trends_data_coverage(self) -> Dict[str, Any]:\n        \"\"\"Evaluate coverage and quality of Google Trends data.\"\"\"\n        self.logger.info(\"Starting trends data evaluation...\")\n        \n        trends_report = {\n            \"trends_files_found\": 0,\n            \"keywords_tracked\": [],\n            \"date_range\": {},\n            \"data_quality_issues\": [],\n            \"storage_size_mb\": 0,\n            \"symbols_with_trends\": []\n        }\n        \n        trends_dir = os.path.join(self.actual_data_root, \"trends_data\")\n        \n        if os.path.exists(trends_dir):\n            trends_files = [f for f in os.listdir(trends_dir) if f.endswith('.json')]\n            trends_report[\"trends_files_found\"] = len(trends_files)\n            \n            # Calculate total storage size\n            total_size = 0\n            keywords_found = set()\n            symbols_found = set()\n            dates_found = []\n            \n            for file in trends_files[:50]:  # Sample first 50 files\n                file_path = os.path.join(trends_dir, file)\n                file_size = self._get_file_size(file_path)\n                total_size += file_size\n                \n                # Extract keyword and date from filename (keyword_YYYYMMDD.json)\n                try:\n                    parts = file.replace('.json', '').split('_')\n                    if len(parts) >= 2:\n                        keyword = '_'.join(parts[:-1])  # Everything except last part (date)\n                        date_str = parts[-1]\n                        \n                        keywords_found.add(keyword)\n                        dates_found.append(date_str)\n                        \n                        # Check if keyword is a stock symbol\n                        if keyword.upper() in self.all_instruments:\n                            symbols_found.add(keyword.upper())\n                        \n                        # Try to load and validate the JSON file\n                        with open(file_path, 'r', encoding='utf-8') as f:\n                            trends_data = json.load(f)\n                            \n                            # Validate expected structure\n                            if not isinstance(trends_data, dict):\n                                trends_report[\"data_quality_issues\"].append({\n                                    \"file\": file,\n                                    \"issue\": \"Invalid JSON structure\",\n                                    \"details\": \"Expected object with trends data\"\n                                })\n                            elif 'search_interest' not in trends_data:\n                                trends_report[\"data_quality_issues\"].append({\n                                    \"file\": file,\n                                    \"issue\": \"Missing search_interest data\",\n                                    \"details\": \"No search interest time series found\"\n                                })\n                                \n                except Exception as e:\n                    trends_report[\"data_quality_issues\"].append({\n                        \"file\": file,\n                        \"issue\": \"File processing error\",\n                        \"details\": str(e)\n                    })\n            \n            # Set summary statistics\n            trends_report[\"keywords_tracked\"] = sorted(list(keywords_found))\n            trends_report[\"symbols_with_trends\"] = sorted(list(symbols_found))\n            trends_report[\"storage_size_mb\"] = round(total_size, 2)\n            \n            if dates_found:\n                dates_found.sort()\n                trends_report[\"date_range\"] = {\n                    \"start\": dates_found[0],\n                    \"end\": dates_found[-1]\n                }\n        \n        self.report_data[\"trends_data_analysis\"] = trends_report\n        return trends_report\n    \n    def _get_file_size(self, file_path: str) -> float:\n        \"\"\"Get file size in MB, return 0 if file doesn't exist.\"\"\"\n        try:\n            if os.path.exists(file_path):\n                return round(os.path.getsize(file_path) / (1024 * 1024), 2)\n            return 0.0\n        except Exception:\n            return 0.0\n    \n    def analyze_missing_data_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in missing data to identify systematic issues."""
        self.logger.info("Analyzing missing data patterns...")
        
        missing_analysis = {
            "summary": {},
            "patterns": {},
            "recommendations": []
        }
        
        # Analyze missing data by instrument type
        instrument_types = {
            "technology": [s for s in self.favorites if s in ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC"]],
            "finance": [s for s in self.favorites if s in ["V", "PYPL", "BRK-B"]],
            "etf": [s for s in self.favorites if s in ["VOO", "GLD", "VWO", "IEFA"]]
        }
        
        for category, symbols in instrument_types.items():
            category_missing = []
            for symbol in symbols:
                for interval in self.intervals:
                    if f"{symbol}:{interval.value}" in self.report_data.get("data_coverage", {}).get("missing_combinations", []):
                        category_missing.append(f"{symbol}:{interval.value}")
            
            missing_analysis["patterns"][category] = {
                "total_combinations": len(symbols) * len(self.intervals),
                "missing_combinations": len(category_missing),
                "missing_percentage": round(len(category_missing) / (len(symbols) * len(self.intervals)) * 100, 2),
                "missing_list": category_missing
            }
        
        # Generate recommendations based on patterns
        missing_analysis["recommendations"] = self._generate_missing_data_recommendations(missing_analysis["patterns"])
        
        self.report_data["missing_data_analysis"] = missing_analysis
        return missing_analysis
    
    def _generate_missing_data_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on missing data patterns."""
        recommendations = []
        
        # Check for high missing percentages
        for category, data in patterns.items():
            if data["missing_percentage"] > 50:
                recommendations.append(
                    f"HIGH PRIORITY: {category.title()} instruments have {data['missing_percentage']:.1f}% missing data combinations. "
                    "Consider prioritizing data collection for this category."
                )
            elif data["missing_percentage"] > 25:
                recommendations.append(
                    f"MEDIUM PRIORITY: {category.title()} instruments have {data['missing_percentage']:.1f}% missing data combinations. "
                    "Review data collection processes for this category."
                )
        
        # General recommendations
        total_missing = sum(data["missing_combinations"] for data in patterns.values())
        total_combinations = sum(data["total_combinations"] for data in patterns.values())
        
        if total_missing > 0:
            recommendations.append(
                f"Overall data completeness: {((total_combinations - total_missing) / total_combinations * 100):.1f}%. "
                f"Focus on collecting {total_missing} missing data combinations."
            )
        
        return recommendations
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        self.logger.info("Generating summary statistics...")
        
        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "data_root_location": self.actual_data_root,
            "instruments_evaluated": len(self.all_instruments),
            "data_intervals_checked": len(self.intervals),
            "total_combinations_possible": len(self.all_instruments) * len(self.intervals),
            "storage_analysis": self._analyze_storage_usage()
        }
        
        # Add coverage statistics
        if "market_data_analysis" in self.report_data:
            coverage = self.report_data["market_data_analysis"]
            summary["missing_data_combinations"] = len(coverage.get("missing_combinations", []))
            summary["data_availability_percentage"] = round(
                (summary["total_combinations_possible"] - summary["missing_data_combinations"]) / 
                summary["total_combinations_possible"] * 100, 2
            )
        
        # Add statistics for other data types
        data_type_stats = {}
        if "economic_data_analysis" in self.report_data:
            econ = self.report_data["economic_data_analysis"]
            data_type_stats["economic_data"] = {
                "indicators_available": len(econ.get("available_indicators", {})),
                "indicators_missing": len(econ.get("missing_indicators", [])),
                "coverage_percentage": econ.get("coverage_percentage", 0)
            }
        
        if "news_data_analysis" in self.report_data:
            news = self.report_data["news_data_analysis"] 
            data_type_stats["news_data"] = {
                "files_found": news.get("news_files_found", 0),
                "total_articles": news.get("total_articles", 0),
                "storage_mb": news.get("storage_size_mb", 0)
            }
        
        if "trends_data_analysis" in self.report_data:
            trends = self.report_data["trends_data_analysis"]
            data_type_stats["trends_data"] = {
                "files_found": trends.get("trends_files_found", 0), 
                "keywords_tracked": len(trends.get("keywords_tracked", [])),
                "symbols_with_trends": len(trends.get("symbols_with_trends", [])),
                "storage_mb": trends.get("storage_size_mb", 0)
            }
        
        summary["data_type_statistics"] = data_type_stats
        
        # Add quality statistics
        if "data_quality_metrics" in self.report_data:
            quality = self.report_data["data_quality_metrics"]
            summary["quality_issues_found"] = len(quality.get("issues_found", []))
            summary["overall_quality_score"] = quality.get("overall_quality_score", "N/A")
        
        self.report_data["evaluation_summary"] = summary
        return summary
    
    def _analyze_storage_usage(self) -> Dict[str, Any]:
        """Analyze storage usage patterns."""
        storage_info = {
            "total_size_mb": 0,
            "file_count": 0,
            "by_interval": {}
        }
        
        try:
            market_data_dir = os.path.join(self.actual_data_root, "market_data")
            
            if os.path.exists(market_data_dir):
                for interval_dir in os.listdir(market_data_dir):
                    interval_path = os.path.join(market_data_dir, interval_dir)
                    if os.path.isdir(interval_path):
                        interval_size = 0
                        interval_files = 0
                        
                        for file in os.listdir(interval_path):
                            if file.endswith('.parquet'):
                                file_path = os.path.join(interval_path, file)
                                file_size = os.path.getsize(file_path)
                                interval_size += file_size
                                interval_files += 1
                        
                        storage_info["by_interval"][interval_dir] = {
                            "size_mb": round(interval_size / (1024 * 1024), 2),
                            "file_count": interval_files
                        }
                        storage_info["total_size_mb"] += interval_size / (1024 * 1024)
                        storage_info["file_count"] += interval_files
            
            storage_info["total_size_mb"] = round(storage_info["total_size_mb"], 2)
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze storage usage: {e}")
            storage_info["error"] = str(e)
        
        return storage_info
    
    def save_report(self, report_type: str = "comprehensive") -> str:
        """Save the evaluation report to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_type == "comprehensive":
            filename = f"data_quality_report_{timestamp}.json"
        else:
            filename = f"data_quality_summary_{timestamp}.json"
        
        report_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.report_data, f, indent=2, default=str)
            
            self.logger.info(f"Report saved to: {report_path}")
            
            # Also create a human-readable summary
            self._save_readable_summary(timestamp)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            raise
    
    def _save_readable_summary(self, timestamp: str):
        """Save a human-readable summary report."""
        summary_path = os.path.join(self.output_dir, f"data_quality_summary_{timestamp}.txt")
        
        try:
            with open(summary_path, 'w') as f:
                f.write("2TRADE DATA QUALITY EVALUATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Summary statistics
                if "evaluation_summary" in self.report_data:
                    summary = self.report_data["evaluation_summary"]
                    f.write("SUMMARY STATISTICS\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Evaluation Date: {summary.get('evaluation_date', 'N/A')}\n")
                    f.write(f"Data Root: {summary.get('data_root_location', 'N/A')}\n")
                    f.write(f"Instruments Evaluated: {summary.get('instruments_evaluated', 'N/A')}\n")
                    f.write(f"Data Availability: {summary.get('data_availability_percentage', 'N/A')}%\n")
                    f.write(f"Quality Score: {summary.get('overall_quality_score', 'N/A')}/100\n")
                    f.write(f"Quality Issues Found: {summary.get('quality_issues_found', 'N/A')}\n\n")
                
                # Data type statistics
                if "evaluation_summary" in self.report_data and "data_type_statistics" in self.report_data["evaluation_summary"]:
                    data_stats = self.report_data["evaluation_summary"]["data_type_statistics"]
                    f.write("DATA TYPE ANALYSIS\n")
                    f.write("-" * 18 + "\n")
                    
                    if "economic_data" in data_stats:
                        econ = data_stats["economic_data"]
                        f.write(f"Economic Data: {econ['indicators_available']}/{econ['indicators_available'] + econ['indicators_missing']} indicators available ({econ['coverage_percentage']}%)\n")
                    
                    if "news_data" in data_stats:
                        news = data_stats["news_data"] 
                        f.write(f"News Data: {news['files_found']} files, {news['total_articles']} articles ({news['storage_mb']} MB)\n")
                    
                    if "trends_data" in data_stats:
                        trends = data_stats["trends_data"]
                        f.write(f"Trends Data: {trends['files_found']} files, {trends['keywords_tracked']} keywords, {trends['symbols_with_trends']} symbols ({trends['storage_mb']} MB)\n")
                    f.write("\n")

                # Storage information
                if "evaluation_summary" in self.report_data and "storage_analysis" in self.report_data["evaluation_summary"]:
                    storage = self.report_data["evaluation_summary"]["storage_analysis"]
                    f.write("MARKET DATA STORAGE\n")
                    f.write("-" * 19 + "\n")
                    f.write(f"Total Storage Used: {storage.get('total_size_mb', 'N/A')} MB\n")
                    f.write(f"Total Files: {storage.get('file_count', 'N/A')}\n\n")
                    
                    if "by_interval" in storage:
                        f.write("Storage by Interval:\n")
                        for interval, data in storage["by_interval"].items():
                            f.write(f"  {interval}: {data['size_mb']} MB ({data['file_count']} files)\n")
                        f.write("\n")
                
                # Quality issues
                if "data_quality_metrics" in self.report_data:
                    issues = self.report_data["data_quality_metrics"].get("issues_found", [])
                    if issues:
                        f.write("DATA QUALITY ISSUES\n")
                        f.write("-" * 19 + "\n")
                        for issue in issues[:10]:  # Show first 10 issues
                            f.write(f"- {issue.get('symbol', 'Unknown')}: {issue.get('issue', 'Unknown issue')} "
                                   f"({issue.get('severity', 'unknown')} severity)\n")
                        
                        if len(issues) > 10:
                            f.write(f"... and {len(issues) - 10} more issues\n")
                        f.write("\n")
                
                # Recommendations
                if "missing_data_analysis" in self.report_data:
                    recommendations = self.report_data["missing_data_analysis"].get("recommendations", [])
                    if recommendations:
                        f.write("RECOMMENDATIONS\n")
                        f.write("-" * 15 + "\n")
                        for i, rec in enumerate(recommendations, 1):
                            f.write(f"{i}. {rec}\n")
                        f.write("\n")
                
                f.write("For detailed analysis, see the corresponding JSON report file.\n")
            
            self.logger.info(f"Human-readable summary saved to: {summary_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save readable summary: {e}")
    
    def run_full_evaluation(self) -> str:
        """Run the complete data quality evaluation process."""
        self.logger.info("Starting comprehensive data quality evaluation...")
        
        try:
            # 1. Evaluate market data coverage
            self.evaluate_data_coverage()
            
            # 2. Evaluate economic data coverage  
            self.evaluate_economic_data_coverage()
            
            # 3. Evaluate news data coverage
            self.evaluate_news_data_coverage()
            
            # 4. Evaluate trends data coverage
            self.evaluate_trends_data_coverage()
            
            # 5. Analyze market data quality (detailed)
            self.analyze_data_quality()
            
            # 6. Analyze missing data patterns
            self.analyze_missing_data_patterns()
            
            # 7. Generate summary statistics
            self.generate_summary_statistics()
            
            # 8. Save comprehensive report
            report_path = self.save_report("comprehensive")
            
            self.logger.info("Data quality evaluation completed successfully!")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main entry point for the data quality evaluator script."""
    parser = argparse.ArgumentParser(
        description="Evaluate data quality and completeness for 2Trade market data"
    )
    parser.add_argument(
        "--data-root", 
        type=str, 
        help="Custom data root directory (default: uses Odin's Eye default)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Custom output directory for reports (default: consuela/house-keeping-report)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick evaluation with limited analysis"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize evaluator
        evaluator = DataQualityEvaluator(
            data_root=args.data_root,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        if args.quick:
            # Quick evaluation - coverage for all data types and basic stats
            evaluator.evaluate_data_coverage()
            evaluator.evaluate_economic_data_coverage() 
            evaluator.evaluate_news_data_coverage()
            evaluator.evaluate_trends_data_coverage()
            evaluator.generate_summary_statistics()
            report_path = evaluator.save_report("summary")
        else:
            # Full comprehensive evaluation
            report_path = evaluator.run_full_evaluation()
        
        print(f"\nData quality evaluation completed!")
        print(f"Report saved to: {report_path}")
        print(f"Check the output directory for additional files: {evaluator.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()