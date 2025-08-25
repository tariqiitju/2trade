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
    
    def __init__(self, data_root: Optional[str] = None, output_dir: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the data quality evaluator."""
        self.data_root = data_root
        self.output_dir = output_dir or str(project_root / "consuela" / "house-keeping-report")
        
        # Load configuration
        self.config_path = config_path or str(project_root / "consuela" / "config" / "data-quality-config.yml")
        self.config = self._load_config()
        
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
            "earnings_data": {
                "path": "earnings_data",
                "description": "Company earnings data and forecasts",
                "format": "json",
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
        
        # Common economic indicators to check (expanded list)
        self.economic_indicators = [
            # Core Economic Indicators
            "UNRATE", "FEDFUNDS", "GDP", "GDPC1", "CPIAUCSL", "CPILFESL", "PAYEMS", "INDPRO",
            # Market Indicators
            "DEXUSEU", "DGS10", "DGS2", "VIXCLS", "DCOILWTICO", "GOLDAMGBD228NLBM", "SP500",
            # Employment & Labor
            "CIVPART", "EMRATIO", "NFCI", "AHETPI", "HOUST", "PERMIT",
            # Consumer & Business
            "UMCSENT", "CSUSHPINSA", "RRSFS", "BOGMBASE", "M2SL", "WALCL",
            # International
            "DTWEXBGS", "DTWEXAFEGS", "CPIENGSL", "WPU10170301"
        ]
        
        # Setup logging
        self._setup_logging()
        
        # Initialize report data
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "data_root": self.actual_data_root,
            "config_used": self.config_path,
            "evaluation_summary": {},
            "market_data_analysis": {},
            "economic_data_analysis": {},
            "earnings_data_analysis": {},
            "news_data_analysis": {},
            "trends_data_analysis": {},
            "data_quality_metrics": {},
            "missing_data_analysis": {},
            "recommendations": [],
            "quality_scores_by_type": {}
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load validation configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                print(f"Warning: Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available."""
        return {
            "validation_rules": {
                "market_data": {
                    "price_validation": {"max_daily_change": 0.50},
                    "completeness": {"min_completeness_ratio": 0.80}
                },
                "economic_data": {
                    "null_thresholds": {"max_null_percentage": 15}
                },
                "news_data": {
                    "article_quality": {"min_headline_length": 10}
                },
                "trends_data": {
                    "search_interest": {"valid_score_range": {"min": 0, "max": 100}}
                }
            },
            "quality_scoring": {
                "penalty_weights": {"critical": 25, "high": 15, "medium": 8, "low": 3}
            },
            "performance": {
                "sampling": {"max_instruments_sample": 50}
            }
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
        
        # Sample a subset of instruments for detailed analysis (configurable)
        max_sample = self.config.get('performance', {}).get('sampling', {}).get('max_instruments_sample', 20)
        sample_instruments = self.all_instruments[:max_sample]
        
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
            # Calculate date range
            try:
                if len(data) > 1:
                    date_diff = data.index.max() - data.index.min()
                    if hasattr(date_diff, 'days'):
                        metrics["date_range_days"] = date_diff.days
                    else:
                        # Handle other types of date differences
                        metrics["date_range_days"] = int(date_diff / pd.Timedelta(days=1))
                else:
                    metrics["date_range_days"] = 0
            except Exception:
                metrics["date_range_days"] = 0
            
            # Data freshness
            if len(data) > 0:
                latest_date = data.index.max()
                try:
                    if hasattr(latest_date, 'date'):
                        days_since_update = (datetime.now().date() - latest_date.date()).days
                    elif isinstance(latest_date, pd.Timestamp):
                        days_since_update = (datetime.now() - latest_date).days
                    else:
                        # Handle other date formats
                        latest_date_dt = pd.to_datetime(latest_date)
                        days_since_update = (datetime.now() - latest_date_dt).days
                    
                    metrics["days_since_last_update"] = days_since_update
                    metrics["data_freshness_status"] = self._get_freshness_status(days_since_update)
                except Exception as e:
                    # If date calculation fails, skip freshness metrics
                    pass
            
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
                
                # Check for duplicate prices (potential stale data)
                duplicate_ohlc = (data['open'] == data['high']) & (data['high'] == data['low']) & (data['low'] == data['close'])
                metrics["flat_price_days"] = int(duplicate_ohlc.sum())
                
                # Price gap analysis
                if 'close' in data.columns and len(data) > 1:
                    price_changes = data['close'].pct_change().abs()
                    large_gaps = (price_changes > 0.2).sum()  # More than 20% price change
                    metrics["large_price_gaps"] = int(large_gaps)
                
                # Calculate price volatility and statistics
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > 1:
                        metrics["return_volatility"] = float(returns.std())
                        metrics["max_single_day_return"] = float(returns.abs().max())
                        metrics["avg_daily_return"] = float(returns.mean())
                        
                        # Check for extreme returns (potential data errors)
                        extreme_returns = (returns.abs() > 0.5).sum()  # >50% daily return
                        metrics["extreme_return_days"] = int(extreme_returns)
            
            # Volume analysis
            if 'volume' in data.columns:
                metrics["zero_volume_days"] = int((data['volume'] == 0).sum())
                metrics["negative_volume_days"] = int((data['volume'] < 0).sum())
                metrics["avg_volume"] = float(data['volume'].mean())
                
                # Volume consistency checks
                if len(data) > 10:
                    volume_median = data['volume'].median()
                    very_low_volume = (data['volume'] < volume_median * 0.01).sum()  # <1% of median
                    very_high_volume = (data['volume'] > volume_median * 100).sum()   # >100x median
                    metrics["abnormal_volume_days"] = int(very_low_volume + very_high_volume)
            
            # Technical indicators presence
            technical_indicators = [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 
                'bollinger_upper', 'bollinger_lower', 'atr', 'vwap'
            ]
            present_indicators = [col for col in technical_indicators if col in data.columns]
            metrics["technical_indicators_present"] = len(present_indicators)
            metrics["technical_indicators_list"] = present_indicators
            
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
                
                # Gap analysis - find longest missing period
                if len(data) > 2:
                    try:
                        # Convert to datetime if needed and calculate differences
                        index_series = pd.Series(data.index)
                        if not pd.api.types.is_datetime64_any_dtype(index_series):
                            index_series = pd.to_datetime(index_series)
                        
                        date_diffs = index_series.diff()
                        # Convert to days
                        if hasattr(date_diffs, 'dt'):
                            date_diffs_days = date_diffs.dt.days.dropna()
                        else:
                            # Handle timedelta directly
                            date_diffs_days = (date_diffs / pd.Timedelta(days=1)).dropna()
                        
                        weekend_adjusted_gaps = date_diffs_days[date_diffs_days > 3]  # More than weekend gap
                        metrics["longest_data_gap_days"] = int(weekend_adjusted_gaps.max()) if len(weekend_adjusted_gaps) > 0 else 0
                        metrics["data_gaps_count"] = int(len(weekend_adjusted_gaps))
                    except Exception as gap_error:
                        # If gap analysis fails, set defaults
                        metrics["longest_data_gap_days"] = 0
                        metrics["data_gaps_count"] = 0
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics for {symbol}: {e}")
            metrics["calculation_error"] = str(e)
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a quality score (0-100) based on metrics and configuration."""
        score = 100.0
        
        # Get penalty weights from config
        penalties = self.config.get('quality_scoring', {}).get('penalty_weights', {
            'critical': 25, 'high': 15, 'medium': 8, 'low': 3
        })
        
        # Get market data validation rules
        market_rules = self.config.get('validation_rules', {}).get('market_data', {})
        quality_weights = market_rules.get('quality_weights', {
            'null_percentage': 0.25, 'price_errors': 0.30, 'completeness': 0.25, 'freshness': 0.20
        })
        
        # Penalize high null percentage
        if "null_percentage" in metrics:
            null_penalty = metrics["null_percentage"] * quality_weights.get('null_percentage', 0.25) * 2
            score -= min(null_penalty, penalties.get('high', 15))
        
        # Penalize price relationship errors
        if "price_relationship_errors" in metrics and metrics.get("total_records", 0) > 0:
            error_rate = metrics["price_relationship_errors"] / metrics["total_records"]
            price_penalty = error_rate * 100 * quality_weights.get('price_errors', 0.30)
            score -= min(price_penalty, penalties.get('critical', 25))
        
        # Penalize invalid prices
        if "invalid_prices" in metrics and metrics.get("total_records", 0) > 0:
            invalid_rate = metrics["invalid_prices"] / metrics["total_records"]
            invalid_penalty = invalid_rate * 100 * quality_weights.get('price_errors', 0.30)
            score -= min(invalid_penalty, penalties.get('critical', 25))
        
        # Penalize low completeness ratio
        completeness_threshold = market_rules.get('completeness', {}).get('min_completeness_ratio', 0.8)
        if "completeness_ratio" in metrics:
            if metrics["completeness_ratio"] < completeness_threshold:
                completeness_penalty = (completeness_threshold - metrics["completeness_ratio"]) * \
                                     100 * quality_weights.get('completeness', 0.25)
                score -= min(completeness_penalty, penalties.get('medium', 8))
        
        # Penalize stale data
        freshness_threshold = market_rules.get('completeness', {}).get('freshness_threshold_days', 7)
        if "days_since_last_update" in metrics:
            days_old = metrics["days_since_last_update"]
            if days_old > freshness_threshold:
                freshness_penalty = min((days_old - freshness_threshold) / 30, 1.0) * \
                                  quality_weights.get('freshness', 0.20) * 100
                score -= min(freshness_penalty, penalties.get('medium', 8))
        
        return max(0.0, min(100.0, score))
    
    def _identify_quality_issues(self, data: pd.DataFrame, symbol: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific data quality issues based on configuration."""
        issues = []
        
        # Get validation rules from config
        market_rules = self.config.get('validation_rules', {}).get('market_data', {})
        price_rules = market_rules.get('price_validation', {})
        volume_rules = market_rules.get('volume_validation', {})
        completeness_rules = market_rules.get('completeness', {})
        
        # High null percentage
        null_threshold = 5  # Default threshold
        if metrics.get("null_percentage", 0) > null_threshold:
            issues.append({
                "symbol": symbol,
                "issue": "High null value percentage",
                "severity": "high" if metrics["null_percentage"] > 15 else "medium",
                "details": f"{metrics['null_percentage']}% null values",
                "threshold": null_threshold
            })
        
        # Price relationship errors
        if metrics.get("price_relationship_errors", 0) > 0:
            issues.append({
                "symbol": symbol,
                "issue": "Price relationship violations",
                "severity": "critical",
                "details": f"{metrics['price_relationship_errors']} records with impossible OHLC relationships",
                "recommendation": "Verify data source and check for data corruption"
            })
        
        # Invalid prices
        if metrics.get("invalid_prices", 0) > 0:
            issues.append({
                "symbol": symbol,
                "issue": "Invalid price values",
                "severity": "critical",
                "details": f"{metrics['invalid_prices']} records with zero or negative prices",
                "recommendation": "Remove or correct invalid price data"
            })
        
        # Extreme returns
        if metrics.get("extreme_return_days", 0) > 0:
            issues.append({
                "symbol": symbol,
                "issue": "Extreme daily returns detected",
                "severity": "medium",
                "details": f"{metrics['extreme_return_days']} days with >50% price changes",
                "recommendation": "Verify these are legitimate price movements, not data errors"
            })
        
        # Large price gaps
        if metrics.get("large_price_gaps", 0) > 5:  # More than 5 large gaps
            issues.append({
                "symbol": symbol,
                "issue": "Frequent large price gaps",
                "severity": "medium",
                "details": f"{metrics['large_price_gaps']} instances of >20% price gaps",
                "recommendation": "Check for missing data or corporate actions"
            })
        
        # Low completeness
        min_completeness = completeness_rules.get('min_completeness_ratio', 0.8)
        if metrics.get("completeness_ratio", 1) < min_completeness:
            severity = "high" if metrics["completeness_ratio"] < 0.5 else "medium"
            issues.append({
                "symbol": symbol,
                "issue": "Low data completeness",
                "severity": severity,
                "details": f"Only {metrics['completeness_ratio']*100:.1f}% of expected trading days present",
                "threshold": f"Minimum {min_completeness*100:.0f}%",
                "recommendation": "Fill missing data gaps or update data collection process"
            })
        
        # Data freshness issues
        freshness_threshold = completeness_rules.get('freshness_threshold_days', 7)
        if metrics.get("days_since_last_update", 0) > freshness_threshold:
            days_old = metrics["days_since_last_update"]
            severity = "high" if days_old > 30 else "medium"
            issues.append({
                "symbol": symbol,
                "issue": "Stale data",
                "severity": severity,
                "details": f"Last updated {days_old} days ago",
                "threshold": f"Should be updated within {freshness_threshold} days",
                "recommendation": "Update data collection schedule or check data source"
            })
        
        # Volume anomalies
        if metrics.get("abnormal_volume_days", 0) > metrics.get("total_records", 0) * 0.05:  # >5% abnormal
            issues.append({
                "symbol": symbol,
                "issue": "Volume anomalies detected",
                "severity": "low",
                "details": f"{metrics['abnormal_volume_days']} days with abnormal volume",
                "recommendation": "Review volume data for accuracy"
            })
        
        # Long data gaps
        max_gap = completeness_rules.get('max_gap_days', 10)
        if metrics.get("longest_data_gap_days", 0) > max_gap:
            issues.append({
                "symbol": symbol,
                "issue": "Long data gap detected",
                "severity": "medium",
                "details": f"Longest gap: {metrics['longest_data_gap_days']} days",
                "threshold": f"Maximum gap: {max_gap} days",
                "recommendation": "Fill data gaps or investigate cause"
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
            "coverage_percentage": 0,
            "by_category": {},
            "data_freshness": {},
            "frequency_distribution": {}
        }
        
        economic_dir = os.path.join(self.actual_data_root, "economic_data")
        available_count = 0
        categories = {}
        frequencies = {}
        
        for indicator in self.economic_indicators:
            try:
                # Try to get economic data using Odin's Eye
                data = self.eye.get_economic_data(indicator)
                
                if data is not None and not data.empty:
                    # Calculate metrics for this indicator
                    null_count = int(data['value'].isnull().sum()) if 'value' in data.columns else 0
                    
                    # Data freshness analysis
                    latest_date = None
                    days_since_update = None
                    if 'date' in data.columns and not data.empty:
                        latest_date = data['date'].max()
                        try:
                            if hasattr(latest_date, 'date'):
                                days_since_update = (datetime.now().date() - latest_date.date()).days
                            elif isinstance(latest_date, pd.Timestamp):
                                days_since_update = (datetime.now() - latest_date).days
                            else:
                                latest_date_dt = pd.to_datetime(latest_date)
                                days_since_update = (datetime.now() - latest_date_dt).days
                        except Exception:
                            days_since_update = None
                    
                    metrics = {
                        "available": True,
                        "rows": len(data),
                        "date_range": {
                            "start": data['date'].min().isoformat() if 'date' in data.columns and not data.empty else None,
                            "end": data['date'].max().isoformat() if 'date' in data.columns and not data.empty else None
                        },
                        "null_count": null_count,
                        "null_percentage": round((null_count / len(data)) * 100, 2) if len(data) > 0 else 0,
                        "file_size_mb": self._get_file_size(os.path.join(economic_dir, f"{indicator}.parquet")),
                        "days_since_last_update": days_since_update,
                        "data_freshness_status": self._get_freshness_status(days_since_update),
                        "indicator_info": {
                            "category": data['category'].iloc[0] if 'category' in data.columns and len(data) > 0 else "Unknown",
                            "frequency": data['frequency'].iloc[0] if 'frequency' in data.columns and len(data) > 0 else "Unknown",
                            "importance": data['importance'].iloc[0] if 'importance' in data.columns and len(data) > 0 else "Unknown"
                        }
                    }
                    
                    # Track categories and frequencies
                    category = metrics["indicator_info"]["category"]
                    frequency = metrics["indicator_info"]["frequency"]
                    categories[category] = categories.get(category, 0) + 1
                    frequencies[frequency] = frequencies.get(frequency, 0) + 1
                    
                    # Enhanced quality checks
                    issues = []
                    
                    # High null percentage
                    if metrics["null_percentage"] > 10:
                        issues.append({
                            "indicator": indicator,
                            "issue": "High null value percentage",
                            "details": f"{metrics['null_percentage']:.1f}% null values",
                            "severity": "high" if metrics["null_percentage"] > 25 else "medium"
                        })
                    
                    # Data staleness
                    if days_since_update and days_since_update > 90:
                        issues.append({
                            "indicator": indicator,
                            "issue": "Stale data",
                            "details": f"Last updated {days_since_update} days ago",
                            "severity": "high" if days_since_update > 180 else "medium"
                        })
                    
                    # Data completeness - check for recent gaps
                    if 'date' in data.columns and len(data) > 1:
                        data_sorted = data.sort_values('date')
                        recent_data = data_sorted.tail(12)  # Last 12 observations
                        if len(recent_data) < 6:  # Less than 6 recent observations
                            issues.append({
                                "indicator": indicator,
                                "issue": "Insufficient recent data",
                                "details": f"Only {len(recent_data)} recent observations",
                                "severity": "medium"
                            })
                    
                    # Check for extreme values (potential data errors)
                    if 'value' in data.columns:
                        values = data['value'].dropna()
                        if len(values) > 10:
                            q99 = values.quantile(0.99)
                            q01 = values.quantile(0.01)
                            extreme_values = ((values > q99 * 10) | (values < q01 * 0.1)).sum()
                            if extreme_values > 0:
                                issues.append({
                                    "indicator": indicator,
                                    "issue": "Potential extreme values detected",
                                    "details": f"{extreme_values} values outside reasonable bounds",
                                    "severity": "low"
                                })
                    
                    economic_report["data_quality_issues"].extend(issues)
                    economic_report["available_indicators"][indicator] = metrics
                    available_count += 1
                else:
                    economic_report["missing_indicators"].append(indicator)
                    
            except Exception as e:
                economic_report["missing_indicators"].append(indicator)
                economic_report["data_quality_issues"].append({
                    "indicator": indicator,
                    "issue": "Data access failed",
                    "details": str(e),
                    "severity": "high"
                })
        
        economic_report["coverage_percentage"] = round((available_count / len(self.economic_indicators)) * 100, 2)
        economic_report["by_category"] = categories
        economic_report["frequency_distribution"] = frequencies
        
        self.report_data["economic_data_analysis"] = economic_report
        return economic_report
    
    def _get_freshness_status(self, days_since_update: Optional[int]) -> str:
        """Determine data freshness status based on days since last update."""
        if days_since_update is None:
            return "unknown"
        elif days_since_update <= 7:
            return "fresh"
        elif days_since_update <= 30:
            return "recent"
        elif days_since_update <= 90:
            return "aging"
        else:
            return "stale"
    
    def evaluate_news_data_coverage(self) -> Dict[str, Any]:
        """Evaluate coverage and quality of news data."""
        self.logger.info("Starting news data evaluation...")
        
        news_report = {
            "symbol_files_found": 0,
            "aggregate_files_found": 0,
            "total_articles": 0,
            "articles_by_symbol": {},
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "source_distribution": {},
            "dummy_data_found": [],
            "data_quality_issues": [],
            "storage_size_mb": 0,
            "date_coverage": {},
            "real_vs_dummy_ratio": {"real": 0, "dummy": 0}
        }
        
        news_dir = os.path.join(self.actual_data_root, "news_data")
        
        if os.path.exists(news_dir):
            # Analyze symbol-specific news files
            symbol_news_files = [f for f in os.listdir(news_dir) if f.endswith('.json') and '_news.json' in f and not f.startswith('news_')]
            aggregate_news_files = [f for f in os.listdir(news_dir) if f.endswith('.json') and f.startswith('news_')]
            
            news_report["symbol_files_found"] = len(symbol_news_files)
            news_report["aggregate_files_found"] = len(aggregate_news_files)
            
            total_size = 0
            total_articles = 0
            real_articles = 0
            dummy_articles = 0
            
            # Analyze sample of symbol files for quality
            sample_files = symbol_news_files[:50] if len(symbol_news_files) > 50 else symbol_news_files
            
            for file in sample_files:
                file_path = os.path.join(news_dir, file)
                file_size = self._get_file_size(file_path)
                total_size += file_size
                
                symbol = file.replace('_news.json', '')
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        news_data = json.load(f)
                        
                        if isinstance(news_data, list):
                            symbol_articles = len(news_data)
                            total_articles += symbol_articles
                            news_report["articles_by_symbol"][symbol] = symbol_articles
                            
                            # Analyze article quality
                            symbol_real = 0
                            symbol_dummy = 0
                            
                            for article in news_data[:10]:  # Sample first 10 articles per symbol
                                # Check for dummy data
                                headline = article.get('headline', '')
                                source = article.get('source', '')
                                
                                is_dummy = (
                                    source == 'sample_news_generator' or
                                    'Sample news headline' in headline or
                                    headline.startswith('Sample news headline')
                                )
                                
                                if is_dummy:
                                    symbol_dummy += 1
                                    dummy_articles += 1
                                    if symbol not in news_report["dummy_data_found"]:
                                        news_report["dummy_data_found"].append(symbol)
                                else:
                                    symbol_real += 1
                                    real_articles += 1
                                    
                                    # Analyze real article quality
                                    sentiment = article.get('sentiment_label', 'neutral')
                                    if sentiment in news_report["sentiment_distribution"]:
                                        news_report["sentiment_distribution"][sentiment] += 1
                                    
                                    source = article.get('source', 'unknown')
                                    news_report["source_distribution"][source] = news_report["source_distribution"].get(source, 0) + 1
                                    
                                    # Date coverage
                                    date = article.get('date', '')[:7]  # YYYY-MM format
                                    news_report["date_coverage"][date] = news_report["date_coverage"].get(date, 0) + 1
                                    
                                    # Quality checks
                                    if not headline.strip():
                                        news_report["data_quality_issues"].append({
                                            "symbol": symbol,
                                            "issue": "Empty headline",
                                            "severity": "medium"
                                        })
                                    
                                    if len(headline) < 10:
                                        news_report["data_quality_issues"].append({
                                            "symbol": symbol,
                                            "issue": "Very short headline",
                                            "details": f"Only {len(headline)} characters",
                                            "severity": "low"
                                        })
                                    
                                    if not article.get('date'):
                                        news_report["data_quality_issues"].append({
                                            "symbol": symbol,
                                            "issue": "Missing date",
                                            "severity": "high"
                                        })
                            
                            # Report dummy data contamination
                            if symbol_dummy > 0:
                                contamination_rate = symbol_dummy / (symbol_dummy + symbol_real) * 100
                                news_report["data_quality_issues"].append({
                                    "symbol": symbol,
                                    "issue": "Dummy data contamination",
                                    "details": f"{contamination_rate:.1f}% dummy articles ({symbol_dummy}/{symbol_dummy + symbol_real})",
                                    "severity": "high" if contamination_rate > 50 else "medium"
                                })
                        else:
                            news_report["data_quality_issues"].append({
                                "file": file,
                                "issue": "Invalid JSON structure - not an array",
                                "severity": "high"
                            })
                            
                except json.JSONDecodeError as e:
                    news_report["data_quality_issues"].append({
                        "file": file,
                        "issue": f"Invalid JSON format: {str(e)}",
                        "severity": "high"
                    })
                except Exception as e:
                    news_report["data_quality_issues"].append({
                        "file": file,
                        "issue": f"File error: {str(e)}",
                        "severity": "medium"
                    })
            
            news_report["total_articles"] = total_articles
            news_report["storage_size_mb"] = round(total_size, 2)
            news_report["real_vs_dummy_ratio"] = {"real": real_articles, "dummy": dummy_articles}
            
            # Calculate data quality metrics
            if real_articles + dummy_articles > 0:
                news_report["real_data_percentage"] = round(real_articles / (real_articles + dummy_articles) * 100, 2)
            else:
                news_report["real_data_percentage"] = 0
        
        self.report_data["news_data_analysis"] = news_report
        return news_report
    
    def evaluate_trends_data_coverage(self) -> Dict[str, Any]:
        """Evaluate coverage and quality of Google Trends data."""
        self.logger.info("Starting trends data evaluation...")
        
        trends_report = {
            "trends_files_found": 0,
            "keywords_tracked": [],
            "symbols_with_trends": [],
            "storage_size_mb": 0,
            "data_quality_issues": [],
            "date_coverage": {},
            "search_interest_stats": {},
            "related_queries_coverage": 0
        }
        
        trends_dir = os.path.join(self.actual_data_root, "trends_data")
        
        if os.path.exists(trends_dir):
            trends_files = [f for f in os.listdir(trends_dir) if f.endswith('.json')]
            trends_report["trends_files_found"] = len(trends_files)
            
            total_size = 0
            keywords_found = set()
            symbols_found = set()
            search_scores = []
            files_with_related_queries = 0
            
            # Sample files for analysis
            sample_files = trends_files[:50] if len(trends_files) > 50 else trends_files
            
            for file in sample_files:
                file_path = os.path.join(trends_dir, file)
                file_size = self._get_file_size(file_path)
                total_size += file_size
                
                try:
                    # Extract keyword from filename
                    parts = file.replace('.json', '').split('_')
                    if len(parts) >= 2:
                        keyword = '_'.join(parts[:-1])
                        keywords_found.add(keyword)
                        
                        if keyword.upper() in self.all_instruments:
                            symbols_found.add(keyword.upper())
                    
                    # Analyze file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        trends_data = json.load(f)
                        
                        # Quality checks for trends data structure
                        if isinstance(trends_data, dict):
                            # Check for search interest data
                            if 'search_interest' in trends_data:
                                interest_data = trends_data['search_interest']
                                if isinstance(interest_data, list) and len(interest_data) > 0:
                                    # Analyze search scores
                                    for entry in interest_data[:10]:  # Sample first 10 entries
                                        score = entry.get('search_score', 0)
                                        if isinstance(score, (int, float)) and 0 <= score <= 100:
                                            search_scores.append(score)
                                        
                                        # Date coverage
                                        date = entry.get('date', '')[:7]  # YYYY-MM format
                                        if date:
                                            trends_report["date_coverage"][date] = trends_report["date_coverage"].get(date, 0) + 1
                                else:
                                    trends_report["data_quality_issues"].append({
                                        "file": file,
                                        "issue": "Empty or invalid search interest data",
                                        "severity": "high"
                                    })
                            else:
                                trends_report["data_quality_issues"].append({
                                    "file": file,
                                    "issue": "Missing search_interest field",
                                    "severity": "high"
                                })
                            
                            # Check for related queries
                            if 'related_queries' in trends_data:
                                related_queries = trends_data['related_queries']
                                if isinstance(related_queries, dict) and len(related_queries) > 0:
                                    files_with_related_queries += 1
                            
                            # Check for metadata
                            required_fields = ['keyword', 'geo', 'timeframe']
                            for field in required_fields:
                                if field not in trends_data:
                                    trends_report["data_quality_issues"].append({
                                        "file": file,
                                        "issue": f"Missing {field} metadata",
                                        "severity": "medium"
                                    })
                        else:
                            trends_report["data_quality_issues"].append({
                                "file": file,
                                "issue": "Invalid JSON structure - not an object",
                                "severity": "high"
                            })
                            
                except json.JSONDecodeError as e:
                    trends_report["data_quality_issues"].append({
                        "file": file,
                        "issue": f"Invalid JSON format: {str(e)}",
                        "severity": "high"
                    })
                except Exception as e:
                    trends_report["data_quality_issues"].append({
                        "file": file,
                        "issue": f"File error: {str(e)}",
                        "severity": "medium"
                    })
            
            trends_report["keywords_tracked"] = sorted(list(keywords_found))
            trends_report["symbols_with_trends"] = sorted(list(symbols_found))
            trends_report["storage_size_mb"] = round(total_size, 2)
            
            # Calculate search interest statistics
            if search_scores:
                trends_report["search_interest_stats"] = {
                    "avg_score": round(sum(search_scores) / len(search_scores), 2),
                    "max_score": max(search_scores),
                    "min_score": min(search_scores),
                    "samples_analyzed": len(search_scores)
                }
            
            # Related queries coverage
            if len(sample_files) > 0:
                trends_report["related_queries_coverage"] = round((files_with_related_queries / len(sample_files)) * 100, 2)
        
        self.report_data["trends_data_analysis"] = trends_report
        return trends_report
    
    def evaluate_earnings_data_coverage(self) -> Dict[str, Any]:
        """Evaluate coverage and quality of earnings data."""
        self.logger.info("Starting earnings data evaluation...")
        
        earnings_report = {
            "earnings_files_found": 0,
            "calendar_files": 0,
            "individual_earnings_files": 0,
            "symbols_with_earnings": [],
            "data_quality_issues": [],
            "storage_size_mb": 0,
            "earnings_coverage": {},
            "temporal_coverage": {},
            "data_sources": []
        }
        
        earnings_dir = os.path.join(self.actual_data_root, "earnings_data")
        
        if os.path.exists(earnings_dir):
            earnings_files = [f for f in os.listdir(earnings_dir) if f.endswith('.json')]
            earnings_report["earnings_files_found"] = len(earnings_files)
            
            total_size = 0
            symbols_found = set()
            sources_found = set()
            
            for file in earnings_files:
                file_path = os.path.join(earnings_dir, file)
                file_size = self._get_file_size(file_path)
                total_size += file_size
                
                try:
                    # Categorize file types
                    if 'calendar' in file.lower():
                        earnings_report["calendar_files"] += 1
                    elif '_earnings.json' in file:
                        earnings_report["individual_earnings_files"] += 1
                        symbol = file.replace('_earnings.json', '')
                        symbols_found.add(symbol)
                    
                    # Analyze file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        earnings_data = json.load(f)
                        
                        if isinstance(earnings_data, dict):
                            # Individual earnings file
                            symbol = earnings_data.get('symbol')
                            if symbol:
                                symbols_found.add(symbol)
                                
                                # Check data sources
                                data_sources = earnings_data.get('data_sources', {})
                                for source_type, source_data in data_sources.items():
                                    if isinstance(source_data, dict):
                                        source_name = source_data.get('source')
                                        if source_name:
                                            sources_found.add(source_name)
                                
                                # Basic validation
                                if not data_sources:
                                    earnings_report["data_quality_issues"].append({
                                        "file": file,
                                        "issue": "No data sources found",
                                        "severity": "high"
                                    })
                            
                        elif isinstance(earnings_data, list):
                            # Earnings calendar
                            for entry in earnings_data[:10]:
                                if isinstance(entry, dict):
                                    symbol = entry.get('symbol')
                                    if symbol:
                                        symbols_found.add(symbol)
                                    
                                    # Check temporal coverage
                                    date = entry.get('date', '')
                                    if date:
                                        year_month = date[:7]
                                        earnings_report["temporal_coverage"][year_month] = \
                                            earnings_report["temporal_coverage"].get(year_month, 0) + 1
                        else:
                            earnings_report["data_quality_issues"].append({
                                "file": file,
                                "issue": "Invalid data structure",
                                "severity": "high"
                            })
                            
                except json.JSONDecodeError as e:
                    earnings_report["data_quality_issues"].append({
                        "file": file,
                        "issue": f"Invalid JSON: {str(e)}",
                        "severity": "high"
                    })
                except Exception as e:
                    earnings_report["data_quality_issues"].append({
                        "file": file,
                        "issue": f"File error: {str(e)}",
                        "severity": "medium"
                    })
            
            earnings_report["symbols_with_earnings"] = sorted(list(symbols_found))
            earnings_report["storage_size_mb"] = round(total_size, 2)
            earnings_report["data_sources"] = list(sources_found)
            
            # Coverage analysis
            earnings_coverage = len(symbols_found) / len(self.all_instruments) * 100 if self.all_instruments else 0
            earnings_report["earnings_coverage"] = {
                "symbols_with_data": len(symbols_found),
                "total_tracked_symbols": len(self.all_instruments),
                "coverage_percentage": round(earnings_coverage, 2)
            }
        
        self.report_data["earnings_data_analysis"] = earnings_report
        return earnings_report
    
    def _get_file_size(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            if os.path.exists(file_path):
                return round(os.path.getsize(file_path) / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0
    
    def analyze_missing_data_patterns(self) -> Dict[str, Any]:
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
        
        if "earnings_data_analysis" in self.report_data:
            earnings = self.report_data["earnings_data_analysis"]
            data_type_stats["earnings_data"] = {
                "files_found": earnings.get("earnings_files_found", 0),
                "symbols_with_earnings": len(earnings.get("symbols_with_earnings", [])),
                "calendar_files": earnings.get("calendar_files", 0),
                "individual_files": earnings.get("individual_earnings_files", 0),
                "storage_mb": earnings.get("storage_size_mb", 0),
                "coverage_percentage": earnings.get("earnings_coverage", {}).get("coverage_percentage", 0)
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
            
            # 3. Evaluate earnings data coverage
            self.evaluate_earnings_data_coverage()
            
            # 4. Evaluate news data coverage
            self.evaluate_news_data_coverage()
            
            # 5. Evaluate trends data coverage
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
            evaluator.evaluate_earnings_data_coverage()
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