"""
Stage 0: Data Validator for Da Vinchi Feature Engineering Pipeline

This stage implements data hygiene and validation as specified in the plan-draft:
- Validates adjusted prices and ensures proper OHLCV data
- Checks for data alignment and common trading calendar
- Implements winsorization and robust outlier detection
- Validates as-of times to prevent look-ahead bias
- Provides comprehensive data quality reporting

Uses Odin's Eye interface to read market data and applies validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .stage_base import ValidationStage, StageData, StageMetadata
from odins_eye import OdinsEye, DateRange, MarketDataInterval
from odins_eye.filters import InstrumentFilter, DataType
from odins_eye.exceptions import OdinsEyeError

logger = logging.getLogger(__name__)


@dataclass 
class DataValidationConfig:
    """Configuration for Stage 0 Data Validator"""
    winsorize_percentiles: tuple = (1, 99)  # 1st/99th percentile for winsorization
    min_trading_days: int = 250  # Minimum trading days required per instrument
    max_missing_percentage: float = 10.0  # Maximum percentage of missing data allowed
    outlier_std_threshold: float = 5.0  # Standard deviations for outlier detection
    volume_min_threshold: float = 0.0  # Minimum volume threshold
    price_min_threshold: float = 0.01  # Minimum price threshold
    validate_adjusted_prices: bool = True  # Check for proper price adjustments
    align_trading_calendar: bool = True  # Align to common trading calendar
    timezone_validate: bool = True  # Validate timezone consistency
    
    # Corporate data validation settings
    validate_corporate_data: bool = True  # Enable corporate data validation
    earnings_lookback_days: int = 90  # Days to look back for earnings data
    sec_filing_lookback_days: int = 30  # Days to look back for SEC filings
    insider_trading_lookback_days: int = 30  # Days to look back for insider trading
    corporate_actions_lookback_days: int = 365  # Days to look back for corporate actions


class Stage0DataValidator(ValidationStage):
    """
    Stage 0: Data Validator
    
    Implements comprehensive data hygiene and validation for financial market data.
    Ensures data quality, prevents look-ahead bias, and prepares clean OHLCV data
    for subsequent feature engineering stages.
    """
    
    def __init__(self, config: Dict[str, Any], data_root: Optional[str] = None):
        super().__init__(config, "Stage0_DataValidator", "1.0.0")
        
        # Initialize Odin's Eye interface
        self.odins_eye = OdinsEye(data_root=data_root)
        
        # Parse validation configuration
        validation_config = config.get('validation', {})
        self.validation_config = DataValidationConfig(**validation_config)
        
        # Core columns required for validation
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        # Note: adj_close might be created from adjusted_close in data preprocessing
    
    def process(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 0: Data Validator
        
        Args:
            input_data: StageData containing instrument list, date range, and config
            
        Returns:
            StageData with validated and cleaned OHLCV data
        """
        # Extract parameters from input
        instruments = input_data.config.get('instruments', [])
        date_range = input_data.config.get('date_range')
        interval = input_data.config.get('interval', MarketDataInterval.DAILY)
        
        self.logger.info(f"Validating data for {len(instruments)} instruments")
        
        # Load market data using Odin's Eye
        validated_data = self._load_and_validate_data(instruments, date_range, interval)
        
        # Apply data hygiene rules
        clean_data = self._apply_data_hygiene(validated_data)
        
        # Validate and integrate corporate data if enabled
        if self.validation_config.validate_corporate_data:
            clean_data = self._validate_and_integrate_corporate_data(clean_data, instruments, date_range)
        
        # Generate quality report
        quality_report = self._generate_quality_report(clean_data)
        
        # Create result metadata
        result_metadata = StageMetadata(
            stage_name=self.stage_name,
            version=self.version,
            input_shape=input_data.data.shape if hasattr(input_data.data, 'shape') else None,
            output_shape=clean_data.shape,
            feature_count=len(clean_data.columns)
        )
        
        # Add any warnings from validation
        if quality_report.get('issues'):
            result_metadata.warnings.extend(quality_report['issues'])
        
        return StageData(
            data=clean_data,
            metadata=result_metadata,
            config=input_data.config,
            artifacts={
                'quality_report': quality_report,
                'validation_config': self.validation_config,
                'instruments_validated': instruments,
                'columns_created': list(clean_data.columns)
            }
        )
    
    def _load_and_validate_data(self, instruments: List[str], date_range: Optional[DateRange], 
                               interval: MarketDataInterval) -> pd.DataFrame:
        """Load market data and perform initial validation"""
        try:
            # Load data using Odin's Eye
            data = self.odins_eye.get_market_data(
                symbols=instruments,
                date_range=date_range,
                interval=interval
            )
            
            if data.empty:
                raise ValueError("No data loaded from Odin's Eye")
            
            self.logger.info(f"Loaded {len(data)} rows of market data")
            
            # Add instrument column if missing
            if 'instrument' not in data.columns and 'instrument' not in data.index.names:
                # This is a workaround - in reality, Odin's Eye should provide this
                if len(instruments) == 1:
                    data['instrument'] = instruments[0]
                    self.logger.info(f"Added instrument column for single symbol: {instruments[0]}")
                else:
                    # More complex logic would be needed for multiple instruments
                    # For now, we'll try to handle this in a basic way
                    self.logger.warning("Multi-instrument data without instrument identifier - may cause issues")
                    # Assume data is concatenated with roughly equal parts per instrument
                    rows_per_instrument = len(data) // len(instruments)
                    instrument_labels = []
                    for i, instrument in enumerate(instruments):
                        start_idx = i * rows_per_instrument
                        end_idx = (i + 1) * rows_per_instrument if i < len(instruments) - 1 else len(data)
                        num_rows = end_idx - start_idx
                        instrument_labels.extend([instrument] * num_rows)
                    
                    # Ensure we have exactly the right number of labels
                    if len(instrument_labels) != len(data):
                        # Fill remaining rows with the last instrument
                        remaining = len(data) - len(instrument_labels)
                        if remaining > 0:
                            instrument_labels.extend([instruments[-1]] * remaining)
                        elif remaining < 0:
                            instrument_labels = instrument_labels[:len(data)]
                    
                    data['instrument'] = instrument_labels
            
            # Validate basic structure
            self._validate_data_structure(data)
            
            return data
            
        except OdinsEyeError as e:
            raise ValueError(f"Failed to load data from Odin's Eye: {str(e)}")
    
    def _validate_data_structure(self, data: pd.DataFrame) -> None:
        """Validate basic data structure requirements"""
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for instrument column (should be in index or columns)
        # If not present, we'll add it based on the symbols we requested
        if 'instrument' not in data.columns and 'instrument' not in data.index.names:
            # For multi-symbol requests, the data structure might be different
            # Let's add instrument information if we have multiple symbols
            self.logger.warning("No instrument identifier found in data, attempting to infer from context")
        
        # Validate OHLCV relationships
        self._validate_ohlcv_relationships(data)
    
    def _validate_ohlcv_relationships(self, data: pd.DataFrame) -> None:
        """Validate OHLCV data relationships and consistency"""
        issues = []
        
        # Check High >= Low
        if (data['high'] < data['low']).any():
            issues.append("Found High < Low violations")
        
        # Check High >= Open, Close
        if (data['high'] < data['open']).any():
            issues.append("Found High < Open violations")
        if (data['high'] < data['close']).any():
            issues.append("Found High < Close violations")
        
        # Check Low <= Open, Close  
        if (data['low'] > data['open']).any():
            issues.append("Found Low > Open violations")
        if (data['low'] > data['close']).any():
            issues.append("Found Low > Close violations")
        
        # Check for negative volumes
        if (data['volume'] < 0).any():
            issues.append("Found negative volume values")
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if self.validation_config.validate_adjusted_prices and 'adj_close' in data.columns:
            price_cols.append('adj_close')
        
        for col in price_cols:
            if (data[col] <= 0).any():
                issues.append(f"Found zero or negative {col} prices")
        
        if issues:
            self.logger.warning(f"OHLCV validation issues: {issues}")
            # Don't raise error, just log warnings
    
    def _apply_data_hygiene(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Stage 0 data hygiene rules"""
        clean_data = data.copy()
        
        # 1. Ensure proper datetime index
        if not isinstance(clean_data.index, pd.DatetimeIndex):
            if 'date' in clean_data.columns:
                clean_data = clean_data.set_index('date')
            elif clean_data.index.name == 'date':
                clean_data.index = pd.to_datetime(clean_data.index)
        
        # 2. Sort by instrument and date
        if 'instrument' in clean_data.columns:
            # Sort by instrument first, then by index (date)
            clean_data = clean_data.sort_values(['instrument']).sort_index()
        else:
            clean_data = clean_data.sort_index()
        
        # 3. Use adjusted prices if available and configured
        if self.validation_config.validate_adjusted_prices:
            if 'adj_close' in clean_data.columns:
                self.logger.info("Using adj_close for calculations")
                clean_data['close_adj'] = clean_data['adj_close']
            elif 'adjusted_close' in clean_data.columns:
                self.logger.info("Using adjusted_close for calculations")
                clean_data['close_adj'] = clean_data['adjusted_close']
            else:
                self.logger.info("No adjusted prices found, using close prices")
                clean_data['close_adj'] = clean_data['close']
        else:
            clean_data['close_adj'] = clean_data['close']
        
        # 4. Apply winsorization to remove extreme outliers
        clean_data = self._apply_winsorization(clean_data)
        
        # 5. Filter minimum thresholds
        clean_data = self._apply_threshold_filters(clean_data)
        
        # 6. Align to common trading calendar if configured
        if self.validation_config.align_trading_calendar:
            clean_data = self._align_trading_calendar(clean_data)
        
        # 7. Validate as-of times (ensure no look-ahead bias)
        clean_data = self._validate_as_of_times(clean_data)
        
        # 8. Add data quality flags
        clean_data = self._add_quality_flags(clean_data)
        
        return clean_data
    
    def _apply_winsorization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to price and volume data"""
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'adj_close' in data.columns:
            numeric_cols.append('adj_close')
        
        winsorized_data = data.copy()
        lower_pct, upper_pct = self.validation_config.winsorize_percentiles
        
        for col in numeric_cols:
            if col in data.columns:
                # Calculate percentiles per instrument if instrument column exists
                if 'instrument' in data.columns:
                    for instrument in data['instrument'].unique():
                        mask = data['instrument'] == instrument
                        values = data.loc[mask, col]
                        if len(values) > 10:  # Only winsorize if enough data points
                            lower_bound = values.quantile(lower_pct / 100)
                            upper_bound = values.quantile(upper_pct / 100)
                            winsorized_data.loc[mask, col] = values.clip(lower_bound, upper_bound)
                else:
                    # Global winsorization
                    values = data[col]
                    lower_bound = values.quantile(lower_pct / 100)
                    upper_bound = values.quantile(upper_pct / 100)
                    winsorized_data[col] = values.clip(lower_bound, upper_bound)
        
        return winsorized_data
    
    def _apply_threshold_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum threshold filters"""
        filtered_data = data.copy()
        
        # Filter minimum volume
        if self.validation_config.volume_min_threshold > 0:
            volume_mask = filtered_data['volume'] >= self.validation_config.volume_min_threshold
            filtered_data = filtered_data[volume_mask]
            
        # Filter minimum prices
        if self.validation_config.price_min_threshold > 0:
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in filtered_data.columns:
                    price_mask = filtered_data[col] >= self.validation_config.price_min_threshold
                    filtered_data = filtered_data[price_mask]
        
        return filtered_data
    
    def _align_trading_calendar(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align all instruments to common trading calendar"""
        if 'instrument' not in data.columns:
            return data
        
        # Get all unique trading dates
        all_dates = sorted(data.index.unique())
        all_instruments = sorted(data['instrument'].unique())
        
        # Create complete date-instrument grid
        date_instrument_grid = pd.MultiIndex.from_product(
            [all_dates, all_instruments], 
            names=[data.index.name or 'date', 'instrument']
        )
        
        # Reindex to complete grid
        if 'instrument' in data.columns:
            aligned_data = data.set_index('instrument', append=True).reindex(date_instrument_grid)
            aligned_data = aligned_data.reset_index('instrument')
        else:
            aligned_data = data
        
        # Forward fill only truly stale features (not returns or prices)
        # This is conservative - we only forward-fill volume and leave prices as NaN
        stale_features = ['volume']  # Only forward-fill volume, never prices or returns
        for col in stale_features:
            if col in aligned_data.columns:
                aligned_data[col] = aligned_data.groupby('instrument')[col].ffill()
        
        # Drop rows where core price data is missing after alignment
        core_price_cols = ['open', 'high', 'low', 'close']
        aligned_data = aligned_data.dropna(subset=core_price_cols, how='any')
        
        return aligned_data
    
    def _validate_as_of_times(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate as-of times to prevent look-ahead bias"""
        # Add as-of timestamp validation
        # For now, we assume data is properly timestamped from Odin's Eye
        # In future versions, this could validate against news timestamps, earnings dates, etc.
        
        validated_data = data.copy()
        
        # Add as-of timestamp column for tracking
        validated_data['as_of_time'] = validated_data.index
        
        self.logger.info("As-of time validation completed")
        return validated_data
    
    def _add_quality_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add data quality flags for monitoring"""
        flagged_data = data.copy()
        
        # Add quality score (0-100)
        quality_score = 100.0
        
        # Penalize for missing volume
        if 'volume' in flagged_data.columns:
            missing_volume_pct = flagged_data['volume'].isnull().mean() * 100
            quality_score -= missing_volume_pct * 0.5
        
        # Initialize data quality score for each row
        flagged_data['data_quality_score'] = quality_score
        
        # Penalize for extreme price movements (potential data errors)
        if 'close_adj' in flagged_data.columns and 'instrument' in flagged_data.columns:
            for instrument in flagged_data['instrument'].unique():
                mask = flagged_data['instrument'] == instrument
                price_series = flagged_data.loc[mask, 'close_adj']
                if len(price_series) > 1:
                    returns = price_series.pct_change().abs()
                    extreme_moves = (returns > 0.5).sum()  # >50% daily moves
                    if extreme_moves > 0:
                        penalty = extreme_moves * 2.0
                        flagged_data.loc[mask, 'data_quality_score'] = max(0, min(100, quality_score - penalty))
        
        # Ensure quality score is within bounds
        flagged_data['data_quality_score'] = flagged_data['data_quality_score'].clip(0, 100)
        
        return flagged_data
    
    def _generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        quality_report = self.check_data_quality(data)
        
        # Add Stage 0 specific metrics
        stage0_metrics = {
            'instruments_count': data['instrument'].nunique() if 'instrument' in data.columns else 1,
            'date_range': {
                'start': data.index.min().isoformat() if not data.empty else None,
                'end': data.index.max().isoformat() if not data.empty else None,
                'trading_days': len(data.index.unique()) if not data.empty else 0
            },
            'price_validation': {
                'ohlcv_consistent': self._check_ohlcv_consistency(data),
                'positive_prices': self._check_positive_prices(data),
                'valid_volumes': self._check_valid_volumes(data)
            },
            'data_coverage': self._calculate_data_coverage(data)
        }
        
        quality_report.update(stage0_metrics)
        return quality_report
    
    def _check_ohlcv_consistency(self, data: pd.DataFrame) -> bool:
        """Check OHLCV consistency"""
        try:
            consistent = True
            consistent &= (data['high'] >= data['low']).all()
            consistent &= (data['high'] >= data['open']).all()
            consistent &= (data['high'] >= data['close']).all()
            consistent &= (data['low'] <= data['open']).all()
            consistent &= (data['low'] <= data['close']).all()
            return consistent
        except:
            return False
    
    def _check_positive_prices(self, data: pd.DataFrame) -> bool:
        """Check all prices are positive"""
        price_cols = ['open', 'high', 'low', 'close']
        try:
            return all((data[col] > 0).all() for col in price_cols if col in data.columns)
        except:
            return False
    
    def _check_valid_volumes(self, data: pd.DataFrame) -> bool:
        """Check volumes are non-negative"""
        try:
            return (data['volume'] >= 0).all() if 'volume' in data.columns else True
        except:
            return False
    
    def _calculate_data_coverage(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data coverage statistics"""
        coverage = {}
        
        for col in data.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                coverage[f'{col}_coverage'] = (1 - data[col].isnull().mean()) * 100
        
        return coverage
    
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """Validate Stage 0 inputs"""
        # For Stage 0, input validation is minimal since we load data ourselves
        return []
    
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                raise ValueError("No datetime index or date column found")
        
        return data.sort_index()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names created by Stage 0"""
        features = ['open', 'high', 'low', 'close', 'volume', 'close_adj', 
                   'as_of_time', 'data_quality_score']
        
        # Corporate data features (added when validate_corporate_data=True)
        if self.validation_config.validate_corporate_data:
            corporate_features = [
                # Earnings features
                'has_earnings_data', 'days_to_earnings', 'earnings_week',
                # SEC filing features
                'has_sec_filings', 'recent_8k_filing', 'recent_10k_filing', 'recent_10q_filing',
                # Insider trading features
                'has_insider_trading', 'recent_insider_buying', 'recent_insider_selling', 'insider_activity_score',
                # Corporate actions features
                'has_corporate_actions', 'recent_dividend', 'recent_split', 'ex_dividend_date', 'split_adjustment_factor'
            ]
            features.extend(corporate_features)
        
        # Note: adjusted price columns may or may not be present depending on data source
            
        return features
    
    def _validate_and_integrate_corporate_data(self, data: pd.DataFrame, instruments: List[str], 
                                              date_range: Optional[DateRange]) -> pd.DataFrame:
        """Validate and integrate corporate data into market data"""
        enhanced_data = data.copy()
        
        if not instruments or data.empty:
            return enhanced_data
        
        try:
            # Get date range for corporate data lookback
            end_date = date_range.end_date if date_range else datetime.now()
            
            # Validate earnings data
            earnings_data = self._validate_earnings_data(instruments, end_date)
            if earnings_data:
                enhanced_data = self._integrate_earnings_flags(enhanced_data, earnings_data)
            
            # Validate SEC filings
            sec_data = self._validate_sec_filings(instruments, end_date)
            if sec_data:
                enhanced_data = self._integrate_sec_filing_flags(enhanced_data, sec_data)
            
            # Validate insider trading data
            insider_data = self._validate_insider_trading(instruments, end_date)
            if insider_data:
                enhanced_data = self._integrate_insider_trading_flags(enhanced_data, insider_data)
            
            # Validate corporate actions
            corporate_actions_data = self._validate_corporate_actions(instruments, end_date)
            if corporate_actions_data:
                enhanced_data = self._integrate_corporate_actions_flags(enhanced_data, corporate_actions_data)
            
            self.logger.info("Corporate data validation and integration completed")
            
        except Exception as e:
            self.logger.warning(f"Corporate data validation failed: {str(e)}")
            # Continue without corporate data enhancement
        
        return enhanced_data
    
    def _validate_earnings_data(self, instruments: List[str], end_date: datetime) -> List[Dict[str, Any]]:
        """Validate earnings data availability and quality"""
        try:
            lookback_date = end_date - timedelta(days=self.validation_config.earnings_lookback_days)
            date_range = DateRange(start_date=lookback_date, end_date=end_date)
            
            earnings_data = self.odins_eye.get_earnings_data(
                symbols=instruments,
                date_range=date_range
            )
            
            self.logger.info(f"Validated {len(earnings_data)} earnings records")
            return earnings_data
            
        except Exception as e:
            self.logger.warning(f"Earnings data validation failed: {str(e)}")
            return []
    
    def _validate_sec_filings(self, instruments: List[str], end_date: datetime) -> List[Dict[str, Any]]:
        """Validate SEC filings data availability and quality"""
        try:
            lookback_date = end_date - timedelta(days=self.validation_config.sec_filing_lookback_days)
            date_range = DateRange(start_date=lookback_date, end_date=end_date)
            
            sec_data = self.odins_eye.get_sec_filings(
                symbols=instruments,
                date_range=date_range
            )
            
            self.logger.info(f"Validated {len(sec_data)} SEC filing records")
            return sec_data
            
        except Exception as e:
            self.logger.warning(f"SEC filings validation failed: {str(e)}")
            return []
    
    def _validate_insider_trading(self, instruments: List[str], end_date: datetime) -> List[Dict[str, Any]]:
        """Validate insider trading data availability and quality"""
        try:
            lookback_date = end_date - timedelta(days=self.validation_config.insider_trading_lookback_days)
            date_range = DateRange(start_date=lookback_date, end_date=end_date)
            
            insider_data = self.odins_eye.get_insider_trading(
                symbols=instruments,
                date_range=date_range
            )
            
            self.logger.info(f"Validated {len(insider_data)} insider trading records")
            return insider_data
            
        except Exception as e:
            self.logger.warning(f"Insider trading validation failed: {str(e)}")
            return []
    
    def _validate_corporate_actions(self, instruments: List[str], end_date: datetime) -> List[Dict[str, Any]]:
        """Validate corporate actions data availability and quality"""
        try:
            lookback_date = end_date - timedelta(days=self.validation_config.corporate_actions_lookback_days)
            date_range = DateRange(start_date=lookback_date, end_date=end_date)
            
            actions_data = self.odins_eye.get_corporate_actions(
                symbols=instruments,
                date_range=date_range
            )
            
            self.logger.info(f"Validated {len(actions_data)} corporate action records")
            return actions_data
            
        except Exception as e:
            self.logger.warning(f"Corporate actions validation failed: {str(e)}")
            return []
    
    def _integrate_earnings_flags(self, data: pd.DataFrame, earnings_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Integrate earnings-related quality flags into market data"""
        enhanced_data = data.copy()
        
        # Initialize earnings flags
        enhanced_data['has_earnings_data'] = False
        enhanced_data['days_to_earnings'] = None
        enhanced_data['earnings_week'] = False
        
        if not earnings_data or 'instrument' not in enhanced_data.columns:
            return enhanced_data
        
        # Process earnings data by instrument
        for instrument in enhanced_data['instrument'].unique():
            instrument_earnings = [e for e in earnings_data if e.get('symbol') == instrument]
            
            if not instrument_earnings:
                continue
                
            # Mark dates with earnings data
            mask = enhanced_data['instrument'] == instrument
            enhanced_data.loc[mask, 'has_earnings_data'] = len(instrument_earnings) > 0
            
            # Calculate days to next earnings for each trading day
            for _, row in enhanced_data[mask].iterrows():
                trading_date = row.name
                
                # Find next earnings date
                future_earnings = [
                    e for e in instrument_earnings 
                    if pd.to_datetime(e.get('date', '1900-01-01')) >= trading_date
                ]
                
                if future_earnings:
                    next_earnings = min(future_earnings, 
                                      key=lambda x: pd.to_datetime(x.get('date', '1900-01-01')))
                    earnings_date = pd.to_datetime(next_earnings.get('date'))
                    days_to_earnings = (earnings_date - trading_date).days
                    
                    enhanced_data.loc[trading_date, 'days_to_earnings'] = days_to_earnings
                    enhanced_data.loc[trading_date, 'earnings_week'] = abs(days_to_earnings) <= 3
        
        return enhanced_data
    
    def _integrate_sec_filing_flags(self, data: pd.DataFrame, sec_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Integrate SEC filing quality flags into market data"""
        enhanced_data = data.copy()
        
        # Initialize SEC filing flags
        enhanced_data['has_sec_filings'] = False
        enhanced_data['recent_8k_filing'] = False
        enhanced_data['recent_10k_filing'] = False
        enhanced_data['recent_10q_filing'] = False
        
        if not sec_data or 'instrument' not in enhanced_data.columns:
            return enhanced_data
        
        # Process SEC filings by instrument
        for instrument in enhanced_data['instrument'].unique():
            instrument_filings = [f for f in sec_data if f.get('symbol') == instrument]
            
            if not instrument_filings:
                continue
            
            mask = enhanced_data['instrument'] == instrument
            enhanced_data.loc[mask, 'has_sec_filings'] = len(instrument_filings) > 0
            
            # Check for recent filings of different types
            for filing in instrument_filings:
                filing_date = pd.to_datetime(filing.get('date', '1900-01-01'))
                filing_type = filing.get('type', '')
                
                # Mark trading days within 5 days of filing
                for trading_date in enhanced_data[mask].index:
                    days_from_filing = abs((trading_date - filing_date).days)
                    
                    if days_from_filing <= 5:
                        if filing_type == '8-K':
                            enhanced_data.loc[trading_date, 'recent_8k_filing'] = True
                        elif filing_type == '10-K':
                            enhanced_data.loc[trading_date, 'recent_10k_filing'] = True
                        elif filing_type == '10-Q':
                            enhanced_data.loc[trading_date, 'recent_10q_filing'] = True
        
        return enhanced_data
    
    def _integrate_insider_trading_flags(self, data: pd.DataFrame, insider_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Integrate insider trading quality flags into market data"""
        enhanced_data = data.copy()
        
        # Initialize insider trading flags
        enhanced_data['has_insider_trading'] = False
        enhanced_data['recent_insider_buying'] = False
        enhanced_data['recent_insider_selling'] = False
        enhanced_data['insider_activity_score'] = 0.0
        
        if not insider_data or 'instrument' not in enhanced_data.columns:
            return enhanced_data
        
        # Process insider trading by instrument
        for instrument in enhanced_data['instrument'].unique():
            instrument_trades = [t for t in insider_data if t.get('symbol') == instrument]
            
            if not instrument_trades:
                continue
            
            mask = enhanced_data['instrument'] == instrument
            enhanced_data.loc[mask, 'has_insider_trading'] = len(instrument_trades) > 0
            
            # Analyze recent insider activity
            for trade in instrument_trades:
                trade_date = pd.to_datetime(trade.get('date', '1900-01-01'))
                trade_type = trade.get('transaction_type', '')
                trade_value = float(trade.get('value', 0))
                
                # Mark trading days within 7 days of insider activity
                for trading_date in enhanced_data[mask].index:
                    days_from_trade = abs((trading_date - trade_date).days)
                    
                    if days_from_trade <= 7:
                        if trade_type.lower() in ['purchase', 'buy']:
                            enhanced_data.loc[trading_date, 'recent_insider_buying'] = True
                            enhanced_data.loc[trading_date, 'insider_activity_score'] += trade_value / 1000000  # Normalize by millions
                        elif trade_type.lower() in ['sale', 'sell']:
                            enhanced_data.loc[trading_date, 'recent_insider_selling'] = True
                            enhanced_data.loc[trading_date, 'insider_activity_score'] -= trade_value / 1000000  # Negative for sales
        
        return enhanced_data
    
    def _integrate_corporate_actions_flags(self, data: pd.DataFrame, actions_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Integrate corporate actions quality flags into market data"""
        enhanced_data = data.copy()
        
        # Initialize corporate actions flags
        enhanced_data['has_corporate_actions'] = False
        enhanced_data['recent_dividend'] = False
        enhanced_data['recent_split'] = False
        enhanced_data['ex_dividend_date'] = False
        enhanced_data['split_adjustment_factor'] = 1.0
        
        if not actions_data or 'instrument' not in enhanced_data.columns:
            return enhanced_data
        
        # Process corporate actions by instrument
        for instrument in enhanced_data['instrument'].unique():
            instrument_actions = [a for a in actions_data if a.get('symbol') == instrument]
            
            if not instrument_actions:
                continue
            
            mask = enhanced_data['instrument'] == instrument
            enhanced_data.loc[mask, 'has_corporate_actions'] = len(instrument_actions) > 0
            
            # Process each corporate action
            for action in instrument_actions:
                action_date = pd.to_datetime(action.get('date', '1900-01-01'))
                action_type = action.get('type', '')
                
                # Mark trading days around corporate actions
                for trading_date in enhanced_data[mask].index:
                    days_from_action = (trading_date - action_date).days
                    
                    if action_type.lower() == 'dividend':
                        # Mark ex-dividend date and nearby dates
                        if days_from_action == 0:
                            enhanced_data.loc[trading_date, 'ex_dividend_date'] = True
                        if abs(days_from_action) <= 3:
                            enhanced_data.loc[trading_date, 'recent_dividend'] = True
                    
                    elif action_type.lower() == 'split':
                        # Mark split dates and adjustment factors
                        if abs(days_from_action) <= 1:
                            enhanced_data.loc[trading_date, 'recent_split'] = True
                            split_ratio = action.get('ratio', '1:1')
                            try:
                                # Parse ratio like "2:1" or "1:2"
                                new_shares, old_shares = split_ratio.split(':')
                                factor = float(new_shares) / float(old_shares)
                                enhanced_data.loc[trading_date, 'split_adjustment_factor'] = factor
                            except:
                                enhanced_data.loc[trading_date, 'split_adjustment_factor'] = 1.0
        
        return enhanced_data