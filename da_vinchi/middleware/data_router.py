"""
Data Router for Da Vinchi Pipeline.

Handles data routing between stages, buffering, and parallel processing coordination.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
from datetime import datetime

from ..core.stage_base import StageData

logger = logging.getLogger(__name__)


class DataBuffer:
    """Thread-safe data buffer for inter-stage communication"""
    
    def __init__(self, max_size_mb: int = 500):
        self.max_size_mb = max_size_mb
        self.buffer = Queue()
        self.lock = threading.Lock()
        self.current_size_mb = 0
        
    def put(self, data: StageData) -> bool:
        """
        Add data to buffer if space available.
        
        Args:
            data: StageData to buffer
            
        Returns:
            True if data was buffered, False if buffer full
        """
        data_size_mb = data.data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        with self.lock:
            if self.current_size_mb + data_size_mb > self.max_size_mb:
                return False
            
            self.buffer.put(data)
            self.current_size_mb += data_size_mb
            return True
    
    def get(self, timeout: float = 1.0) -> Optional[StageData]:
        """
        Get data from buffer.
        
        Args:
            timeout: Maximum time to wait for data
            
        Returns:
            StageData or None if timeout
        """
        try:
            data = self.buffer.get(timeout=timeout)
            data_size_mb = data.data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            with self.lock:
                self.current_size_mb -= data_size_mb
                
            return data
        except Empty:
            return None
    
    def size(self) -> int:
        """Get current buffer size"""
        return self.buffer.qsize()


class DataRouter:
    """
    Routes data between pipeline stages with buffering and coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffer_size_mb = config.get("buffer_size_mb", 500)
        self.parallel_processing = config.get("parallel_processing", True)
        
        # Data buffers for each stage
        self.buffers = {}
        self.executor = ThreadPoolExecutor(max_workers=4) if self.parallel_processing else None
        
        logger.info(f"Data router initialized (parallel: {self.parallel_processing})")
    
    def route_data(self, data: StageData, target_stage: str) -> StageData:
        """
        Route data to target stage with appropriate preprocessing.
        
        Args:
            data: Input StageData
            target_stage: Name of target stage
            
        Returns:
            Processed StageData ready for target stage
        """
        logger.debug(f"Routing data to {target_stage}")
        
        # Apply stage-specific data transformations
        processed_data = self._preprocess_for_stage(data, target_stage)
        
        # Apply data quality checks
        quality_issues = self._check_data_quality(processed_data)
        if quality_issues:
            processed_data.metadata.warnings.extend(quality_issues)
        
        # Buffer data if parallel processing
        if self.parallel_processing:
            buffer = self._get_buffer(target_stage)
            if not buffer.put(processed_data):
                logger.warning(f"Buffer full for {target_stage}, processing synchronously")
        
        return processed_data
    
    def _preprocess_for_stage(self, data: StageData, target_stage: str) -> StageData:
        """Apply stage-specific preprocessing"""
        processed_data = StageData(
            data=data.data.copy(),
            metadata=data.metadata,
            config=data.config,
            artifacts=data.artifacts.copy()
        )
        
        # Stage-specific preprocessing
        if target_stage == "stage0_data_validator":
            processed_data = self._prepare_validator_data(processed_data)
        elif target_stage == "stage1_ohlcv_features":
            processed_data = self._prepare_ohlcv_data(processed_data)
        elif target_stage == "stage2_cross_sectional":
            processed_data = self._prepare_cross_sectional_data(processed_data)
        elif target_stage == "stage3_regimes_seasonal":
            processed_data = self._prepare_regimes_seasonal_data(processed_data)
        elif target_stage == "stage4_relationships":
            processed_data = self._prepare_relationships_data(processed_data)
        elif target_stage == "stage6_news_sentiment":
            processed_data = self._prepare_news_sentiment_data(processed_data)
        elif target_stage == "stage6_alt_data":
            processed_data = self._prepare_alt_data(processed_data)
        
        return processed_data
    
    def _prepare_ohlcv_data(self, data: StageData) -> StageData:
        """Prepare data for OHLCV features stage"""
        df = data.data
        
        # Ensure required OHLCV columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            data.metadata.warnings.append(f"Missing OHLCV columns: {missing_cols}")
        
        # Ensure adjusted close exists
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close']
            data.metadata.warnings.append("Using close price as adjusted close")
        
        # Sort by date and instrument
        if 'instrument' in df.columns:
            df = df.sort_values(['instrument', df.index])
        else:
            df = df.sort_index()
        
        data.data = df
        return data
    
    def _prepare_cross_sectional_data(self, data: StageData) -> StageData:
        """Prepare data for cross-sectional stage"""
        df = data.data
        
        # Ensure we have multiple instruments for cross-sectional analysis
        if 'instrument' not in df.columns:
            data.metadata.warnings.append("No instrument column for cross-sectional analysis")
            return data
        
        instrument_counts = df['instrument'].value_counts()
        min_observations = 100
        
        # Filter instruments with sufficient data
        valid_instruments = instrument_counts[instrument_counts >= min_observations].index
        if len(valid_instruments) < len(instrument_counts):
            filtered_count = len(instrument_counts) - len(valid_instruments)
            data.metadata.warnings.append(f"Filtered {filtered_count} instruments with <{min_observations} observations")
            df = df[df['instrument'].isin(valid_instruments)]
        
        data.data = df
        return data
    
    def _prepare_relationships_data(self, data: StageData) -> StageData:
        """Prepare data for relationships stage"""
        df = data.data
        
        # Ensure we have return data for correlation analysis
        if 'log_ret' not in df.columns and 'close' in df.columns:
            if 'instrument' in df.columns:
                df['log_ret'] = df.groupby('instrument')['close'].pct_change().fillna(0)
            else:
                df['log_ret'] = df['close'].pct_change().fillna(0)
            data.metadata.warnings.append("Computed simple returns for correlation analysis")
        
        data.data = df
        return data
    
    def _prepare_alt_data(self, data: StageData) -> StageData:
        """Prepare data for alternative data stage"""
        # Alternative data preparation will be implemented when integrating news/sentiment
        return data
    
    def _check_data_quality(self, data: StageData) -> List[str]:
        """Check data quality and return list of issues"""
        issues = []
        df = data.data
        
        # Check for excessive missing data
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            issues.append(f"High missing data (>50%): {high_missing}")
        
        # Check for constant columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns detected: {constant_cols}")
        
        # Check for extreme outliers (beyond 5 standard deviations)
        for col in numeric_cols:
            if df[col].std() > 0:  # Avoid division by zero
                outliers = abs(df[col] - df[col].mean()) > 5 * df[col].std()
                outlier_count = outliers.sum()
                if outlier_count > len(df) * 0.01:  # More than 1% outliers
                    issues.append(f"Excessive outliers in {col}: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")
        
        # Corporate data specific quality checks
        issues.extend(self._check_corporate_data_quality(df))
        
        return issues
    
    def _check_corporate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check corporate data specific quality issues"""
        issues = []
        
        # Check earnings data consistency
        if 'days_to_earnings' in df.columns:
            invalid_earnings_dates = (df['days_to_earnings'] < -365) | (df['days_to_earnings'] > 365)
            if invalid_earnings_dates.sum() > 0:
                issues.append(f"Invalid earnings date calculations: {invalid_earnings_dates.sum()} rows")
        
        # Check insider activity score for reasonable ranges
        if 'insider_activity_score' in df.columns:
            extreme_scores = (df['insider_activity_score'].abs() > 1000).sum()
            if extreme_scores > 0:
                issues.append(f"Extreme insider activity scores (>$1B): {extreme_scores} rows")
        
        # Check split adjustment factors for reasonableness
        if 'split_adjustment_factor' in df.columns:
            invalid_splits = ((df['split_adjustment_factor'] < 0.1) | 
                            (df['split_adjustment_factor'] > 10)).sum()
            if invalid_splits > 0:
                issues.append(f"Invalid split adjustment factors: {invalid_splits} rows")
        
        # Check for data quality score consistency
        if 'data_quality_score' in df.columns:
            low_quality_count = (df['data_quality_score'] < 50).sum()
            if low_quality_count > len(df) * 0.1:  # More than 10% low quality
                issues.append(f"High proportion of low quality data: {low_quality_count}/{len(df)} rows")
        
        # Check for missing corporate data when expected
        corporate_cols = [col for col in df.columns if any(x in col.lower() for x in 
                         ['earnings', 'sec_filing', 'insider', 'corporate_action'])]
        
        if corporate_cols:
            for col in corporate_cols:
                if col in df.columns:
                    missing_rate = df[col].isnull().mean()
                    if missing_rate > 0.8:  # More than 80% missing
                        issues.append(f"High missing rate in corporate data column '{col}': {missing_rate*100:.1f}%")
        
        return issues
    
    def _get_buffer(self, stage_name: str) -> DataBuffer:
        """Get or create buffer for stage"""
        if stage_name not in self.buffers:
            self.buffers[stage_name] = DataBuffer(self.buffer_size_mb)
        return self.buffers[stage_name]
    
    def get_buffer_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all buffers"""
        return {
            name: {
                "size": buffer.size(),
                "current_size_mb": buffer.current_size_mb,
                "max_size_mb": buffer.max_size_mb
            }
            for name, buffer in self.buffers.items()
        }
    
    def clear_buffers(self) -> None:
        """Clear all buffers"""
        for buffer in self.buffers.values():
            while buffer.get(timeout=0.1) is not None:
                pass
        logger.info("All buffers cleared")
    
    def shutdown(self) -> None:
        """Shutdown the data router"""
        self.clear_buffers()
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Data router shutdown complete")
    
    def _prepare_validator_data(self, data: StageData) -> StageData:
        """Prepare data for Stage 0 Data Validator with corporate data routing"""
        df = data.data
        
        # Ensure basic structure for validation
        if df.empty:
            data.metadata.warnings.append("Empty dataset provided to validator")
            return data
        
        # Check for corporate data flags from previous processing
        corporate_flags = [col for col in df.columns if any(x in col.lower() for x in 
                          ['earnings', 'sec_filing', 'insider', 'corporate_action', 'dividend', 'split'])]
        
        if corporate_flags:
            data.artifacts['corporate_data_present'] = True
            data.artifacts['corporate_flags'] = corporate_flags
            logger.info(f"Corporate data flags detected: {len(corporate_flags)} columns")
        else:
            data.artifacts['corporate_data_present'] = False
            logger.debug("No corporate data flags detected in validator input")
        
        # Ensure proper datetime indexing for corporate event alignment
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                try:
                    df = df.set_index('date')
                    data.metadata.warnings.append("Converted date column to datetime index")
                except:
                    data.metadata.warnings.append("Failed to convert date column to datetime index")
        
        data.data = df
        return data
    
    def _prepare_regimes_seasonal_data(self, data: StageData) -> StageData:
        """Prepare data for Stage 3 Regimes & Seasonal with corporate event integration"""
        df = data.data
        
        # Check for corporate earnings flags
        earnings_cols = [col for col in df.columns if 'earnings' in col.lower()]
        if earnings_cols:
            data.artifacts['has_earnings_features'] = True
            data.artifacts['earnings_columns'] = earnings_cols
            logger.info(f"Earnings features available for seasonal analysis: {earnings_cols}")
        
        # Check for corporate action timing data
        corporate_timing_cols = [col for col in df.columns if any(x in col.lower() for x in 
                               ['days_to_earnings', 'earnings_week', 'recent_dividend', 'recent_split'])]
        
        if corporate_timing_cols:
            data.artifacts['has_corporate_timing'] = True
            data.artifacts['corporate_timing_columns'] = corporate_timing_cols
            logger.info(f"Corporate timing features available: {corporate_timing_cols}")
        else:
            data.metadata.warnings.append("No corporate timing features available - using date-based proxies only")
        
        data.data = df
        return data
    
    def _prepare_news_sentiment_data(self, data: StageData) -> StageData:
        """Prepare data for Stage 6 News Sentiment with corporate event context"""
        df = data.data
        
        # Check for corporate event flags that can enhance news sentiment
        event_context_cols = [col for col in df.columns if any(x in col.lower() for x in 
                            ['earnings_week', 'recent_8k', 'recent_10k', 'recent_10q', 
                             'insider_trading', 'recent_dividend', 'recent_split'])]
        
        if event_context_cols:
            data.artifacts['has_event_context'] = True
            data.artifacts['event_context_columns'] = event_context_cols
            logger.info(f"Corporate event context available for sentiment analysis: {event_context_cols}")
            
            # Create composite event score for news weighting
            event_score = 0
            for col in event_context_cols:
                if col in df.columns:
                    # Weight different event types
                    if 'earnings' in col:
                        event_score += df[col].astype(int) * 3  # Earnings events have high impact
                    elif '8k' in col or '10k' in col or '10q' in col:
                        event_score += df[col].astype(int) * 2  # SEC filings moderate impact
                    elif 'insider' in col:
                        event_score += df[col].astype(int) * 1  # Insider trading lower impact
                    elif 'dividend' in col or 'split' in col:
                        event_score += df[col].astype(int) * 2  # Corporate actions moderate impact
            
            df['corporate_event_intensity'] = event_score
            data.artifacts['computed_event_intensity'] = True
        else:
            data.metadata.warnings.append("No corporate event context available - news sentiment will use baseline approach")
        
        data.data = df
        return data