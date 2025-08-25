"""
Stage 2: Cross-sectional Risk & Market Context for Da Vinchi Feature Engineering Pipeline

This stage implements cross-sectional features as specified in the plan-draft:
- Rolling beta/alpha calculations vs benchmark
- Idiosyncratic volatility computation  
- Cross-sectional ranking within universe per day
- Market context features relative to benchmark

Requires Stage 1 features (returns) and benchmark data to compute relative metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from scipy import stats
from sklearn.preprocessing import RobustScaler

from .stage_base import FeatureStage, StageData, StageMetadata

logger = logging.getLogger(__name__)


class Stage2CrossSectional(FeatureStage):
    """
    Stage 2: Cross-sectional Risk & Market Context
    
    Computes market-relative features and cross-sectional rankings that provide
    context for each instrument relative to the broader market and peer universe.
    Essential for relative value and market-neutral strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "Stage2_CrossSectional", "1.0.0")
        
        # Cross-sectional parameters
        self.params = {
            # Beta/Alpha computation
            'beta_windows': config.get('beta_windows', [30, 60, 126, 252]),
            'min_observations': config.get('min_observations', 20),  # Min obs for beta calc
            'benchmark_symbol': config.get('benchmark_symbol', 'SPY'),
            'fallback_benchmark': config.get('fallback_benchmark', 'VOO'),
            
            # Cross-sectional ranking
            'ranking_windows': config.get('ranking_windows', [21, 63, 126, 252]),
            'ranking_method': config.get('ranking_method', 'percent_rank'),  # 'rank' or 'percent_rank'
            'ranking_features': config.get('ranking_features', [
                'return_21d', 'return_63d', 'return_126d', 'vol_cc_20d', 'vol_cc_60d'
            ]),
            
            # Universe definition
            'min_price': config.get('min_price', 1.0),  # Min price for universe inclusion
            'min_volume': config.get('min_volume', 100000),  # Min dollar volume
            'min_trading_days': config.get('min_trading_days', 15)  # Min days in window
        }
        
    def _get_required_columns(self) -> List[str]:
        """Required columns from Stage 1"""
        return ['close_adj', 'volume', 'log_return', 'return_21d', 'return_63d', 
                'return_126d', 'vol_cc_20d', 'vol_cc_60d']
    
    
    def process(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 2: Cross-sectional Features
        
        Args:
            input_data: StageData from Stage 1 with base features
            
        Returns:
            StageData with cross-sectional features added
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        self.logger.info(f"Generating cross-sectional features for {data.shape[0]} observations")
        
        # Get benchmark data from artifacts or create synthetic benchmark
        benchmark_data = input_data.artifacts.get('benchmark_data')
        if benchmark_data is None:
            self.logger.info("No benchmark data provided, creating synthetic benchmark")
            benchmark_data = self._create_synthetic_benchmark(data)
        
        data = self._generate_beta_alpha_features(data, benchmark_data)
        
        # Generate cross-sectional rankings
        data = self._generate_cross_sectional_rankings(data)
        
        # Generate universe membership features
        data = self._generate_universe_features(data)
        
        # Get list of new features created
        new_features = [col for col in data.columns if col not in input_data.data.columns]
        
        # Create result metadata
        result_metadata = StageMetadata(
            stage_name=self.stage_name,
            version=self.version,
            input_shape=input_data.data.shape,
            output_shape=data.shape,
            feature_count=len(new_features)
        )
        
        self.logger.info(f"Created {len(new_features)} cross-sectional features")
        
        return StageData(
            data=data,
            metadata=result_metadata,
            config=input_data.config,
            artifacts={
                **input_data.artifacts,
                'stage2_features': new_features,
                'stage2_params': self.params,
                'benchmark_symbol': self.params['benchmark_symbol']
            }
        )
    
    def _create_synthetic_benchmark(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a synthetic benchmark if none provided"""
        if 'instrument' not in data.columns:
            # Single instrument - create simple benchmark
            benchmark_returns = data['log_return'].rolling(5).mean().fillna(0) * 0.8
        else:
            # Multi-instrument - create equal-weighted index
            returns_pivot = data.pivot(index=data.index, columns='instrument', values='log_return')
            benchmark_returns = returns_pivot.mean(axis=1, skipna=True).fillna(0)
        
        return pd.DataFrame({
            'benchmark_return': benchmark_returns
        }, index=data.index)
    
    def _generate_beta_alpha_features(self, data: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling beta and alpha features (Section 2.1)"""
        self.logger.info("Generating beta and alpha features")
        
        # Merge with benchmark data
        data = data.join(benchmark, how='left')
        
        # Group by instrument if column exists
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Calculate rolling betas for different windows
        for window in self.params['beta_windows']:
            min_obs = max(self.params['min_observations'], window // 3)
            
            def rolling_beta(group_data):
                """Calculate rolling beta using linear regression"""
                result = pd.Series(index=group_data.index, dtype=float)
                
                for i in range(len(group_data)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < min_obs:
                        continue
                    
                    y = group_data['log_return'].iloc[start_idx:end_idx].dropna()
                    x = group_data['benchmark_return'].iloc[start_idx:end_idx].dropna()
                    
                    # Align the series
                    aligned_data = pd.DataFrame({'y': y, 'x': x}).dropna()
                    
                    if len(aligned_data) < min_obs:
                        continue
                    
                    # Calculate beta using covariance method (more numerically stable)
                    cov_xy = aligned_data['x'].cov(aligned_data['y'])
                    var_x = aligned_data['x'].var()
                    
                    if var_x > 1e-10:  # Avoid division by zero
                        beta = cov_xy / var_x
                        result.iloc[i] = beta
                
                return result
            
            # Initialize beta column
            data[f'beta_{window}d'] = np.nan
            
            # Apply rolling beta calculation by instrument
            for instrument, group in grouped:
                beta_series = rolling_beta(group)
                # Use loc to assign values to the correct rows
                mask = data['instrument'] == instrument
                data.loc[mask, f'beta_{window}d'] = beta_series.values
            
            # Calculate corresponding alpha (excess return)
            data[f'alpha_{window}d'] = (data['log_return'] - 
                                       data[f'beta_{window}d'] * data['benchmark_return'])
            
            # Calculate idiosyncratic volatility (std of alpha)
            data[f'idio_vol_{window}d'] = grouped[f'alpha_{window}d'].apply(
                lambda s: s.rolling(window).std() * np.sqrt(252)
            )
            
            # Calculate R-squared (explanatory power of market)
            def rolling_r_squared(group_data):
                """Calculate rolling R-squared"""
                result = pd.Series(index=group_data.index, dtype=float)
                
                for i in range(len(group_data)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < min_obs:
                        continue
                    
                    y = group_data['log_return'].iloc[start_idx:end_idx].dropna()
                    x = group_data['benchmark_return'].iloc[start_idx:end_idx].dropna()
                    
                    aligned_data = pd.DataFrame({'y': y, 'x': x}).dropna()
                    
                    if len(aligned_data) < min_obs:
                        continue
                    
                    correlation = aligned_data['x'].corr(aligned_data['y'])
                    if not np.isnan(correlation):
                        result.iloc[i] = correlation ** 2
                
                return result
            
            data[f'r_squared_{window}d'] = grouped.apply(rolling_r_squared).values
        
        return data
    
    def _generate_cross_sectional_rankings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectional rankings (Section 2.2)"""
        self.logger.info("Generating cross-sectional rankings")
        
        # Ensure we have required features
        available_features = [f for f in self.params['ranking_features'] if f in data.columns]
        if not available_features:
            self.logger.warning("No ranking features available")
            return data
        
        # Group by date for cross-sectional ranking
        date_groups = data.groupby(data.index.date) if isinstance(data.index, pd.DatetimeIndex) else data.groupby(level=0)
        
        for feature in available_features:
            if feature not in data.columns:
                continue
                
            self.logger.debug(f"Computing cross-sectional rank for {feature}")
            
            if self.params['ranking_method'] == 'percent_rank':
                # Percentile ranking (0-1)
                data[f'{feature}_rank'] = date_groups[feature].apply(
                    lambda x: x.rank(pct=True, na_option='keep')
                ).values
            else:
                # Simple ranking
                data[f'{feature}_rank'] = date_groups[feature].apply(
                    lambda x: x.rank(na_option='keep')
                ).values
            
            # Z-score ranking (normalized)
            data[f'{feature}_zscore'] = date_groups[feature].apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            ).values
            
            # Decile rankings (1-10)
            data[f'{feature}_decile'] = date_groups[feature].apply(
                lambda x: pd.qcut(x.dropna(), q=10, labels=range(1, 11), duplicates='drop')
            ).values
        
        # Generate composite momentum and quality rankings
        momentum_features = [f for f in available_features if 'return' in f]
        quality_features = [f for f in available_features if 'vol_cc' in f or 'alpha' in f]
        
        if momentum_features:
            # Momentum composite (average of return rankings)
            momentum_ranks = [f'{f}_rank' for f in momentum_features if f'{f}_rank' in data.columns]
            if momentum_ranks:
                data['momentum_composite_rank'] = data[momentum_ranks].mean(axis=1, skipna=True)
        
        if quality_features:
            # Quality composite (inverse of volatility, higher alpha)
            quality_ranks = []
            for f in quality_features:
                if f'{f}_rank' in data.columns:
                    if 'vol' in f:
                        # Invert volatility rank (lower vol = higher quality)
                        quality_ranks.append(1.0 - data[f'{f}_rank'])
                    else:
                        quality_ranks.append(data[f'{f}_rank'])
            
            if quality_ranks:
                data['quality_composite_rank'] = np.column_stack(quality_ranks).mean(axis=1)
        
        return data
    
    def _generate_universe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate universe membership and filtering features"""
        self.logger.info("Generating universe membership features")
        
        # Calculate dollar volume
        if 'volume' in data.columns and 'close_adj' in data.columns:
            data['dollar_volume'] = data['volume'] * data['close_adj']
        
        # Universe membership flags
        data['universe_price_filter'] = (data['close_adj'] >= self.params['min_price'])
        
        if 'dollar_volume' in data.columns:
            data['universe_volume_filter'] = (data['dollar_volume'] >= self.params['min_volume'])
        else:
            data['universe_volume_filter'] = True
        
        # Trading activity filter (sufficient trading days)
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Count non-null trading days in rolling window
        data['trading_days_21d'] = grouped['close_adj'].apply(
            lambda s: s.rolling(21).count()
        )
        
        data['universe_activity_filter'] = (data['trading_days_21d'] >= self.params['min_trading_days'])
        
        # Combined universe filter
        data['in_universe'] = (data['universe_price_filter'] & 
                              data['universe_volume_filter'] & 
                              data['universe_activity_filter'])
        
        # Universe size (number of instruments in universe each day)
        if 'instrument' in data.columns:
            date_groups = data.groupby(data.index.date) if isinstance(data.index, pd.DatetimeIndex) else data.groupby(level=0)
            data['universe_size'] = date_groups['in_universe'].transform('sum')
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this stage creates"""
        features = []
        
        # Beta/Alpha features
        for window in self.params['beta_windows']:
            features.extend([
                f'beta_{window}d',
                f'alpha_{window}d', 
                f'idio_vol_{window}d',
                f'r_squared_{window}d'
            ])
        
        # Cross-sectional ranking features
        for feature in self.params['ranking_features']:
            features.extend([
                f'{feature}_rank',
                f'{feature}_zscore',
                f'{feature}_decile'
            ])
        
        # Composite rankings
        features.extend([
            'momentum_composite_rank',
            'quality_composite_rank'
        ])
        
        # Universe features
        features.extend([
            'dollar_volume',
            'universe_price_filter',
            'universe_volume_filter', 
            'universe_activity_filter',
            'trading_days_21d',
            'in_universe',
            'universe_size'
        ])
        
        return features