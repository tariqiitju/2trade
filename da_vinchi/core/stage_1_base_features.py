"""
Stage 1: Base Feature Generator for Da Vinchi Feature Engineering Pipeline

This stage implements baseline OHLCV features as specified in the plan-draft:
- Returns & volatility (simple, log, rolling, Parkinson, Garman-Klass)
- Trend/momentum (SMA/EMA, MACD, RSI, Stochastic, ADX)
- Range, bands, volatility-of-price (ATR, Bollinger Bands)
- Liquidity & microstructure (Dollar Volume, Amihud, Roll Spread)

All features are computed per instrument with proper handling of time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from .stage_base import FeatureStage, StageData, StageMetadata

logger = logging.getLogger(__name__)


class Stage1BaseFeatures(FeatureStage):
    """
    Stage 1: Base Feature Generator
    
    Generates fundamental OHLCV-based features that form the foundation
    for more advanced feature engineering stages. Includes returns, volatility,
    momentum, trend, and microstructure features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "Stage1_BaseFeatures", "1.0.0")
        
        # Feature computation parameters
        self.params = {
            # Returns & Volatility
            'volatility_windows': config.get('volatility_windows', [5, 10, 20, 60]),
            'return_windows': config.get('return_windows', [5, 10, 20, 60, 126, 252]),
            
            # Trend/Momentum
            'ema_spans': config.get('ema_spans', [12, 26, 50, 200]),
            'sma_windows': config.get('sma_windows', [10, 20, 50, 200]),
            'rsi_period': config.get('rsi_period', 14),
            'macd_fast': config.get('macd_fast', 12),
            'macd_slow': config.get('macd_slow', 26),
            'macd_signal': config.get('macd_signal', 9),
            'stoch_k_period': config.get('stoch_k_period', 14),
            'stoch_d_period': config.get('stoch_d_period', 3),
            'adx_period': config.get('adx_period', 14),
            
            # Range & Bands
            'atr_period': config.get('atr_period', 14),
            'bollinger_period': config.get('bollinger_period', 20),
            'bollinger_std': config.get('bollinger_std', 2),
            
            # Liquidity
            'amihud_windows': config.get('amihud_windows', [5, 20, 60]),
            'roll_spread_window': config.get('roll_spread_window', 20)
        }
    
    def _get_required_columns(self) -> List[str]:
        """Required columns from Stage 0"""
        return ['open', 'high', 'low', 'close', 'volume', 'close_adj']
    
    def process(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 1: Base Features
        
        Args:
            input_data: StageData from Stage 0 with validated OHLCV data
            
        Returns:
            StageData with base features added
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        self.logger.info(f"Generating base features for {data.shape[0]} observations")
        
        # Generate features by category
        data = self._generate_returns_volatility(data)
        data = self._generate_trend_momentum(data)
        data = self._generate_range_bands(data)
        data = self._generate_liquidity_features(data)
        
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
        
        self.logger.info(f"Created {len(new_features)} base features")
        
        return StageData(
            data=data,
            metadata=result_metadata,
            config=input_data.config,
            artifacts={
                **input_data.artifacts,
                'stage1_features': new_features,
                'stage1_params': self.params
            }
        )
    
    def _generate_returns_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate returns and volatility features (Section 1.1)"""
        self.logger.info("Generating returns and volatility features")
        
        # Group by instrument if column exists
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Simple and log returns
        data['simple_return'] = grouped['close_adj'].apply(lambda s: s.pct_change())
        data['log_return'] = grouped['close_adj'].apply(lambda s: np.log(s / s.shift(1)))
        
        # Rolling returns for multiple windows
        for window in self.params['return_windows']:
            data[f'return_{window}d'] = grouped['log_return'].apply(
                lambda s: s.rolling(window).sum()
            )
        
        # Realized volatility (close-to-close)
        for window in self.params['volatility_windows']:
            data[f'vol_cc_{window}d'] = grouped['log_return'].apply(
                lambda s: s.rolling(window).std() * np.sqrt(252)  # Annualized
            )
        
        # Parkinson estimator (high-low volatility)
        data['log_hl_ratio'] = np.log(data['high'] / data['low'])
        for window in self.params['volatility_windows']:
            data[f'vol_parkinson_{window}d'] = grouped['log_hl_ratio'].apply(
                lambda s: np.sqrt(s.rolling(window).apply(
                    lambda x: (x**2).mean() / (4 * np.log(2))
                )) * np.sqrt(252)
            )
        
        # Garman-Klass estimator
        data['log_oc_ratio'] = np.log(data['close'] / data['open'])
        data['gk_component'] = 0.5 * data['log_hl_ratio']**2 - (2 * np.log(2) - 1) * data['log_oc_ratio']**2
        
        for window in self.params['volatility_windows']:
            data[f'vol_gk_{window}d'] = grouped['gk_component'].apply(
                lambda s: np.sqrt(s.rolling(window).mean()) * np.sqrt(252)
            )
        
        return data
    
    def _generate_trend_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend and momentum features (Section 1.2)"""
        self.logger.info("Generating trend and momentum features")
        
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Simple Moving Averages (SMA)
        for window in self.params['sma_windows']:
            data[f'sma_{window}'] = grouped['close_adj'].apply(
                lambda s: s.rolling(window).mean()
            )
        
        # Exponential Moving Averages (EMA)
        for span in self.params['ema_spans']:
            data[f'ema_{span}'] = grouped['close_adj'].apply(
                lambda s: s.ewm(span=span, adjust=False).mean()
            )
        
        # MACD
        ema_fast = grouped['close_adj'].apply(
            lambda s: s.ewm(span=self.params['macd_fast'], adjust=False).mean()
        )
        ema_slow = grouped['close_adj'].apply(
            lambda s: s.ewm(span=self.params['macd_slow'], adjust=False).mean()
        )
        data['macd'] = ema_fast - ema_slow
        
        # MACD Signal Line
        if 'instrument' in data.columns:
            macd_grouped = data.groupby('instrument', group_keys=False)['macd']
        else:
            macd_grouped = data.groupby(lambda x: 'single', group_keys=False)['macd']
            
        data['macd_signal'] = macd_grouped.apply(
            lambda s: s.ewm(span=self.params['macd_signal'], adjust=False).mean()
        )
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI (Relative Strength Index)
        data['rsi'] = grouped['close_adj'].apply(
            lambda s: self._calculate_rsi(s, self.params['rsi_period'])
        )
        
        # Stochastic Oscillator
        stoch_k_values = self._calculate_stochastic_k(data, self.params['stoch_k_period'])
        data['stoch_k'] = stoch_k_values
        
        if 'instrument' in data.columns:
            stoch_grouped = data.groupby('instrument', group_keys=False)['stoch_k']
        else:
            stoch_grouped = data.groupby(lambda x: 'single', group_keys=False)['stoch_k']
            
        data['stoch_d'] = stoch_grouped.apply(
            lambda s: s.rolling(self.params['stoch_d_period']).mean()
        )
        
        # ADX (Average Directional Index) - calculate after ensuring true_range exists
        if 'true_range' not in data.columns:
            # Calculate true range here if not already calculated
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - grouped['close_adj'].shift(1))
            low_close_prev = abs(data['low'] - grouped['close_adj'].shift(1))
            data['true_range'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        data = self._calculate_adx(data, self.params['adx_period'])
        
        return data
    
    def _generate_range_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate range and band features (Section 1.3)"""
        self.logger.info("Generating range and band features")
        
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # True Range
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - grouped['close_adj'].shift(1))
        low_close_prev = abs(data['low'] - grouped['close_adj'].shift(1))
        
        data['true_range'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Average True Range (ATR) - Wilder's smoothing
        data['atr'] = grouped['true_range'].apply(
            lambda s: s.ewm(alpha=1/self.params['atr_period'], adjust=False).mean()
        )
        
        # Bollinger Bands
        period = self.params['bollinger_period']
        std_mult = self.params['bollinger_std']
        
        bb_middle = grouped['close_adj'].apply(lambda s: s.rolling(period).mean())
        bb_std = grouped['close_adj'].apply(lambda s: s.rolling(period).std())
        
        data['bb_middle'] = bb_middle
        data['bb_upper'] = bb_middle + (std_mult * bb_std)
        data['bb_lower'] = bb_middle - (std_mult * bb_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_percent'] = (data['close_adj'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        return data
    
    def _generate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate liquidity and microstructure features (Section 1.4)"""
        self.logger.info("Generating liquidity and microstructure features")
        
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Dollar Volume
        data['dollar_volume'] = data['close_adj'] * data['volume']
        
        # Volume-based features
        for window in [5, 10, 20, 60]:
            data[f'volume_sma_{window}'] = grouped['volume'].apply(
                lambda s: s.rolling(window).mean()
            )
            data[f'volume_ratio_{window}'] = data['volume'] / data[f'volume_sma_{window}']
            
            # Dollar volume rolling averages
            data[f'dollar_vol_sma_{window}'] = grouped['dollar_volume'].apply(
                lambda s: s.rolling(window).mean()
            )
        
        # Amihud Illiquidity measure
        for window in self.params['amihud_windows']:
            data['abs_return'] = abs(data['log_return'])
            data['illiq_daily'] = data['abs_return'] / (data['dollar_volume'] + 1e-10)  # Avoid division by zero
            data[f'amihud_{window}d'] = grouped['illiq_daily'].apply(
                lambda s: s.rolling(window).mean()
            )
        
        # Roll Spread (if no bid-ask data available)
        window = self.params['roll_spread_window']
        data['price_changes'] = grouped['close_adj'].apply(lambda s: s.diff())
        
        # Calculate covariance between consecutive price changes
        data['cov_changes'] = grouped['price_changes'].apply(
            lambda s: s.rolling(window).apply(
                lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan
            )
        )
        
        # Roll spread estimate (only when covariance is negative)
        data['roll_spread'] = np.where(
            data['cov_changes'] < 0, 
            2 * np.sqrt(-data['cov_changes']), 
            np.nan
        )
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Wilder's smoothing"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Wilder's smoothing (exponential with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        if 'instrument' in data.columns:
            # Calculate per instrument
            stoch_values = []
            for instrument in data['instrument'].unique():
                mask = data['instrument'] == instrument
                instrument_data = data[mask].copy()
                
                high_roll = instrument_data['high'].rolling(period).max()
                low_roll = instrument_data['low'].rolling(period).min()
                stoch_k = 100 * (instrument_data['close_adj'] - low_roll) / (high_roll - low_roll + 1e-10)
                stoch_values.append(stoch_k)
            
            return pd.concat(stoch_values).sort_index()
        else:
            # Single instrument
            high_roll = data['high'].rolling(period).max()
            low_roll = data['low'].rolling(period).min()
            return 100 * (data['close_adj'] - low_roll) / (high_roll - low_roll + 1e-10)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # True Range already calculated
        tr = data['true_range']
        
        # Directional Movement
        high_diff = grouped['high'].apply(lambda s: s.diff())
        low_diff = grouped['low'].apply(lambda s: -s.diff())
        
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Smoothed True Range and Directional Movement (Wilder's smoothing)
        if 'instrument' in data.columns:
            tr_grouped = data.groupby('instrument', group_keys=False)['true_range']
            pos_dm_series = pd.Series(pos_dm, index=data.index)
            neg_dm_series = pd.Series(neg_dm, index=data.index)
        else:
            tr_grouped = data.groupby(lambda x: 'single', group_keys=False)['true_range']
            pos_dm_series = pd.Series(pos_dm, index=data.index)
            neg_dm_series = pd.Series(neg_dm, index=data.index)
        
        tr_smooth = tr_grouped.apply(lambda s: s.ewm(alpha=1/period, adjust=False).mean())
        
        if 'instrument' in data.columns:
            pos_dm_grouped = pos_dm_series.groupby(data['instrument'], group_keys=False)
            neg_dm_grouped = neg_dm_series.groupby(data['instrument'], group_keys=False)
        else:
            pos_dm_grouped = pos_dm_series.groupby(lambda x: 'single', group_keys=False)
            neg_dm_grouped = neg_dm_series.groupby(lambda x: 'single', group_keys=False)
        
        pos_dm_smooth = pos_dm_grouped.apply(lambda s: s.ewm(alpha=1/period, adjust=False).mean())
        neg_dm_smooth = neg_dm_grouped.apply(lambda s: s.ewm(alpha=1/period, adjust=False).mean())
        
        # Directional Indicators
        data['plus_di'] = 100 * pos_dm_smooth / (tr_smooth + 1e-10)
        data['minus_di'] = 100 * neg_dm_smooth / (tr_smooth + 1e-10)
        
        # ADX calculation
        dx = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'] + 1e-10)
        
        if 'instrument' in data.columns:
            dx_grouped = dx.groupby(data['instrument'], group_keys=False)
        else:
            dx_grouped = dx.groupby(lambda x: 'single', group_keys=False)
            
        data['adx'] = dx_grouped.apply(lambda s: s.ewm(alpha=1/period, adjust=False).mean())
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names that will be created by Stage 1"""
        features = []
        
        # Returns & Volatility
        features.extend(['simple_return', 'log_return'])
        
        for window in self.params['return_windows']:
            features.append(f'return_{window}d')
            
        for window in self.params['volatility_windows']:
            features.extend([
                f'vol_cc_{window}d',
                f'vol_parkinson_{window}d', 
                f'vol_gk_{window}d'
            ])
        
        # Trend/Momentum
        for window in self.params['sma_windows']:
            features.append(f'sma_{window}')
            
        for span in self.params['ema_spans']:
            features.append(f'ema_{span}')
            
        features.extend([
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'stoch_k', 'stoch_d',
            'plus_di', 'minus_di', 'adx'
        ])
        
        # Range & Bands
        features.extend([
            'true_range', 'atr',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent'
        ])
        
        # Liquidity
        features.append('dollar_volume')
        
        for window in [5, 10, 20, 60]:
            features.extend([
                f'volume_sma_{window}',
                f'volume_ratio_{window}',
                f'dollar_vol_sma_{window}'
            ])
        
        for window in self.params['amihud_windows']:
            features.append(f'amihud_{window}d')
            
        features.append('roll_spread')
        
        return features