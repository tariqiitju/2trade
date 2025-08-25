"""
Stage 3: Regimes & Seasonal Structure for Da Vinchi Feature Engineering Pipeline

This stage implements regime detection and seasonality features as specified in the plan-draft:
- Market regime identification using clustering (k-means, HMM)
- Seasonal/cyclical time encodings (day-of-week, month, turn-of-month)
- Event windows and time-since-event features
- Volatility and trend regime classification

Uses contemporaneous features only to avoid look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

from .stage_base import FeatureStage, StageData, StageMetadata

logger = logging.getLogger(__name__)


class Stage3RegimesSeasonal(FeatureStage):
    """
    Stage 3: Regimes & Seasonal Structure
    
    Identifies market regimes and seasonal patterns that provide important context
    for feature engineering and model training. Includes regime clustering,
    cyclical time features, and event-based features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Extract stage-specific config
        stage_config = config.get('stages', {}).get('stage3_regimes', {})
        
        # Regime detection parameters with model selection
        self.params = {
            # Model selection for regime detection
            'available_models': {
                'kmeans': {'class': 'KMeans', 'library': 'sklearn'},
                'gmm': {'class': 'GaussianMixture', 'library': 'sklearn'}, 
                'hmm': {'class': 'GaussianHMM', 'library': 'hmmlearn'},
                'dbscan': {'class': 'DBSCAN', 'library': 'sklearn'}
            },
            
            # Regime clustering
            'regime_method': stage_config.get('regime_method', 'kmeans'),
            'n_regimes': stage_config.get('n_regimes', 4),
            'regime_features': stage_config.get('regime_features', [
                'vol_cc_20d', 'return_21d', 'dollar_volume'
            ]),
            'majority_filter_days': stage_config.get('majority_filter_days', 3),
            'regime_windows': stage_config.get('regime_windows', [60, 126]),
            
            # Seasonality features
            'cyclical_features': stage_config.get('cyclical_features', [
                'day_of_week', 'month_of_year', 'quarter'
            ]),
            'turn_of_month_window': stage_config.get('turn_of_month_window', 3),
            'turn_of_quarter_window': stage_config.get('turn_of_quarter_window', 5),
            'holiday_effects': stage_config.get('holiday_effects', True),
            
            # Event features
            'earnings_window': stage_config.get('earnings_window', 3),  # Days before/after earnings
            'dividend_window': stage_config.get('dividend_window', 2),
            'max_days_since_event': stage_config.get('max_days_since_event', 90),
            
            # Volatility regime thresholds
            'vol_regime_quantiles': stage_config.get('vol_regime_quantiles', [0.25, 0.5, 0.75]),
            'trend_regime_threshold': stage_config.get('trend_regime_threshold', 0.02)  # 2% threshold
        }
        
        # Initialize model selector
        self.model_selector = self._initialize_model_selector()
        
        # Now call parent __init__ after params are set
        super().__init__(config, "Stage3_RegimesSeasonal", "1.0.0")
    
    def _initialize_model_selector(self) -> Dict[str, Any]:
        """Initialize clustering model based on configuration"""
        method = self.params['regime_method']
        available_models = self.params['available_models']
        
        if method not in available_models:
            self.logger.warning(f"Unknown regime method {method}, using kmeans")
            method = 'kmeans'
        
        model_info = available_models[method]
        
        try:
            if method == 'kmeans':
                from sklearn.cluster import KMeans
                model = KMeans(
                    n_clusters=self.params['n_regimes'],
                    random_state=42,
                    n_init=10
                )
            elif method == 'gmm':
                from sklearn.mixture import GaussianMixture
                model = GaussianMixture(
                    n_components=self.params['n_regimes'],
                    random_state=42
                )
            elif method == 'hmm':
                try:
                    from hmmlearn.hmm import GaussianHMM
                    model = GaussianHMM(
                        n_components=self.params['n_regimes'],
                        random_state=42
                    )
                except ImportError:
                    self.logger.warning("hmmlearn not available, falling back to kmeans")
                    from sklearn.cluster import KMeans
                    model = KMeans(n_clusters=self.params['n_regimes'], random_state=42)
            elif method == 'dbscan':
                from sklearn.cluster import DBSCAN
                model = DBSCAN(eps=0.5, min_samples=5)
            else:
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=self.params['n_regimes'], random_state=42)
            
            return {
                'model': model,
                'method': method,
                'available': True
            }
        
        except ImportError as e:
            self.logger.error(f"Failed to import {method}: {e}")
            # Fallback to basic implementation
            return {
                'model': None,
                'method': 'basic',
                'available': False
            }
    
    def _get_required_columns(self) -> List[str]:
        """Required columns from previous stages"""
        base_cols = ['close_adj', 'log_return']
        # Add regime features if they exist
        for feature in self.params['regime_features']:
            if feature not in base_cols:
                base_cols.append(feature)
        return base_cols
    
    def process(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 3: Regimes & Seasonal Structure
        
        Args:
            input_data: StageData from previous stages
            
        Returns:
            StageData with regime and seasonal features added
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        self.logger.info(f"Generating regime and seasonal features for {data.shape[0]} observations")
        
        # Generate regime features
        data = self._generate_regime_features(data)
        
        # Generate seasonal features
        data = self._generate_seasonal_features(data)
        
        # Generate event-based features
        data = self._generate_event_features(data)
        
        # Generate volatility regime features
        data = self._generate_volatility_regimes(data)
        
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
        
        self.logger.info(f"Created {len(new_features)} regime and seasonal features")
        
        return StageData(
            data=data,
            metadata=result_metadata,
            config=input_data.config,
            artifacts={
                **input_data.artifacts,
                'stage3_features': new_features,
                'stage3_params': self.params
            }
        )
    
    def _generate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime identification features (Section 3.1)"""
        self.logger.info("Generating market regime features")
        
        # Prepare features for regime detection
        regime_features = []
        feature_matrix_cols = []
        
        for feature in self.params['regime_features']:
            if feature in data.columns:
                regime_features.append(feature)
                feature_matrix_cols.append(feature)
            elif feature == 'dollar_volume' and 'volume' in data.columns and 'close_adj' in data.columns:
                data['dollar_volume'] = data['volume'] * data['close_adj']
                regime_features.append('dollar_volume')
                feature_matrix_cols.append('dollar_volume')
        
        if not regime_features:
            self.logger.warning("No regime features available for clustering")
            return data
        
        # Group by instrument if column exists
        if 'instrument' in data.columns:
            instruments = data['instrument'].unique()
        else:
            instruments = ['single']
            data['instrument'] = 'single'
        
        # Detect regimes for each instrument
        for instrument in instruments:
            if instrument != 'single':
                inst_data = data[data['instrument'] == instrument].copy()
            else:
                inst_data = data.copy()
            
            if len(inst_data) < 50:  # Skip if insufficient data
                continue
            
            # Create feature matrix for clustering
            feature_matrix = inst_data[regime_features].copy()
            
            # Handle missing values
            feature_matrix = feature_matrix.ffill().bfill()
            
            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix.dropna())
            
            if len(feature_matrix_scaled) < self.params['n_regimes']:
                continue
            
            # Apply clustering using selected model
            if self.model_selector['available']:
                model = self.model_selector['model']
                method = self.model_selector['method']
                
                if method == 'hmm':
                    # HMM requires fitting and prediction separately
                    model.fit(feature_matrix_scaled)
                    regime_labels = model.predict(feature_matrix_scaled)
                elif method == 'gmm':
                    # GMM uses fit_predict
                    regime_labels = model.fit_predict(feature_matrix_scaled)
                else:
                    # KMeans, DBSCAN use fit_predict
                    regime_labels = model.fit_predict(feature_matrix_scaled)
            else:
                # Fallback to simple quantile-based regime detection
                self.logger.warning("Using fallback quantile-based regime detection")
                regime_labels = self._simple_regime_detection(feature_matrix_scaled)
            
            # Create regime labels series
            regime_series = pd.Series(
                regime_labels,
                index=feature_matrix.dropna().index,
                name='regime_raw'
            )
            
            # Apply majority filter to smooth regime transitions
            regime_filtered = self._apply_majority_filter(
                regime_series, 
                window=self.params['majority_filter_days']
            )
            
            # Assign back to main dataframe
            if instrument != 'single':
                data.loc[data['instrument'] == instrument, 'regime_id'] = regime_filtered
            else:
                data['regime_id'] = regime_filtered
        
        # Create regime indicator variables
        if 'regime_id' in data.columns:
            for regime_id in range(self.params['n_regimes']):
                data[f'regime_{regime_id}'] = (data['regime_id'] == regime_id).astype(int)
            
            # Regime persistence (days since regime change)
            if 'instrument' in data.columns:
                grouped = data.groupby('instrument', group_keys=False)
            else:
                grouped = data.groupby(lambda x: 'single', group_keys=False)
            
            data['regime_persistence'] = grouped['regime_id'].apply(
                lambda s: (s != s.shift(1)).cumsum().groupby(s).cumcount() + 1
            )
        
        return data
    
    def _apply_majority_filter(self, regime_series: pd.Series, window: int) -> pd.Series:
        """Apply majority filter to smooth regime transitions"""
        if window <= 1:
            return regime_series
        
        filtered_series = regime_series.copy()
        
        for i in range(len(regime_series)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(regime_series), i + window // 2 + 1)
            
            window_values = regime_series.iloc[start_idx:end_idx]
            if len(window_values) > 0:
                # Use mode (most frequent value) in the window
                mode_value = window_values.mode()
                if len(mode_value) > 0:
                    filtered_series.iloc[i] = mode_value.iloc[0]
        
        return filtered_series
    
    def _simple_regime_detection(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Fallback regime detection using quantiles"""
        if len(feature_matrix.shape) != 2 or feature_matrix.shape[1] == 0:
            return np.zeros(len(feature_matrix), dtype=int)
        
        # Use the first feature (typically volatility) for simple regime detection
        primary_feature = feature_matrix[:, 0]
        
        # Create quantile-based regimes
        n_regimes = self.params['n_regimes']
        quantiles = np.linspace(0, 1, n_regimes + 1)
        regime_labels = np.digitize(primary_feature, 
                                  np.quantile(primary_feature, quantiles[1:-1])) - 1
        
        # Ensure labels are in valid range
        regime_labels = np.clip(regime_labels, 0, n_regimes - 1)
        
        return regime_labels
    
    def _generate_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate seasonal and cyclical features (Section 3.2)"""
        self.logger.info("Generating seasonal features")
        
        # Ensure we have datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        # Day of week features (cyclical encoding)
        if 'day_of_week' in self.params['cyclical_features']:
            data['day_of_week'] = data.index.dayofweek
            data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            
            # Monday effect
            data['is_monday'] = (data['day_of_week'] == 0).astype(int)
            data['is_friday'] = (data['day_of_week'] == 4).astype(int)
        
        # Month of year features (cyclical encoding)
        if 'month_of_year' in self.params['cyclical_features']:
            data['month_of_year'] = data.index.month
            data['month_sin'] = np.sin(2 * np.pi * data['month_of_year'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month_of_year'] / 12)
            
            # January effect
            data['is_january'] = (data['month_of_year'] == 1).astype(int)
        
        # Quarter features
        if 'quarter' in self.params['cyclical_features']:
            data['quarter'] = data.index.quarter
            data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
            data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)
        
        # Turn of month effect
        if self.params['turn_of_month_window'] > 0:
            data['turn_of_month'] = self._get_turn_of_month_flag(
                data.index, self.params['turn_of_month_window']
            )
        
        # Turn of quarter effect
        if self.params['turn_of_quarter_window'] > 0:
            data['turn_of_quarter'] = self._get_turn_of_quarter_flag(
                data.index, self.params['turn_of_quarter_window']
            )
        
        # Holiday effects (basic US market holidays)
        if self.params['holiday_effects']:
            data = self._add_holiday_effects(data)
        
        return data
    
    def _get_turn_of_month_flag(self, date_index: pd.DatetimeIndex, window: int) -> pd.Series:
        """Create turn-of-month indicator (last N and first N trading days)"""
        flags = pd.Series(0, index=date_index, dtype=int)
        
        # Group by month
        months = date_index.to_series().groupby([date_index.year, date_index.month])
        
        for (year, month), month_dates in months:
            month_dates_sorted = month_dates.sort_values()
            n_days = len(month_dates_sorted)
            
            if n_days >= 2 * window:
                # First N days
                first_n = month_dates_sorted.iloc[:window]
                flags[first_n.index] = 1
                
                # Last N days
                last_n = month_dates_sorted.iloc[-window:]
                flags[last_n.index] = 1
        
        return flags
    
    def _get_turn_of_quarter_flag(self, date_index: pd.DatetimeIndex, window: int) -> pd.Series:
        """Create turn-of-quarter indicator"""
        flags = pd.Series(0, index=date_index, dtype=int)
        
        # Group by quarter
        quarters = date_index.to_series().groupby([date_index.year, date_index.quarter])
        
        for (year, quarter), quarter_dates in quarters:
            quarter_dates_sorted = quarter_dates.sort_values()
            n_days = len(quarter_dates_sorted)
            
            if n_days >= 2 * window:
                # First N days
                first_n = quarter_dates_sorted.iloc[:window]
                flags[first_n.index] = 1
                
                # Last N days
                last_n = quarter_dates_sorted.iloc[-window:]
                flags[last_n.index] = 1
        
        return flags
    
    def _add_holiday_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic US market holiday effects"""
        # Pre/post holiday effects (simplified)
        dates = data.index.to_series()
        
        # New Year effect (first few trading days of January)
        data['new_year_effect'] = (
            (dates.dt.month == 1) & (dates.dt.day <= 10)
        ).astype(int)
        
        # Year-end effect (last few trading days of December)
        data['year_end_effect'] = (
            (dates.dt.month == 12) & (dates.dt.day >= 20)
        ).astype(int)
        
        # Quarter end effect (last day of quarter)
        quarter_ends = dates.dt.is_quarter_end
        data['quarter_end'] = quarter_ends.astype(int)
        
        # Month end effect (last trading day of month)
        month_ends = dates.dt.is_month_end
        data['month_end'] = month_ends.astype(int)
        
        return data
    
    def _generate_event_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate event-based features (earnings, dividends, etc.)"""
        self.logger.info("Generating event features")
        
        # Since we don't have access to earnings/dividend data in this basic implementation,
        # we create placeholder logic that could be extended with real event data
        
        # Time-based event proxies
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Identify potential earnings periods (simplified - every quarter around same time)
        # This would be replaced with actual earnings calendar data
        data['potential_earnings_period'] = (
            (data.index.month.isin([1, 4, 7, 10])) & 
            (data.index.day.isin(range(15, 30)))
        ).astype(int)
        
        # High volatility events (proxy for earnings/news)
        if 'vol_cc_20d' in data.columns:
            vol_threshold = data['vol_cc_20d'].quantile(0.9)
            data['high_vol_event'] = (data['vol_cc_20d'] > vol_threshold).astype(int)
            
            # Days since high vol event
            data['days_since_high_vol'] = grouped['high_vol_event'].apply(
                lambda s: self._days_since_event(s)
            ).fillna(self.params['max_days_since_event'])
        
        # High volume events (proxy for news/events)
        if 'dollar_volume' in data.columns:
            volume_threshold = grouped['dollar_volume'].apply(
                lambda s: s.rolling(20).quantile(0.9)
            )
            data['high_volume_event'] = (data['dollar_volume'] > volume_threshold).astype(int)
            
            data['days_since_high_volume'] = grouped['high_volume_event'].apply(
                lambda s: self._days_since_event(s)
            ).fillna(self.params['max_days_since_event'])
        
        return data
    
    def _days_since_event(self, event_series: pd.Series) -> pd.Series:
        """Calculate days since last event"""
        result = pd.Series(index=event_series.index, dtype=float)
        last_event_idx = None
        
        for i, (idx, value) in enumerate(event_series.items()):
            if value == 1:
                last_event_idx = i
                result.loc[idx] = 0
            elif last_event_idx is not None:
                days_since = i - last_event_idx
                if days_since <= self.params['max_days_since_event']:
                    result.loc[idx] = days_since
                else:
                    result.loc[idx] = self.params['max_days_since_event']
            else:
                result.loc[idx] = self.params['max_days_since_event']
        
        return result
    
    def _generate_volatility_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime classifications"""
        self.logger.info("Generating volatility regime features")
        
        if 'vol_cc_20d' not in data.columns:
            return data
        
        # Group by instrument if column exists
        if 'instrument' in data.columns:
            grouped = data.groupby('instrument', group_keys=False)
        else:
            grouped = data.groupby(lambda x: 'single', group_keys=False)
        
        # Rolling volatility quantiles
        for window in [60, 126, 252]:
            vol_quantiles = grouped['vol_cc_20d'].apply(
                lambda s: s.rolling(window).rank(pct=True)
            )
            data[f'vol_regime_{window}d'] = vol_quantiles
            
            # Regime classification (low/med/high)
            data[f'vol_regime_{window}d_class'] = pd.cut(
                vol_quantiles,
                bins=[0, 0.33, 0.66, 1.0],
                labels=['low_vol', 'med_vol', 'high_vol'],
                include_lowest=True
            )
        
        # Volatility regime transitions
        vol_regime_col = f'vol_regime_{self.params["regime_windows"][0]}d_class'
        if vol_regime_col in data.columns:
            data['vol_regime_changed'] = grouped[vol_regime_col].apply(
                lambda s: (s != s.shift(1)).astype(int)
            )
        
        # Trend regime (based on return momentum)
        if 'return_21d' in data.columns:
            trend_threshold = self.params['trend_regime_threshold']
            data['trend_regime'] = pd.cut(
                data['return_21d'],
                bins=[-np.inf, -trend_threshold, trend_threshold, np.inf],
                labels=['downtrend', 'sideways', 'uptrend']
            )
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this stage creates"""
        features = []
        
        # Regime features
        for regime_id in range(self.params['n_regimes']):
            features.append(f'regime_{regime_id}')
        
        features.extend([
            'regime_id',
            'regime_persistence'
        ])
        
        # Seasonal features
        if 'day_of_week' in self.params['cyclical_features']:
            features.extend([
                'day_of_week', 'dow_sin', 'dow_cos', 'is_monday', 'is_friday'
            ])
        
        if 'month_of_year' in self.params['cyclical_features']:
            features.extend([
                'month_of_year', 'month_sin', 'month_cos', 'is_january'
            ])
        
        if 'quarter' in self.params['cyclical_features']:
            features.extend([
                'quarter', 'quarter_sin', 'quarter_cos'
            ])
        
        features.extend([
            'turn_of_month', 'turn_of_quarter', 'new_year_effect', 
            'year_end_effect', 'quarter_end', 'month_end'
        ])
        
        # Event features
        features.extend([
            'potential_earnings_period', 'high_vol_event', 'days_since_high_vol',
            'high_volume_event', 'days_since_high_volume'
        ])
        
        # Volatility regime features
        for window in [60, 126, 252]:
            features.extend([
                f'vol_regime_{window}d',
                f'vol_regime_{window}d_class'
            ])
        
        features.extend([
            'vol_regime_changed', 'trend_regime'
        ])
        
        return features