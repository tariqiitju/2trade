#!/usr/bin/env python3
"""
Stage 5: Target Instrument Feature Generation

This stage generates features for a specific target instrument using information
from strongly correlated peer instruments. It leverages relationship analysis 
from Stage 4 to identify relevant peers and creates predictive features.

Key Features:
- Target-specific feature generation
- Peer instrument selection based on correlation/cointegration
- Cross-instrument momentum and mean reversion features
- Relative performance and spread features
- Memory caching for efficiency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import warnings

from .stage_base import StageBase, StageData, StageMetadata


class Stage5TargetFeatures(StageBase):
    """
    Stage 5: Target Instrument Feature Generation
    
    This stage creates features for a target instrument by analyzing its relationships
    with peer instruments. It uses correlation, cointegration, and lead-lag relationships
    to generate predictive features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "Stage5_TargetFeatures", "1.0.0")
        
        # Extract stage-specific config
        stage_config = config.get('stages', {}).get('stage5_target_features', {})
        
        self.params = {
            # Target specification
            'target_instrument': stage_config.get('target_instrument', None),
            'peer_instruments': stage_config.get('peer_instruments', []),
            
            # Peer selection criteria
            'min_correlation': stage_config.get('min_correlation', 0.3),
            'max_peers': stage_config.get('max_peers', 10),
            'use_correlation': stage_config.get('use_correlation', True),
            'use_cointegration': stage_config.get('use_cointegration', True),
            'use_lead_lag': stage_config.get('use_lead_lag', True),
            
            # Feature generation parameters
            'momentum_windows': stage_config.get('momentum_windows', [5, 10, 20]),
            'spread_windows': stage_config.get('spread_windows', [10, 20, 60]),
            'relative_strength_windows': stage_config.get('relative_strength_windows', [5, 10, 20]),
            'mean_reversion_windows': stage_config.get('mean_reversion_windows', [5, 10, 20]),
            
            # Cross-instrument features
            'create_basket_features': stage_config.get('create_basket_features', True),
            'create_spread_features': stage_config.get('create_spread_features', True),
            'create_momentum_features': stage_config.get('create_momentum_features', True),
            'create_mean_reversion_features': stage_config.get('create_mean_reversion_features', True),
            
            # Technical parameters
            'min_observations': stage_config.get('min_observations', 30),
            'outlier_threshold': stage_config.get('outlier_threshold', 3.0),
            'feature_prefix': stage_config.get('feature_prefix', 'tgt'),
            
            # Caching
            'cache_relationships': stage_config.get('cache_relationships', True),
            'cache_duration_hours': stage_config.get('cache_duration_hours', 24)
        }
        
        # Initialize caching system
        self._relationship_cache = {}
        self._feature_cache = {}
        self._cache_timestamps = {}
        
        self.logger.info(f"Stage 5 initialized for target: {self.params.get('target_instrument', 'TBD')}")
        self.logger.info(f"Peer selection: min_corr={self.params['min_correlation']}, max_peers={self.params['max_peers']}")
    
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """Validate input data for Stage 5"""
        errors = []
        
        # Check required columns
        required_columns = ['instrument', 'close_adj', 'log_return']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check target instrument
        if self.params['target_instrument']:
            instruments = data['instrument'].unique() if 'instrument' in data.columns else []
            if self.params['target_instrument'] not in instruments:
                errors.append(f"Target instrument '{self.params['target_instrument']}' not found in data")
        
        # Check minimum data
        if len(data) < self.params['min_observations']:
            errors.append(f"Insufficient data: {len(data)} < {self.params['min_observations']}")
        
        return errors
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this stage will create"""
        prefix = self.params['feature_prefix']
        features = []
        
        # Generate expected feature names based on configuration
        if self.params['create_momentum_features']:
            for window in self.params['momentum_windows']:
                features.extend([
                    f'{prefix}_avg_peer_momentum_{window}d',
                    f'{prefix}_momentum_vs_peers_{window}d'
                ])
        
        if self.params['create_spread_features']:
            for window in self.params['spread_windows']:
                features.append(f'{prefix}_spread_zscore_peer_{window}d')
        
        if self.params['create_basket_features']:
            for window in [5, 10, 20]:
                features.extend([
                    f'{prefix}_vs_basket_momentum_{window}d',
                    f'{prefix}_vol_vs_basket_{window}d',
                    f'{prefix}_beta_to_basket_{window}d'
                ])
        
        return features
    
    def process(self, input_data) -> 'StageData':
        """Process data (Stage 5 works with StageData objects)"""
        # Stage 5 is designed to work with StageData, not just DataFrame
        if isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to StageData if needed
            stage_data = StageData(
                data=input_data,
                metadata=StageMetadata("Stage5", "1.0.0"),
                config={'stages': {}},
                artifacts={}
            )
            result = self._process_impl(stage_data)
            return result
        else:
            # Already StageData
            return self._process_impl(input_data)
    
    def _process_impl(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 5: Target Instrument Feature Generation
        
        Args:
            input_data: StageData from Stage 4 with relationship analysis
            
        Returns:
            StageData with target-specific features
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        # Extract or set target instrument
        target_instrument = self._determine_target_instrument(data)
        if not target_instrument:
            self.logger.warning("No target instrument specified, returning original data")
            return input_data
        
        self.logger.info(f"Generating features for target instrument: {target_instrument}")
        
        # Get relationships from Stage 4 artifacts
        relationships = input_data.artifacts.get('stage4_relationships', {})
        
        # Select peer instruments
        peer_instruments = self._select_peer_instruments(data, target_instrument, relationships)
        
        if not peer_instruments:
            self.logger.warning("No suitable peer instruments found")
            return input_data
        
        self.logger.info(f"Selected {len(peer_instruments)} peer instruments: {peer_instruments}")
        
        # Generate target-specific features
        target_features = self._generate_target_features(data, target_instrument, peer_instruments, relationships)
        
        # Merge features back to original data
        enhanced_data = self._merge_target_features(data, target_features, target_instrument)
        
        # Create updated stage data
        artifacts = input_data.artifacts.copy() if input_data.artifacts else {}
        artifacts['stage5_target_features'] = {
            'target_instrument': target_instrument,
            'peer_instruments': peer_instruments,
            'features_generated': list(target_features.columns),
            'n_features': len(target_features.columns),
            'feature_categories': self._categorize_features(target_features.columns)
        }
        
        return StageData(
            data=enhanced_data,
            metadata=StageMetadata(self.stage_name, self.version, {
                'target_instrument': target_instrument,
                'peer_count': len(peer_instruments),
                'features_created': len(target_features.columns)
            }),
            config=input_data.config,
            artifacts=artifacts
        )
    
    def _determine_target_instrument(self, data: pd.DataFrame) -> Optional[str]:
        """Determine the target instrument"""
        if self.params['target_instrument']:
            return self.params['target_instrument']
        
        # If not specified, use the first instrument or most liquid instrument
        if 'instrument' in data.columns:
            instruments = data['instrument'].unique()
            if len(instruments) > 0:
                # Use instrument with highest average volume if available
                if 'volume' in data.columns:
                    volume_by_instrument = data.groupby('instrument')['volume'].mean()
                    return volume_by_instrument.idxmax()
                else:
                    return instruments[0]
        
        return None
    
    def _select_peer_instruments(self, data: pd.DataFrame, target_instrument: str, 
                                relationships: Dict[str, Any]) -> List[str]:
        """Select peer instruments based on relationships"""
        cache_key = f"peers_{target_instrument}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self._relationship_cache[cache_key]
        
        peer_candidates = []
        
        # Get all available instruments except target
        all_instruments = data['instrument'].unique() if 'instrument' in data.columns else []
        available_instruments = [inst for inst in all_instruments if inst != target_instrument]
        
        # 1. Use correlation-based peers
        if self.params['use_correlation'] and 'correlations' in relationships:
            corr_matrix = relationships['correlations'].get('correlation_matrix')
            if corr_matrix is not None and target_instrument in corr_matrix.index:
                target_correlations = corr_matrix.loc[target_instrument].abs()
                corr_peers = target_correlations[
                    target_correlations >= self.params['min_correlation']
                ].sort_values(ascending=False)
                
                for peer, correlation in corr_peers.items():
                    if peer != target_instrument and peer in available_instruments:
                        peer_candidates.append({
                            'instrument': peer,
                            'correlation': correlation,
                            'score': correlation,
                            'type': 'correlation'
                        })
        
        # 2. Use cointegration relationships
        if self.params['use_cointegration'] and 'cointegration' in relationships:
            cointegrated_pairs = relationships['cointegration'].get('cointegrated_pairs', [])
            for pair_info in cointegrated_pairs:
                if target_instrument in pair_info.get('instruments', []):
                    peer = next((inst for inst in pair_info['instruments'] if inst != target_instrument), None)
                    if peer and peer in available_instruments:
                        peer_candidates.append({
                            'instrument': peer,
                            'cointegration_stat': pair_info.get('test_statistic', 0),
                            'score': abs(pair_info.get('test_statistic', 0)),
                            'type': 'cointegration'
                        })
        
        # 3. Use lead-lag relationships
        if self.params['use_lead_lag'] and 'lead_lag' in relationships:
            lead_lag_pairs = relationships['lead_lag'].get('lead_lag_pairs', [])
            for pair_info in lead_lag_pairs:
                if target_instrument in [pair_info.get('leading'), pair_info.get('lagging')]:
                    peer = (pair_info.get('lagging') if pair_info.get('leading') == target_instrument 
                           else pair_info.get('leading'))
                    if peer and peer in available_instruments:
                        peer_candidates.append({
                            'instrument': peer,
                            'ccf': pair_info.get('max_ccf', 0),
                            'lag': pair_info.get('optimal_lag', 0),
                            'score': abs(pair_info.get('max_ccf', 0)),
                            'type': 'lead_lag'
                        })
        
        # 4. Use clustering information
        if 'clustering' in relationships:
            peer_maps = relationships['clustering'].get('peer_maps', {})
            if target_instrument in peer_maps:
                cluster_peers = peer_maps[target_instrument].get('peers', [])
                for peer_info in cluster_peers:
                    peer = peer_info.get('instrument')
                    if peer and peer in available_instruments:
                        peer_candidates.append({
                            'instrument': peer,
                            'cluster_correlation': peer_info.get('correlation', 0),
                            'score': abs(peer_info.get('correlation', 0)),
                            'type': 'clustering'
                        })
        
        # Combine and rank peers
        peer_scores = {}
        for candidate in peer_candidates:
            instrument = candidate['instrument']
            score = candidate['score']
            
            if instrument in peer_scores:
                # Average scores from different methods
                peer_scores[instrument] = (peer_scores[instrument] + score) / 2
            else:
                peer_scores[instrument] = score
        
        # Select top peers
        selected_peers = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        selected_peers = [peer for peer, score in selected_peers[:self.params['max_peers']]]
        
        # Cache results
        if self.params['cache_relationships']:
            self._relationship_cache[cache_key] = selected_peers
            self._cache_timestamps[cache_key] = datetime.now()
        
        return selected_peers
    
    def _generate_target_features(self, data: pd.DataFrame, target_instrument: str,
                                 peer_instruments: List[str], relationships: Dict[str, Any]) -> pd.DataFrame:
        """Generate features for the target instrument"""
        
        # Filter data for target and peers
        relevant_instruments = [target_instrument] + peer_instruments
        filtered_data = data[data['instrument'].isin(relevant_instruments)].copy()
        
        # Create pivot tables for feature generation
        returns_pivot = self._create_pivot_table(filtered_data, 'log_return')
        prices_pivot = self._create_pivot_table(filtered_data, 'close_adj')
        
        if returns_pivot is None or prices_pivot is None:
            self.logger.warning("Failed to create pivot tables")
            return pd.DataFrame()
        
        # Ensure target instrument is in the data
        if target_instrument not in returns_pivot.columns:
            self.logger.warning(f"Target instrument {target_instrument} not found in pivot data")
            return pd.DataFrame()
        
        target_features = pd.DataFrame(index=returns_pivot.index)
        
        # Generate different types of features
        if self.params['create_momentum_features']:
            momentum_features = self._create_momentum_features(returns_pivot, prices_pivot, target_instrument, peer_instruments)
            target_features = pd.concat([target_features, momentum_features], axis=1)
        
        if self.params['create_spread_features']:
            spread_features = self._create_spread_features(returns_pivot, prices_pivot, target_instrument, peer_instruments)
            target_features = pd.concat([target_features, spread_features], axis=1)
        
        if self.params['create_mean_reversion_features']:
            mean_reversion_features = self._create_mean_reversion_features(returns_pivot, prices_pivot, target_instrument, peer_instruments)
            target_features = pd.concat([target_features, mean_reversion_features], axis=1)
        
        if self.params['create_basket_features']:
            basket_features = self._create_basket_features(returns_pivot, prices_pivot, target_instrument, peer_instruments)
            target_features = pd.concat([target_features, basket_features], axis=1)
        
        # Add relationship-based features
        relationship_features = self._create_relationship_features(returns_pivot, target_instrument, peer_instruments, relationships)
        target_features = pd.concat([target_features, relationship_features], axis=1)
        
        # Clean up features
        target_features = self._clean_features(target_features)
        
        return target_features
    
    def _create_pivot_table(self, data: pd.DataFrame, value_col: str) -> Optional[pd.DataFrame]:
        """Create pivot table for feature generation"""
        try:
            data_copy = data.reset_index()
            date_col = 'date' if 'date' in data_copy.columns else 'index'
            
            pivot = data_copy.pivot(index=date_col, columns='instrument', values=value_col)
            pivot = pivot.dropna(thresh=2)  # Keep rows with at least 2 non-null values
            return pivot
        except Exception as e:
            self.logger.error(f"Error creating pivot table: {e}")
            return None
    
    def _create_momentum_features(self, returns_pivot: pd.DataFrame, prices_pivot: pd.DataFrame,
                                 target_instrument: str, peer_instruments: List[str]) -> pd.DataFrame:
        """Create cross-instrument momentum features"""
        features = pd.DataFrame(index=returns_pivot.index)
        prefix = self.params['feature_prefix']
        
        target_returns = returns_pivot[target_instrument]
        
        for window in self.params['momentum_windows']:
            # Peer momentum vs target
            peer_momentum_sum = 0
            peer_count = 0
            
            for peer in peer_instruments:
                if peer in returns_pivot.columns:
                    peer_returns = returns_pivot[peer]
                    peer_momentum = peer_returns.rolling(window).sum()
                    peer_momentum_sum += peer_momentum
                    peer_count += 1
                    
                    # Individual peer momentum relative to target
                    target_momentum = target_returns.rolling(window).sum()
                    relative_momentum = peer_momentum - target_momentum
                    features[f'{prefix}_peer_momentum_{peer}_{window}d'] = relative_momentum
            
            if peer_count > 0:
                # Average peer momentum
                avg_peer_momentum = peer_momentum_sum / peer_count
                features[f'{prefix}_avg_peer_momentum_{window}d'] = avg_peer_momentum
                
                # Target momentum vs peer average
                target_momentum = target_returns.rolling(window).sum()
                features[f'{prefix}_momentum_vs_peers_{window}d'] = target_momentum - avg_peer_momentum
        
        return features
    
    def _create_spread_features(self, returns_pivot: pd.DataFrame, prices_pivot: pd.DataFrame,
                               target_instrument: str, peer_instruments: List[str]) -> pd.DataFrame:
        """Create spread-based features"""
        features = pd.DataFrame(index=prices_pivot.index)
        prefix = self.params['feature_prefix']
        
        target_log_price = np.log(prices_pivot[target_instrument])
        
        for window in self.params['spread_windows']:
            for peer in peer_instruments:
                if peer in prices_pivot.columns:
                    peer_log_price = np.log(prices_pivot[peer])
                    
                    # Price spread
                    spread = target_log_price - peer_log_price
                    features[f'{prefix}_spread_{peer}_{window}d'] = spread
                    
                    # Mean-reverting spread (normalized by rolling std)
                    spread_mean = spread.rolling(window).mean()
                    spread_std = spread.rolling(window).std()
                    z_score = (spread - spread_mean) / (spread_std + 1e-8)
                    features[f'{prefix}_spread_zscore_{peer}_{window}d'] = z_score
                    
                    # Spread momentum
                    spread_momentum = spread.rolling(5).mean() - spread.rolling(window).mean()
                    features[f'{prefix}_spread_momentum_{peer}_{window}d'] = spread_momentum
        
        return features
    
    def _create_mean_reversion_features(self, returns_pivot: pd.DataFrame, prices_pivot: pd.DataFrame,
                                      target_instrument: str, peer_instruments: List[str]) -> pd.DataFrame:
        """Create mean reversion features"""
        features = pd.DataFrame(index=returns_pivot.index)
        prefix = self.params['feature_prefix']
        
        target_returns = returns_pivot[target_instrument]
        
        for window in self.params['mean_reversion_windows']:
            # Target's position relative to peers
            peer_returns_list = []
            for peer in peer_instruments:
                if peer in returns_pivot.columns:
                    peer_returns_list.append(returns_pivot[peer])
            
            if peer_returns_list:
                # Cross-sectional rank of target returns
                combined_returns = pd.concat([target_returns] + peer_returns_list, axis=1)
                target_rank = combined_returns.rank(axis=1, pct=True)[target_instrument]
                features[f'{prefix}_return_rank_{window}d'] = target_rank.rolling(window).mean()
                
                # Target deviation from peer median
                peer_median = pd.concat(peer_returns_list, axis=1).median(axis=1)
                target_deviation = target_returns - peer_median
                features[f'{prefix}_deviation_from_peers_{window}d'] = target_deviation.rolling(window).mean()
                
                # Mean reversion signal
                target_cumret = target_returns.rolling(window).sum()
                peer_median_cumret = peer_median.rolling(window).sum()
                mean_reversion_signal = peer_median_cumret - target_cumret
                features[f'{prefix}_mean_reversion_{window}d'] = mean_reversion_signal
        
        return features
    
    def _create_basket_features(self, returns_pivot: pd.DataFrame, prices_pivot: pd.DataFrame,
                               target_instrument: str, peer_instruments: List[str]) -> pd.DataFrame:
        """Create basket-based features"""
        features = pd.DataFrame(index=returns_pivot.index)
        prefix = self.params['feature_prefix']
        
        # Equal-weighted basket of peers
        peer_returns_list = []
        for peer in peer_instruments:
            if peer in returns_pivot.columns:
                peer_returns_list.append(returns_pivot[peer])
        
        if not peer_returns_list:
            return features
        
        basket_returns = pd.concat(peer_returns_list, axis=1).mean(axis=1)
        target_returns = returns_pivot[target_instrument]
        
        for window in [5, 10, 20]:
            # Target vs basket momentum
            target_momentum = target_returns.rolling(window).sum()
            basket_momentum = basket_returns.rolling(window).sum()
            features[f'{prefix}_vs_basket_momentum_{window}d'] = target_momentum - basket_momentum
            
            # Target vs basket volatility
            target_vol = target_returns.rolling(window).std()
            basket_vol = basket_returns.rolling(window).std()
            features[f'{prefix}_vol_vs_basket_{window}d'] = target_vol / (basket_vol + 1e-8)
            
            # Beta to basket
            covariance = target_returns.rolling(window).cov(basket_returns)
            basket_variance = basket_returns.rolling(window).var()
            beta_to_basket = covariance / (basket_variance + 1e-8)
            features[f'{prefix}_beta_to_basket_{window}d'] = beta_to_basket
        
        return features
    
    def _create_relationship_features(self, returns_pivot: pd.DataFrame, target_instrument: str,
                                    peer_instruments: List[str], relationships: Dict[str, Any]) -> pd.DataFrame:
        """Create features based on specific relationships"""
        features = pd.DataFrame(index=returns_pivot.index)
        prefix = self.params['feature_prefix']
        
        # Lead-lag features
        if 'lead_lag' in relationships:
            lead_lag_pairs = relationships['lead_lag'].get('lead_lag_pairs', [])
            for pair_info in lead_lag_pairs:
                if target_instrument in [pair_info.get('leading'), pair_info.get('lagging')]:
                    peer = (pair_info.get('lagging') if pair_info.get('leading') == target_instrument 
                           else pair_info.get('leading'))
                    lag = pair_info.get('optimal_lag', 0)
                    
                    if peer in returns_pivot.columns and abs(lag) <= 5:
                        peer_returns = returns_pivot[peer]
                        if lag > 0:
                            # Peer leads target
                            features[f'{prefix}_lead_signal_{peer}'] = peer_returns.shift(lag)
                        elif lag < 0:
                            # Target leads peer
                            features[f'{prefix}_lag_signal_{peer}'] = peer_returns.shift(-lag)
        
        return features
    
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                # If no date column, assume index is already properly set
                pass
        
        return data.sort_index()
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate generated features"""
        if features.empty:
            return features
        
        # Remove features with too many NaN values
        features = features.dropna(thresh=len(features) * 0.3, axis=1)
        
        # Handle outliers
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                # Winsorize extreme outliers
                q99 = features[col].quantile(0.99)
                q01 = features[col].quantile(0.01)
                features[col] = features[col].clip(q01, q99)
        
        # Forward fill remaining NaN values
        features = features.ffill()
        
        return features
    
    def _merge_target_features(self, data: pd.DataFrame, target_features: pd.DataFrame,
                              target_instrument: str) -> pd.DataFrame:
        """Merge target-specific features back to the main dataset"""
        enhanced_data = data.copy()
        
        if target_features.empty:
            return enhanced_data
        
        # Reset index to merge on datetime
        target_features_reset = target_features.reset_index()
        date_col = 'date' if 'date' in target_features_reset.columns else 'index'
        
        # Merge with target instrument data only
        target_mask = enhanced_data['instrument'] == target_instrument
        target_data = enhanced_data[target_mask].copy().reset_index()
        target_date_col = 'date' if 'date' in target_data.columns else 'index'
        
        # Perform merge
        merged_target = target_data.merge(
            target_features_reset, 
            left_on=target_date_col, 
            right_on=date_col, 
            how='left'
        )
        
        # Update enhanced_data with merged features
        for col in target_features.columns:
            if col not in enhanced_data.columns:
                enhanced_data[col] = np.nan
            enhanced_data.loc[target_mask, col] = merged_target[col].values
        
        return enhanced_data
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize generated features"""
        categories = {
            'momentum': [],
            'spread': [],
            'mean_reversion': [],
            'basket': [],
            'relationship': []
        }
        
        for feature in feature_names:
            if 'momentum' in feature:
                categories['momentum'].append(feature)
            elif 'spread' in feature:
                categories['spread'].append(feature)
            elif 'reversion' in feature or 'rank' in feature or 'deviation' in feature:
                categories['mean_reversion'].append(feature)
            elif 'basket' in feature:
                categories['basket'].append(feature)
            elif 'lead' in feature or 'lag' in feature:
                categories['relationship'].append(feature)
        
        return {k: v for k, v in categories.items() if v}  # Only non-empty categories
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if not self.params['cache_relationships']:
            return False
        
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[cache_key]
        max_age = timedelta(hours=self.params['cache_duration_hours'])
        
        return cache_age < max_age
    
    def clear_cache(self):
        """Clear all cached data"""
        self._relationship_cache.clear()
        self._feature_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Stage 5 cache cleared")