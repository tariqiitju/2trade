"""
Stage 4: Instrument Relationships for Da Vinchi Feature Engineering Pipeline

This stage implements cross-instrument relationship analysis as specified in the plan-draft:
- Rolling correlation network analysis
- Lead-lag relationship detection using cross-correlation
- Cointegration analysis (Engle-Granger method)
- Peer clustering and relationship mapping
- Distance-based clustering of instruments

Creates relationship maps that will be used by Stage 5 for cross-instrument features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
from itertools import combinations
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import warnings

from .stage_base import FeatureStage, StageData, StageMetadata

logger = logging.getLogger(__name__)


class Stage4Relationships(FeatureStage):
    """
    Stage 4: Instrument Relationships
    
    Analyzes relationships between instruments to create maps that enable
    cross-instrument feature engineering. Includes correlation networks,
    lead-lag analysis, cointegration testing, and peer clustering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "Stage4_Relationships", "1.0.0")
        
        self.params = {
            # Correlation analysis
            'correlation_windows': config.get('correlation_windows', [30, 60, 90, 126]),
            'correlation_method': config.get('correlation_method', 'spearman'),  # 'pearson', 'spearman'
            'min_correlation': config.get('min_correlation', 0.4),
            'correlation_significance': config.get('correlation_significance', 0.05),
            'shrinkage_method': config.get('shrinkage_method', 'ledoit_wolf'),
            
            # Clustering parameters
            'clustering_method': config.get('clustering_method', 'hierarchical'),  # 'hierarchical', 'kmeans'
            'n_clusters': config.get('n_clusters', 8),
            'distance_threshold': config.get('distance_threshold', 0.7),
            'max_peers_per_instrument': config.get('max_peers_per_instrument', 10),
            
            # Lead-lag analysis
            'max_lags': config.get('max_lags', 5),
            'min_ccf': config.get('min_ccf', 0.3),
            'ccf_significance': config.get('ccf_significance', 0.05),
            'lead_lag_window': config.get('lead_lag_window', 60),
            
            # Cointegration analysis
            'cointegration_method': config.get('cointegration_method', 'engle_granger'),
            'adf_significance': config.get('adf_significance', 0.05),
            'min_half_life': config.get('min_half_life', 1),
            'max_half_life': config.get('max_half_life', 30),
            'cointegration_window': config.get('cointegration_window', 126),
            
            # Universe constraints
            'min_instruments': config.get('min_instruments', 5),
            'min_observations': config.get('min_observations', 50)
        }
    
    def _get_required_columns(self) -> List[str]:
        """Required columns from previous stages"""
        return ['close_adj', 'log_return', 'instrument']
    
    def process(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 4: Instrument Relationships
        
        Args:
            input_data: StageData from previous stages
            
        Returns:
            StageData with relationship artifacts for Stage 5
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        # Check if we have multiple instruments
        if 'instrument' not in data.columns:
            self.logger.warning("No instrument column found, creating single instrument dataset")
            data['instrument'] = 'single'
        
        instruments = data['instrument'].unique()
        n_instruments = len(instruments)
        
        if n_instruments < self.params['min_instruments']:
            self.logger.warning(f"Only {n_instruments} instruments, skipping relationship analysis")
            return input_data
        
        self.logger.info(f"Analyzing relationships for {n_instruments} instruments")
        
        # Create pivot tables for analysis
        returns_pivot = self._create_returns_pivot(data)
        prices_pivot = self._create_prices_pivot(data)
        
        if returns_pivot is None or prices_pivot is None:
            self.logger.warning("Failed to create pivot tables for analysis")
            return input_data
        
        # Analyze relationships
        relationships = {}
        
        # 1. Correlation analysis and clustering
        correlation_results = self._analyze_correlations(returns_pivot)
        relationships['correlations'] = correlation_results
        
        # 2. Peer clustering
        clustering_results = self._perform_clustering(correlation_results['correlation_matrix'])
        relationships['clustering'] = clustering_results
        
        # 3. Lead-lag analysis
        lead_lag_results = self._analyze_lead_lag(returns_pivot)
        relationships['lead_lag'] = lead_lag_results
        
        # 4. Cointegration analysis
        cointegration_results = self._analyze_cointegration(prices_pivot)
        relationships['cointegration'] = cointegration_results
        
        # 5. Create peer maps
        peer_maps = self._create_peer_maps(correlation_results, clustering_results)
        relationships['peer_maps'] = peer_maps
        
        # Create result metadata
        result_metadata = StageMetadata(
            stage_name=self.stage_name,
            version=self.version,
            input_shape=input_data.data.shape,
            output_shape=data.shape,
            feature_count=0  # This stage creates artifacts, not direct features
        )
        
        self.logger.info("Completed relationship analysis")
        
        return StageData(
            data=data,
            metadata=result_metadata,
            config=input_data.config,
            artifacts={
                **input_data.artifacts,
                'stage4_relationships': relationships,
                'stage4_params': self.params,
                'instruments': instruments.tolist()
            }
        )
    
    def _create_returns_pivot(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create pivot table of returns by instrument and date"""
        try:
            # Reset index to use datetime as column for pivot
            data_copy = data.reset_index()
            date_col = 'date' if 'date' in data_copy.columns else 'index'
            
            pivot = data_copy.pivot(index=date_col, columns='instrument', values='log_return')
            pivot = pivot.dropna(thresh=len(pivot.columns) * 0.7)  # Keep rows with 70% non-null
            return pivot
        except Exception as e:
            self.logger.error(f"Error creating returns pivot: {e}")
            return None
    
    def _create_prices_pivot(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create pivot table of log prices by instrument and date"""
        try:
            data_copy = data.reset_index()
            data_copy['log_price'] = np.log(data_copy['close_adj'])
            date_col = 'date' if 'date' in data_copy.columns else 'index'
            
            pivot = data_copy.pivot(index=date_col, columns='instrument', values='log_price')
            pivot = pivot.dropna(thresh=len(pivot.columns) * 0.7)
            return pivot
        except Exception as e:
            self.logger.error(f"Error creating prices pivot: {e}")
            return None
    
    def _analyze_correlations(self, returns_pivot: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rolling correlations (Section 4.1)"""
        self.logger.info("Analyzing correlations")
        
        results = {}
        instruments = returns_pivot.columns.tolist()
        
        # Compute correlation matrices for different windows
        correlation_matrices = {}
        
        for window in self.params['correlation_windows']:
            self.logger.debug(f"Computing {window}-day rolling correlations")
            
            if self.params['correlation_method'] == 'spearman':
                corr_func = lambda x: x.corr(method='spearman')
            else:
                corr_func = lambda x: x.corr(method='pearson')
            
            # Rolling correlation matrix
            rolling_corr = returns_pivot.rolling(window).apply(
                lambda x: corr_func(x.to_frame().T).iloc[0, 0] if len(x.dropna()) >= window//2 else np.nan,
                raw=False
            )
            
            # Use the most recent correlation matrix
            latest_corr = returns_pivot.iloc[-window:].corr(method=self.params['correlation_method'])
            
            # Apply shrinkage if enabled
            if self.params['shrinkage_method'] == 'ledoit_wolf' and len(instruments) > 5:
                try:
                    clean_data = returns_pivot.iloc[-window:].dropna()
                    if len(clean_data) >= len(instruments):
                        lw = LedoitWolf()
                        shrunk_cov = lw.fit(clean_data).covariance_
                        # Convert to correlation
                        std = np.sqrt(np.diag(shrunk_cov))
                        shrunk_corr = shrunk_cov / np.outer(std, std)
                        latest_corr = pd.DataFrame(
                            shrunk_corr, 
                            index=instruments, 
                            columns=instruments
                        )
                except Exception as e:
                    self.logger.warning(f"Shrinkage failed: {e}")
            
            correlation_matrices[f'{window}d'] = latest_corr
        
        # Use the longest window for primary analysis
        primary_window = max(self.params['correlation_windows'])
        primary_corr = correlation_matrices[f'{primary_window}d']
        
        # Compute distance matrix
        distance_matrix = np.sqrt(2 * (1 - primary_corr.values))
        distance_df = pd.DataFrame(
            distance_matrix,
            index=primary_corr.index,
            columns=primary_corr.columns
        )
        
        results['correlation_matrices'] = correlation_matrices
        results['correlation_matrix'] = primary_corr
        results['distance_matrix'] = distance_df
        results['instruments'] = instruments
        
        return results
    
    def _perform_clustering(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering based on correlation distance"""
        self.logger.info("Performing instrument clustering")
        
        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix.values))
        
        # Fill any NaN values
        distance_matrix = pd.DataFrame(distance_matrix).fillna(1.0).values
        
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Get cluster labels using distance threshold
        cluster_labels = fcluster(
            linkage_matrix, 
            t=self.params['distance_threshold'], 
            criterion='distance'
        )
        
        # Alternative: fixed number of clusters
        cluster_labels_fixed = fcluster(
            linkage_matrix,
            t=self.params['n_clusters'],
            criterion='maxclust'
        )
        
        instruments = correlation_matrix.index.tolist()
        
        # Create cluster mappings
        cluster_map = dict(zip(instruments, cluster_labels))
        cluster_map_fixed = dict(zip(instruments, cluster_labels_fixed))
        
        # Create reverse mapping (cluster -> instruments)
        clusters = {}
        for instrument, cluster_id in cluster_map.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(instrument)
        
        results = {
            'linkage_matrix': linkage_matrix,
            'cluster_labels': cluster_labels,
            'cluster_labels_fixed': cluster_labels_fixed,
            'cluster_map': cluster_map,
            'cluster_map_fixed': cluster_map_fixed,
            'clusters': clusters,
            'n_clusters_adaptive': len(clusters),
            'n_clusters_fixed': self.params['n_clusters']
        }
        
        return results
    
    def _analyze_lead_lag(self, returns_pivot: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lead-lag relationships (Section 4.2)"""
        self.logger.info("Analyzing lead-lag relationships")
        
        instruments = returns_pivot.columns.tolist()
        lead_lag_pairs = []
        
        # Analyze pairwise lead-lag relationships
        for i, instr_i in enumerate(instruments):
            for j, instr_j in enumerate(instruments):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                try:
                    # Get aligned data
                    data_i = returns_pivot[instr_i].dropna()
                    data_j = returns_pivot[instr_j].dropna()
                    
                    # Find common dates
                    common_index = data_i.index.intersection(data_j.index)
                    if len(common_index) < self.params['min_observations']:
                        continue
                    
                    series_i = data_i[common_index].iloc[-self.params['lead_lag_window']:]
                    series_j = data_j[common_index].iloc[-self.params['lead_lag_window']:]
                    
                    if len(series_i) < self.params['min_observations']:
                        continue
                    
                    # Compute cross-correlation for different lags
                    max_ccf = 0
                    best_lag = 0
                    ccf_values = []
                    
                    for lag in range(1, self.params['max_lags'] + 1):
                        # i leads j by lag periods
                        if len(series_i) > lag:
                            ccf_pos = series_i.iloc[:-lag].corr(series_j.iloc[lag:])
                            ccf_values.append((lag, ccf_pos, 'i_leads_j'))
                            
                            if abs(ccf_pos) > abs(max_ccf):
                                max_ccf = ccf_pos
                                best_lag = lag
                        
                        # j leads i by lag periods  
                        if len(series_j) > lag:
                            ccf_neg = series_j.iloc[:-lag].corr(series_i.iloc[lag:])
                            ccf_values.append((-lag, ccf_neg, 'j_leads_i'))
                            
                            if abs(ccf_neg) > abs(max_ccf):
                                max_ccf = ccf_neg
                                best_lag = -lag
                    
                    # Check significance
                    if abs(max_ccf) >= self.params['min_ccf']:
                        lead_lag_pairs.append({
                            'instrument_i': instr_i,
                            'instrument_j': instr_j,
                            'best_lag': best_lag,
                            'max_ccf': max_ccf,
                            'ccf_values': ccf_values,
                            'relationship': 'i_leads_j' if best_lag > 0 else 'j_leads_i'
                        })
                
                except Exception as e:
                    self.logger.debug(f"Error in lead-lag analysis for {instr_i}-{instr_j}: {e}")
                    continue
        
        # Create lead-lag mapping
        lead_lag_map = {}
        for pair in lead_lag_pairs:
            leader = pair['instrument_i'] if pair['best_lag'] > 0 else pair['instrument_j']
            follower = pair['instrument_j'] if pair['best_lag'] > 0 else pair['instrument_i']
            lag_value = abs(pair['best_lag'])
            
            if follower not in lead_lag_map:
                lead_lag_map[follower] = []
            
            lead_lag_map[follower].append({
                'leader': leader,
                'lag': lag_value,
                'ccf': pair['max_ccf']
            })
        
        results = {
            'lead_lag_pairs': lead_lag_pairs,
            'lead_lag_map': lead_lag_map,
            'n_relationships': len(lead_lag_pairs)
        }
        
        return results
    
    def _analyze_cointegration(self, prices_pivot: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cointegration relationships (Section 4.3)"""
        self.logger.info("Analyzing cointegration relationships")
        
        instruments = prices_pivot.columns.tolist()
        cointegrated_pairs = []
        
        # Test pairwise cointegration (simplified Engle-Granger)
        for i, instr_i in enumerate(instruments):
            for j, instr_j in enumerate(instruments):
                if i >= j:  # Avoid duplicates
                    continue
                
                try:
                    # Get aligned price data
                    data_i = prices_pivot[instr_i].dropna()
                    data_j = prices_pivot[instr_j].dropna()
                    
                    # Find common dates
                    common_index = data_i.index.intersection(data_j.index)
                    if len(common_index) < self.params['min_observations']:
                        continue
                    
                    series_i = data_i[common_index].iloc[-self.params['cointegration_window']:]
                    series_j = data_j[common_index].iloc[-self.params['cointegration_window']:]
                    
                    if len(series_i) < self.params['min_observations']:
                        continue
                    
                    # Step 1: OLS regression P_i = a + b * P_j + epsilon
                    X = np.column_stack([np.ones(len(series_j)), series_j.values])
                    y = series_i.values
                    
                    try:
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                        intercept, beta = coeffs[0], coeffs[1]
                    except np.linalg.LinAlgError:
                        continue
                    
                    # Calculate residuals (spread)
                    residuals = y - (intercept + beta * series_j.values)
                    
                    # Step 2: Simplified ADF test on residuals
                    # (Using a basic mean reversion test instead of full ADF)
                    spread_series = pd.Series(residuals, index=series_i.index)
                    
                    # Test for mean reversion using AR(1) model
                    spread_lagged = spread_series.shift(1).dropna()
                    spread_diff = spread_series.diff().dropna()
                    
                    if len(spread_lagged) < 20:
                        continue
                    
                    # Simple AR(1): Δspread_t = α + β * spread_{t-1} + error
                    aligned_data = pd.DataFrame({
                        'spread_lag': spread_lagged,
                        'spread_diff': spread_diff
                    }).dropna()
                    
                    if len(aligned_data) < 20:
                        continue
                    
                    try:
                        X_ar = np.column_stack([
                            np.ones(len(aligned_data)),
                            aligned_data['spread_lag'].values
                        ])
                        y_ar = aligned_data['spread_diff'].values
                        
                        ar_coeffs = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
                        alpha, rho = ar_coeffs[0], ar_coeffs[1]
                        
                        # Calculate half-life of mean reversion
                        if rho < 0:  # Mean reverting
                            half_life = -np.log(2) / np.log(1 + rho)
                        else:
                            half_life = np.inf
                        
                        # Simple stationarity test (if rho is significantly negative)
                        # This is a simplified version of ADF test
                        t_stat = rho / (np.std(aligned_data['spread_lag']) + 1e-10)
                        is_stationary = (rho < -0.1) and (half_life <= self.params['max_half_life'])
                        
                        if is_stationary and half_life >= self.params['min_half_life']:
                            # Calculate z-score of current spread
                            spread_mean = np.mean(residuals)
                            spread_std = np.std(residuals)
                            current_z_score = (residuals[-1] - spread_mean) / (spread_std + 1e-10)
                            
                            cointegrated_pairs.append({
                                'instrument_i': instr_i,
                                'instrument_j': instr_j,
                                'beta': beta,
                                'intercept': intercept,
                                'half_life': half_life,
                                'current_z_score': current_z_score,
                                'spread_mean': spread_mean,
                                'spread_std': spread_std,
                                't_stat': t_stat,
                                'rho': rho
                            })
                    
                    except np.linalg.LinAlgError:
                        continue
                
                except Exception as e:
                    self.logger.debug(f"Error in cointegration test for {instr_i}-{instr_j}: {e}")
                    continue
        
        # Create cointegration mapping
        cointegration_map = {}
        for pair in cointegrated_pairs:
            instr_i = pair['instrument_i']
            instr_j = pair['instrument_j']
            
            # Add both directions
            if instr_i not in cointegration_map:
                cointegration_map[instr_i] = []
            if instr_j not in cointegration_map:
                cointegration_map[instr_j] = []
            
            cointegration_map[instr_i].append({
                'partner': instr_j,
                'beta': pair['beta'],
                'half_life': pair['half_life'],
                'current_z_score': pair['current_z_score']
            })
            
            cointegration_map[instr_j].append({
                'partner': instr_i,
                'beta': 1.0 / pair['beta'],  # Inverse relationship
                'half_life': pair['half_life'],
                'current_z_score': -pair['current_z_score']  # Opposite sign
            })
        
        results = {
            'cointegrated_pairs': cointegrated_pairs,
            'cointegration_map': cointegration_map,
            'n_cointegrated_pairs': len(cointegrated_pairs)
        }
        
        return results
    
    def _create_peer_maps(self, correlation_results: Dict, clustering_results: Dict) -> Dict[str, Any]:
        """Create peer mappings for each instrument"""
        self.logger.info("Creating peer mappings")
        
        correlation_matrix = correlation_results['correlation_matrix']
        cluster_map = clustering_results['cluster_map']
        instruments = correlation_results['instruments']
        
        peer_maps = {}
        
        for instrument in instruments:
            # Get correlations for this instrument
            correlations = correlation_matrix.loc[instrument].abs().sort_values(ascending=False)
            
            # Remove self-correlation
            correlations = correlations[correlations.index != instrument]
            
            # Filter by minimum correlation threshold
            strong_correlations = correlations[correlations >= self.params['min_correlation']]
            
            # Take top peers up to maximum
            top_peers = strong_correlations.head(self.params['max_peers_per_instrument'])
            
            # Get cluster peers
            instrument_cluster = cluster_map.get(instrument, -1)
            cluster_peers = [
                instr for instr, cluster_id in cluster_map.items()
                if cluster_id == instrument_cluster and instr != instrument
            ]
            
            # Combine and create peer list
            peer_list = []
            for peer_instrument in top_peers.index:
                correlation = correlation_matrix.loc[instrument, peer_instrument]
                is_cluster_peer = peer_instrument in cluster_peers
                
                peer_list.append({
                    'instrument': peer_instrument,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation),
                    'weight': abs(correlation),  # Use absolute correlation as weight
                    'is_cluster_peer': is_cluster_peer
                })
            
            # Sort by absolute correlation
            peer_list.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            peer_maps[instrument] = {
                'peers': peer_list,
                'cluster_id': instrument_cluster,
                'cluster_peers': cluster_peers,
                'n_peers': len(peer_list)
            }
        
        return peer_maps
    
    def get_feature_names(self) -> List[str]:
        """This stage creates artifacts, not direct features"""
        return []