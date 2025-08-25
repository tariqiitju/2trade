"""
Correlation analysis models for relationship discovery
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau

from ..base import CorrelationModel
from ..config import ModelConfig
from ..exceptions import ModelError, TrainingError


class PearsonCorrelationModel(CorrelationModel):
    """Pearson correlation analysis for linear relationships"""
    
    def _build_model(self):
        # Pearson correlation doesn't need a model object
        return None
    
    def _train_model(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Calculate Pearson correlations"""
        if Y is None:
            # Calculate correlation matrix within X
            self.correlation_matrix = X.corr(method='pearson')
            self.pairwise_correlations = self._get_pairwise_correlations(X, X, pearsonr)
        else:
            # Calculate correlations between X and Y
            self.correlation_matrix = self._cross_correlation_matrix(X, Y, 'pearson')
            self.pairwise_correlations = self._get_pairwise_correlations(X, Y, pearsonr)
        
        # Summary statistics
        correlations = self.correlation_matrix.values
        correlations = correlations[np.triu_indices_from(correlations, k=1)]
        
        history = {
            'n_features_X': X.shape[1],
            'n_features_Y': Y.shape[1] if Y is not None else X.shape[1],
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'max_correlation': float(np.max(correlations)),
            'min_correlation': float(np.min(correlations)),
            'strong_correlations': self._count_strong_correlations(correlations)
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Return correlation analysis results"""
        return {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'pairwise_correlations': self.pairwise_correlations,
            'strong_positive': self._get_strong_correlations(threshold=0.7, direction='positive'),
            'strong_negative': self._get_strong_correlations(threshold=-0.7, direction='negative')
        }
    
    def _get_pairwise_correlations(self, X: pd.DataFrame, Y: pd.DataFrame, corr_func) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise correlations with p-values"""
        results = {}
        for col_x in X.columns:
            results[col_x] = {}
            for col_y in Y.columns:
                if col_x != col_y or X is not Y:
                    corr, p_value = corr_func(X[col_x].dropna(), Y[col_y].dropna())
                    results[col_x][col_y] = {
                        'correlation': float(corr),
                        'p_value': float(p_value)
                    }
        return results
    
    def _cross_correlation_matrix(self, X: pd.DataFrame, Y: pd.DataFrame, method: str) -> pd.DataFrame:
        """Calculate cross-correlation matrix between X and Y"""
        corr_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)
        
        for col_x in X.columns:
            for col_y in Y.columns:
                if method == 'pearson':
                    corr, _ = pearsonr(X[col_x].dropna(), Y[col_y].dropna())
                elif method == 'spearman':
                    corr, _ = spearmanr(X[col_x].dropna(), Y[col_y].dropna())
                elif method == 'kendall':
                    corr, _ = kendalltau(X[col_x].dropna(), Y[col_y].dropna())
                
                corr_matrix.loc[col_x, col_y] = corr
        
        return corr_matrix.astype(float)
    
    def _count_strong_correlations(self, correlations: np.ndarray, threshold: float = 0.7) -> Dict[str, int]:
        """Count strong correlations"""
        return {
            'strong_positive': int(np.sum(correlations > threshold)),
            'strong_negative': int(np.sum(correlations < -threshold)),
            'moderate': int(np.sum((np.abs(correlations) > 0.3) & (np.abs(correlations) <= threshold))),
            'weak': int(np.sum(np.abs(correlations) <= 0.3))
        }
    
    def _get_strong_correlations(self, threshold: float = 0.7, direction: str = 'positive') -> List[Dict[str, Any]]:
        """Get pairs with strong correlations"""
        strong_pairs = []
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                
                if direction == 'positive' and corr_value > threshold:
                    strong_pairs.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
                elif direction == 'negative' and corr_value < threshold:
                    strong_pairs.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return strong_pairs


class SpearmanCorrelationModel(CorrelationModel):
    """Spearman rank correlation for monotonic relationships"""
    
    def _build_model(self):
        return None
    
    def _train_model(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Calculate Spearman correlations"""
        if Y is None:
            self.correlation_matrix = X.corr(method='spearman')
            self.pairwise_correlations = self._get_pairwise_correlations(X, X, spearmanr)
        else:
            self.correlation_matrix = self._cross_correlation_matrix(X, Y, 'spearman')
            self.pairwise_correlations = self._get_pairwise_correlations(X, Y, spearmanr)
        
        correlations = self.correlation_matrix.values
        correlations = correlations[np.triu_indices_from(correlations, k=1)]
        
        history = {
            'n_features_X': X.shape[1],
            'n_features_Y': Y.shape[1] if Y is not None else X.shape[1],
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'max_correlation': float(np.max(correlations)),
            'min_correlation': float(np.min(correlations)),
            'strong_correlations': self._count_strong_correlations(correlations)
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        return {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'pairwise_correlations': self.pairwise_correlations,
            'strong_positive': self._get_strong_correlations(threshold=0.7, direction='positive'),
            'strong_negative': self._get_strong_correlations(threshold=-0.7, direction='negative')
        }
    
    # Inherit utility methods from PearsonCorrelationModel
    _get_pairwise_correlations = PearsonCorrelationModel._get_pairwise_correlations
    _cross_correlation_matrix = PearsonCorrelationModel._cross_correlation_matrix
    _count_strong_correlations = PearsonCorrelationModel._count_strong_correlations
    _get_strong_correlations = PearsonCorrelationModel._get_strong_correlations


class KendallCorrelationModel(CorrelationModel):
    """Kendall's tau correlation for concordant relationships"""
    
    def _build_model(self):
        return None
    
    def _train_model(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Calculate Kendall correlations"""
        if Y is None:
            self.correlation_matrix = X.corr(method='kendall')
            self.pairwise_correlations = self._get_pairwise_correlations(X, X, kendalltau)
        else:
            self.correlation_matrix = self._cross_correlation_matrix(X, Y, 'kendall')
            self.pairwise_correlations = self._get_pairwise_correlations(X, Y, kendalltau)
        
        correlations = self.correlation_matrix.values
        correlations = correlations[np.triu_indices_from(correlations, k=1)]
        
        history = {
            'n_features_X': X.shape[1],
            'n_features_Y': Y.shape[1] if Y is not None else X.shape[1],
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'max_correlation': float(np.max(correlations)),
            'min_correlation': float(np.min(correlations)),
            'strong_correlations': self._count_strong_correlations(correlations)
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        return {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'pairwise_correlations': self.pairwise_correlations,
            'strong_positive': self._get_strong_correlations(threshold=0.7, direction='positive'),
            'strong_negative': self._get_strong_correlations(threshold=-0.7, direction='negative')
        }
    
    # Inherit utility methods
    _get_pairwise_correlations = PearsonCorrelationModel._get_pairwise_correlations
    _cross_correlation_matrix = PearsonCorrelationModel._cross_correlation_matrix
    _count_strong_correlations = PearsonCorrelationModel._count_strong_correlations
    _get_strong_correlations = PearsonCorrelationModel._get_strong_correlations


class TailDependenceModel(CorrelationModel):
    """Tail dependence analysis using copulas for extreme events"""
    
    def _build_model(self):
        try:
            from scipy.stats import rankdata
        except ImportError:
            raise ModelError("scipy not available for tail dependence analysis")
        
        return None
    
    def _train_model(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Calculate tail dependence coefficients"""
        threshold = kwargs.get('threshold', 0.95)
        
        if Y is None:
            # Calculate within X
            self.tail_dependence = self._calculate_tail_dependence_matrix(X, X, threshold)
        else:
            # Calculate between X and Y
            self.tail_dependence = self._calculate_tail_dependence_matrix(X, Y, threshold)
        
        # Summary statistics
        upper_tail_coeffs = [v['upper_tail'] for v in self.tail_dependence.values() 
                           if isinstance(v, dict) and 'upper_tail' in v]
        lower_tail_coeffs = [v['lower_tail'] for v in self.tail_dependence.values() 
                           if isinstance(v, dict) and 'lower_tail' in v]
        
        history = {
            'threshold': threshold,
            'n_pairs': len(upper_tail_coeffs),
            'mean_upper_tail': float(np.mean(upper_tail_coeffs)) if upper_tail_coeffs else 0,
            'mean_lower_tail': float(np.mean(lower_tail_coeffs)) if lower_tail_coeffs else 0,
            'strong_upper_tail_pairs': sum(1 for x in upper_tail_coeffs if x > 0.3),
            'strong_lower_tail_pairs': sum(1 for x in lower_tail_coeffs if x > 0.3)
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        return {
            'tail_dependence': self.tail_dependence,
            'high_upper_tail_pairs': self._get_high_tail_pairs('upper_tail', threshold=0.3),
            'high_lower_tail_pairs': self._get_high_tail_pairs('lower_tail', threshold=0.3)
        }
    
    def _calculate_tail_dependence_matrix(self, X: pd.DataFrame, Y: pd.DataFrame, threshold: float) -> Dict[str, Dict[str, float]]:
        """Calculate tail dependence coefficients"""
        from scipy.stats import rankdata
        
        results = {}
        for col_x in X.columns:
            results[col_x] = {}
            for col_y in Y.columns:
                if col_x != col_y or X is not Y:
                    # Convert to pseudo-observations (uniform margins)
                    u = rankdata(X[col_x]) / (len(X[col_x]) + 1)
                    v = rankdata(Y[col_y]) / (len(Y[col_y]) + 1)
                    
                    # Calculate tail dependence coefficients
                    upper_tail = self._upper_tail_dependence(u, v, threshold)
                    lower_tail = self._lower_tail_dependence(u, v, 1 - threshold)
                    
                    results[col_x][col_y] = {
                        'upper_tail': float(upper_tail),
                        'lower_tail': float(lower_tail)
                    }
        
        return results
    
    def _upper_tail_dependence(self, u: np.ndarray, v: np.ndarray, threshold: float) -> float:
        """Calculate upper tail dependence coefficient"""
        # Count observations where both variables exceed threshold
        both_high = np.sum((u > threshold) & (v > threshold))
        u_high = np.sum(u > threshold)
        
        if u_high == 0:
            return 0.0
        
        # Empirical upper tail dependence coefficient
        return 2 - (both_high / u_high) if u_high > 0 else 0.0
    
    def _lower_tail_dependence(self, u: np.ndarray, v: np.ndarray, threshold: float) -> float:
        """Calculate lower tail dependence coefficient"""
        # Count observations where both variables are below threshold
        both_low = np.sum((u < threshold) & (v < threshold))
        u_low = np.sum(u < threshold)
        
        if u_low == 0:
            return 0.0
        
        # Empirical lower tail dependence coefficient
        return (both_low / u_low) if u_low > 0 else 0.0
    
    def _get_high_tail_pairs(self, tail_type: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get pairs with high tail dependence"""
        high_pairs = []
        
        for col1, pairs in self.tail_dependence.items():
            for col2, coeffs in pairs.items():
                if isinstance(coeffs, dict) and tail_type in coeffs:
                    if coeffs[tail_type] > threshold:
                        high_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'coefficient': coeffs[tail_type],
                            'tail_type': tail_type
                        })
        
        return high_pairs


class MutualInformationModel(CorrelationModel):
    """Mutual information analysis for any statistical relationships"""
    
    def _build_model(self):
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            raise ModelError("scikit-learn not available for mutual information analysis")
        
        return None
    
    def _train_model(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Calculate mutual information scores"""
        if Y is None:
            # Calculate MI matrix within X
            self.mi_matrix = self._calculate_mi_matrix(X)
        else:
            # Calculate MI between X and Y
            self.mi_matrix = self._calculate_cross_mi_matrix(X, Y)
        
        # Summary statistics
        mi_values = []
        for i in range(len(self.mi_matrix.columns)):
            for j in range(len(self.mi_matrix.columns)):
                if i != j:
                    mi_values.append(self.mi_matrix.iloc[i, j])
        
        history = {
            'n_features_X': X.shape[1],
            'n_features_Y': Y.shape[1] if Y is not None else X.shape[1],
            'mean_mi': float(np.mean(mi_values)),
            'std_mi': float(np.std(mi_values)),
            'max_mi': float(np.max(mi_values)),
            'min_mi': float(np.min(mi_values)),
            'high_mi_pairs': sum(1 for x in mi_values if x > np.percentile(mi_values, 75))
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        return {
            'mutual_information_matrix': self.mi_matrix.to_dict(),
            'high_mi_pairs': self._get_high_mi_pairs(threshold=np.percentile(self.mi_matrix.values.flatten(), 75)),
            'feature_rankings': self._rank_features_by_mi()
        }
    
    def _calculate_mi_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate mutual information matrix within X"""
        from sklearn.feature_selection import mutual_info_regression
        
        n_features = X.shape[1]
        mi_matrix = pd.DataFrame(
            np.zeros((n_features, n_features)),
            index=X.columns,
            columns=X.columns
        )
        
        for i, col in enumerate(X.columns):
            # Calculate MI between this column and all others
            X_others = X.drop(columns=[col])
            if not X_others.empty:
                mi_scores = mutual_info_regression(X_others, X[col], random_state=42)
                
                for j, other_col in enumerate(X_others.columns):
                    mi_matrix.loc[col, other_col] = mi_scores[j]
                    mi_matrix.loc[other_col, col] = mi_scores[j]  # Symmetric
        
        return mi_matrix
    
    def _calculate_cross_mi_matrix(self, X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
        """Calculate mutual information matrix between X and Y"""
        from sklearn.feature_selection import mutual_info_regression
        
        mi_matrix = pd.DataFrame(
            np.zeros((len(X.columns), len(Y.columns))),
            index=X.columns,
            columns=Y.columns
        )
        
        for y_col in Y.columns:
            mi_scores = mutual_info_regression(X, Y[y_col], random_state=42)
            
            for i, x_col in enumerate(X.columns):
                mi_matrix.loc[x_col, y_col] = mi_scores[i]
        
        return mi_matrix
    
    def _get_high_mi_pairs(self, threshold: float) -> List[Dict[str, Any]]:
        """Get pairs with high mutual information"""
        high_pairs = []
        
        for i in range(len(self.mi_matrix.columns)):
            for j in range(len(self.mi_matrix.columns)):
                if i != j:
                    mi_value = self.mi_matrix.iloc[i, j]
                    if mi_value > threshold:
                        high_pairs.append({
                            'feature1': self.mi_matrix.index[i],
                            'feature2': self.mi_matrix.columns[j],
                            'mutual_information': float(mi_value)
                        })
        
        return high_pairs
    
    def _rank_features_by_mi(self) -> Dict[str, List[Dict[str, float]]]:
        """Rank features by their mutual information with others"""
        rankings = {}
        
        for col in self.mi_matrix.columns:
            # Get MI scores for this feature with all others
            mi_scores = self.mi_matrix[col].drop(col).sort_values(ascending=False)
            
            rankings[col] = [
                {'feature': idx, 'mutual_information': float(score)}
                for idx, score in mi_scores.items()
            ]
        
        return rankings