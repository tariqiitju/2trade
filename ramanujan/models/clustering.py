"""
Unsupervised clustering models for pattern discovery
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ..base import UnsupervisedModel
from ..config import ModelConfig
from ..exceptions import ModelError, TrainingError


class KMeansModel(UnsupervisedModel):
    """K-Means clustering model"""
    
    def _build_model(self):
        from sklearn.cluster import KMeans
        
        default_params = {
            'n_clusters': 8,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'random_state': 42
        }
        
        params = {**default_params, **self.config.parameters}
        return KMeans(**params)
    
    def _train_model(self, X: pd.DataFrame, y=None, **kwargs) -> Dict[str, Any]:
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        
        # Get cluster labels and centers
        labels = self.model.labels_
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, labels)
        inertia = self.model.inertia_
        
        # Cluster statistics
        cluster_counts = pd.Series(labels).value_counts().to_dict()
        
        history = {
            'n_clusters': self.model.n_clusters,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'cluster_counts': cluster_counts,
            'cluster_centers': {f'cluster_{i}': dict(zip(X.columns, center)) 
                              for i, center in enumerate(centers)},
            'n_iterations': self.model.n_iter_
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict cluster labels for new data"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_cluster_centers(self) -> pd.DataFrame:
        """Get cluster centers in original feature space"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        return pd.DataFrame(centers, columns=self.feature_names)


class GMMModel(UnsupervisedModel):
    """Gaussian Mixture Model for probabilistic clustering"""
    
    def _build_model(self):
        from sklearn.mixture import GaussianMixture
        
        default_params = {
            'n_components': 8,
            'covariance_type': 'full',
            'max_iter': 100,
            'random_state': 42
        }
        
        params = {**default_params, **self.config.parameters}
        return GaussianMixture(**params)
    
    def _train_model(self, X: pd.DataFrame, y=None, **kwargs) -> Dict[str, Any]:
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        
        # Get cluster labels and probabilities
        labels = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, labels)
        aic = self.model.aic(X_scaled)
        bic = self.model.bic(X_scaled)
        log_likelihood = self.model.score(X_scaled)
        
        # Cluster statistics
        cluster_counts = pd.Series(labels).value_counts().to_dict()
        
        history = {
            'n_components': self.model.n_components,
            'silhouette_score': silhouette_avg,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'cluster_counts': cluster_counts,
            'converged': self.model.converged_,
            'n_iterations': self.model.n_iter_,
            'cluster_weights': self.model.weights_.tolist()
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict cluster labels for new data"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster membership probabilities"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def sample(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate samples from the fitted mixture model"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        samples_scaled, labels = self.model.sample(n_samples)
        samples = self.scaler.inverse_transform(samples_scaled)
        
        df = pd.DataFrame(samples, columns=self.feature_names)
        df['cluster'] = labels
        return df


class HMMModel(UnsupervisedModel):
    """Hidden Markov Model for sequential pattern discovery"""
    
    def _build_model(self):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ModelError("hmmlearn not installed. Install with: pip install hmmlearn")
        
        default_params = {
            'n_components': 3,
            'covariance_type': 'full',
            'n_iter': 100,
            'random_state': 42
        }
        
        params = {**default_params, **self.config.parameters}
        return GaussianHMM(**params)
    
    def _train_model(self, X: pd.DataFrame, y=None, **kwargs) -> Dict[str, Any]:
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences (assuming each row is a time step)
        lengths = kwargs.get('sequence_lengths', [len(X_scaled)])
        
        # Fit the model
        self.model.fit(X_scaled, lengths)
        
        # Get hidden states
        hidden_states = self.model.predict(X_scaled, lengths)
        
        # Calculate metrics
        log_likelihood = self.model.score(X_scaled, lengths)
        
        # State statistics
        state_counts = pd.Series(hidden_states).value_counts().to_dict()
        
        # Transition matrix
        transition_matrix = self.model.transmat_.tolist()
        
        history = {
            'n_components': self.model.n_components,
            'log_likelihood': log_likelihood,
            'state_counts': state_counts,
            'transition_matrix': transition_matrix,
            'start_probabilities': self.model.startprob_.tolist(),
            'converged': self.model.monitor_.converged,
            'n_iterations': self.model.monitor_.iter
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict hidden states for new data"""
        X_scaled = self.scaler.transform(X)
        lengths = kwargs.get('sequence_lengths', [len(X_scaled)])
        return self.model.predict(X_scaled, lengths)
    
    def decode(self, X: pd.DataFrame, **kwargs) -> tuple:
        """Find most likely state sequence using Viterbi algorithm"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        X_scaled = self.scaler.transform(X)
        lengths = kwargs.get('sequence_lengths', [len(X_scaled)])
        log_prob, states = self.model.decode(X_scaled, lengths)
        
        return log_prob, states
    
    def sample(self, n_samples: int = 100) -> tuple:
        """Generate samples from the HMM"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        samples_scaled, states = self.model.sample(n_samples)
        samples = self.scaler.inverse_transform(samples_scaled)
        
        df = pd.DataFrame(samples, columns=self.feature_names)
        return df, states
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get the transition matrix as a DataFrame"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        states = [f"State_{i}" for i in range(self.model.n_components)]
        return pd.DataFrame(
            self.model.transmat_,
            index=states,
            columns=states
        )
    
    def get_emission_parameters(self) -> Dict[str, Any]:
        """Get emission distribution parameters"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        return {
            'means': self.model.means_.tolist(),
            'covariances': self.model.covars_.tolist()
        }