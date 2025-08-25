"""
Base classes for the Ramanujan ML framework
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import logging
from datetime import datetime

from .config import ModelConfig, TrainingConfig, PredictionConfig
from .exceptions import ModelError, TrainingError, PredictionError


class BaseModel(ABC):
    """Abstract base class for all models in the Ramanujan framework"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.target_name = None
        self.model_metadata = {
            "model_type": config.model_type.value,
            "model_name": config.name,
            "created_at": datetime.now(),
            "trained_at": None,
            "version": "1.0.0"
        }
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the underlying model with given configuration"""
        pass
    
    @abstractmethod
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model and return training history"""
        pass
    
    @abstractmethod
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions using the trained model"""
        pass
    
    def train(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None, 
              training_config: Optional[TrainingConfig] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model with given data
        
        Args:
            X: Feature data
            y: Target data (for supervised models)
            training_config: Training configuration
            **kwargs: Additional training parameters
            
        Returns:
            Training history and metrics
        """
        try:
            if self.model is None:
                self.model = self._build_model()
            
            # Store feature and target information
            self.feature_names = list(X.columns)
            if y is not None:
                self.target_name = y.name if hasattr(y, 'name') else 'target'
            
            # Validate data
            self._validate_training_data(X, y)
            
            # Train the model
            self.logger.info(f"Starting training for {self.config.name}")
            training_history = self._train_model(X, y, **kwargs)
            
            self.is_trained = True
            self.training_history = training_history
            self.model_metadata["trained_at"] = datetime.now()
            
            self.logger.info(f"Training completed for {self.config.name}")
            return training_history
            
        except Exception as e:
            raise TrainingError(f"Training failed for {self.config.name}: {str(e)}")
    
    def predict(self, X: pd.DataFrame, config: Optional[PredictionConfig] = None) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature data for prediction
            config: Prediction configuration
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise PredictionError(f"Model {self.config.name} is not trained")
        
        try:
            # Validate prediction data
            self._validate_prediction_data(X)
            
            predictions = self._predict(X, **(config.to_dict() if config else {}))
            return predictions
            
        except Exception as e:
            raise PredictionError(f"Prediction failed for {self.config.name}: {str(e)}")
    
    def save_model(self, file_path: Union[str, Path], format: str = "joblib"):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ModelError("Cannot save untrained model")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "config": self.config,
            "metadata": self.model_metadata,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "training_history": self.training_history
        }
        
        if format.lower() == "joblib":
            joblib.dump(model_data, file_path)
        elif format.lower() == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            raise ValueError(f"Unsupported save format: {format}")
        
        self.logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: Union[str, Path], format: str = "joblib"):
        """Load a trained model from disk"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        if format.lower() == "joblib":
            model_data = joblib.load(file_path)
        elif format.lower() == "pickle":
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported load format: {format}")
        
        self.model = model_data["model"]
        self.config = model_data["config"]
        self.model_metadata = model_data["metadata"]
        self.feature_names = model_data["feature_names"]
        self.target_name = model_data["target_name"]
        self.training_history = model_data["training_history"]
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        return {
            "config": self.config,
            "metadata": self.model_metadata,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "training_history": self.training_history
        }
    
    def _validate_training_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Validate training data"""
        if X.empty:
            raise TrainingError("Training data is empty")
        
        if X.isnull().all().any():
            raise TrainingError("Training data contains columns with all null values")
        
        # Check for supervised learning requirements
        if self._requires_target() and y is None:
            raise TrainingError(f"Model {self.config.model_type} requires target data")
        
        if y is not None and len(X) != len(y):
            raise TrainingError("Feature and target data must have same length")
    
    def _validate_prediction_data(self, X: pd.DataFrame):
        """Validate prediction data"""
        if X.empty:
            raise PredictionError("Prediction data is empty")
        
        if self.feature_names and list(X.columns) != self.feature_names:
            raise PredictionError("Prediction data features don't match training features")
    
    def _requires_target(self) -> bool:
        """Check if this model type requires target data"""
        supervised_models = [
            "xgboost", "lightgbm", "catboost", "random_forest",
            "linear_regression", "ridge_regression", "lstm", "garch"
        ]
        return self.config.model_type.value in supervised_models


class SupervisedModel(BaseModel):
    """Base class for supervised learning models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.metrics = {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate the model on given data"""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        predictions = self.predict(X)
        
        if metrics is None:
            metrics = ["rmse", "mae", "r2"]
        
        results = {}
        for metric in metrics:
            results[metric] = self._calculate_metric(y, predictions, metric)
        
        return results
    
    def _calculate_metric(self, y_true: pd.Series, y_pred: np.ndarray, metric: str) -> float:
        """Calculate a specific metric"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        if metric.lower() == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric.lower() == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric.lower() == "r2":
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")


class UnsupervisedModel(BaseModel):
    """Base class for unsupervised learning models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the model and return predictions in one step"""
        self.train(X)
        return self.predict(X)


class CorrelationModel(BaseModel):
    """Base class for correlation analysis models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def analyze(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze correlations and return results"""
        if not self.is_trained:
            self.train(X, Y)
        
        return self.predict(X, Y)