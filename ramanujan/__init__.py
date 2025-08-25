"""
Ramanujan - ML Prediction Framework

A comprehensive machine learning framework for financial prediction and analysis,
providing unified interfaces for training, tuning, and predicting with various models.
"""

from .core import ModelFramework
from .models import (
    # Prediction Models
    XGBoostModel,
    LightGBMModel, 
    CatBoostModel,
    RandomForestModel,
    LinearRegressionModel,
    RidgeRegressionModel,
    LSTMModel,
    GARCHModel,
    
    # Clustering Models
    KMeansModel,
    GMMModel,
    HMMModel,
    
    # Correlation Models
    PearsonCorrelationModel,
    SpearmanCorrelationModel,
    KendallCorrelationModel,
    TailDependenceModel,
    MutualInformationModel
)
from .config import ModelConfig, TrainingConfig
from .exceptions import RamanujanError, ModelError, TrainingError

__version__ = "1.0.0"
__all__ = [
    "ModelFramework",
    # Prediction Models
    "XGBoostModel", "LightGBMModel", "CatBoostModel", "RandomForestModel",
    "LinearRegressionModel", "RidgeRegressionModel", "LSTMModel", "GARCHModel",
    # Clustering Models 
    "KMeansModel", "GMMModel", "HMMModel",
    # Correlation Models
    "PearsonCorrelationModel", "SpearmanCorrelationModel", "KendallCorrelationModel",
    "TailDependenceModel", "MutualInformationModel",
    # Configuration
    "ModelConfig", "TrainingConfig",
    # Exceptions
    "RamanujanError", "ModelError", "TrainingError"
]